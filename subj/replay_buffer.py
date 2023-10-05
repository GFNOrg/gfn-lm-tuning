import torch
import heapq
import random
import pickle
import gzip
import numpy as np

import editdistance

class ReplayBuffer:
    '''
    A relay buffer that uses a heap to keep the max_size items with the highest reward
    '''
    def __init__(self, max_len, eos_token_id, sim_tolerance=0.25):
        self.max_len = max_len
        self.eos_token_id = eos_token_id
        self.sim_tolerance = sim_tolerance
        self.reset()

    def reset(self):
        self._buffer = {}
    
    def add(self, item):
        '''
        add an item to the buffer, where item = [log reward, tensor of shape (seq_len, )]
        '''
        # if item is already in the buffer, skip it
        # import pdb; pdb.set_trace();
        str_query_answer = item['str_query_answer']
        if item['str_rationale'] in self._buffer[str_query_answer]['exists']:
            return
        # if the edit distance between item and any item in the buffer is small, skip it
        tokenized_rationale = [x for x in item['tensor_rationale'].tolist() if x != self.eos_token_id]
        for buffer_item in self._buffer[str_query_answer]['rationales']:
            tokenized_existing_rationale = [x for x in buffer_item[2].tolist() if x != self.eos_token_id]
            if editdistance.eval(tokenized_rationale, tokenized_existing_rationale) < (len(tokenized_rationale) + len(tokenized_existing_rationale))*self.sim_tolerance:
                if buffer_item[0] >= item['logreward']:
                    return
                else:
                    self._buffer[str_query_answer]['exists'].remove(buffer_item[1])
                    self._buffer[str_query_answer]['rationales'].remove(buffer_item)
                    heapq.heapify(self._buffer[str_query_answer]['rationales'])
                    self._buffer[str_query_answer]['exists'].add(item['str_rationale'])
                    heapq.heappush(self._buffer[str_query_answer]['rationales'],
                        (item['logreward'], item['str_rationale'], item['tensor_rationale'], item['full_logrewards']))
                    return
        self._buffer[str_query_answer]['exists'].add(item['str_rationale'])
        if len(self._buffer[str_query_answer]['rationales']) >= self.max_len:
            popped = heapq.heappushpop(self._buffer[str_query_answer]['rationales'],
                                       (item['logreward'], item['str_rationale'], item['tensor_rationale'], item['full_logrewards']))
            self._buffer[str_query_answer]['exists'].remove(popped[1])
        else:
            heapq.heappush(self._buffer[str_query_answer]['rationales'],
                           (item['logreward'], item['str_rationale'], item['tensor_rationale'], item['full_logrewards']))

    def add_batch(self, query, answer, rationales, logrewards, tokenizer):
        '''
        add a batch of items to the buffer
        '''
        str_query = ' '.join([str(x) for x in query.tolist()])
        if answer is not None:
            str_answer = ' '.join([str(x) for x in answer.tolist()])
        else:
            str_answer = 'None'
        str_query_answer = '|'.join([str_query, str_answer])
        if str_query_answer not in self._buffer:
            self._buffer[str_query_answer] = {'tensor_query': query,
                                              'tensor_answer': answer,
                                              'rationales': [],
                                              'exists': set()}
        rationales[(rationales == self.eos_token_id).cumsum(dim=-1) >= 1] = self.eos_token_id
        token_rationales = tokenizer.batch_decode(rationales)
        for i in range(rationales.size(0)):
            #str_rationale = ' '.join([str(x) for x in rationales[i].tolist() if x != self.eos_token_id])
            str_rationale = token_rationales[i].replace('<|endoftext|>', '').strip()
            self.add({'logreward': logrewards[i, (rationales[i] != self.eos_token_id).sum()].item(),
                      'str_query_answer': str_query_answer,
                      'str_rationale': str_rationale,
                      'tensor_rationale': rationales[i],
                      'full_logrewards': logrewards[i, :]})
    
    def sample(self, batch_size, query, answer):
        '''
        uniformly sample a batch of items from the buffer,
        and return a stacked tensor
        '''
        str_query = ' '.join([str(x) for x in query.tolist()])
        if answer is not None:
            str_answer = ' '.join([str(x) for x in answer.tolist()])
        else:
            str_answer = 'None'
        str_query_answer = '|'.join([str_query, str_answer])
        if str_query_answer not in self._buffer:
            return None, None
        query_answer_buffer = self._buffer[str_query_answer]['rationales']
        if len(query_answer_buffer) < batch_size:
            # if the buffer is not full, use pad_sequence and return all items
            return torch.nn.utils.rnn.pad_sequence([item[2] for item in query_answer_buffer],
                                                   batch_first=True,
                                                   padding_value=self.eos_token_id), \
                   torch.nn.utils.rnn.pad_sequence([item[3] for item in query_answer_buffer],
                                                   batch_first=True,
                                                   padding_value=0)
        else:
            # do prioritized sampling
            priorities  = [item[0] for item in query_answer_buffer]
            priorities = np.array(priorities)
            priorities = priorities - np.max(priorities)
            idx = np.random.choice(len(query_answer_buffer), batch_size, p=np.exp(priorities)/np.sum(np.exp(priorities)), replace=True)
            return torch.nn.utils.rnn.pad_sequence([query_answer_buffer[i][2] for i in idx],
                                                   batch_first=True,
                                                   padding_value=self.eos_token_id), \
                   torch.nn.utils.rnn.pad_sequence([query_answer_buffer[i][3] for i in idx],
                                                   batch_first=True,
                                                   padding_value=0)
    def print(self):
        for key in self._buffer:
            print(key)
            for item in self._buffer[key]['rationales']:
                print(item[1])
            print('')
    
    def save(self, path):
        with gzip.open(path, 'wb') as f:
            pickle.dump(self._buffer, f)