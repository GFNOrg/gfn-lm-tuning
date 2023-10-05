from collections import Counter
import torch
import configparser
import argparse
from torch.distributions import Categorical
import math
import itertools

EQUAL_TOK = 796

def lora_to_base(model):
    model.base_model.disable_adapter_layers()
    model.eval()
    
def base_to_lora(model):
    model.base_model.enable_adapter_layers()
    model.train()

def generate(model, encoded_prompt, eos_token_id, max_len=10, temperature=1., vocab_nice_mask=None,
                                    vocab_naughty_mask=None, vocab_alpha=-99, top_k=999999, top_p=1., tokenizer=None, use_tools=False):
    active_seqs = torch.ones(encoded_prompt.size(0)).bool().to(encoded_prompt.device)
    logPF = encoded_prompt.new_zeros(encoded_prompt.size(0)).float()
    actions = encoded_prompt.clone()
    state = encoded_prompt.clone()
    past_key_values = model(state[:, :-1])['past_key_values']
    for i in range(max_len):
        if use_tools and i > 0:
            output = model(state, attention_mask=generated != eos_token_id, position_ids=pos)
        else:
            output = model(state[:, -1:], past_key_values=past_key_values)
        past_key_values = output['past_key_values']
        with torch.no_grad():
            prob = (output['logits'][:, -1, :]).softmax(dim=-1)
            modified_logits = output['logits'][:, -1, :].clone()
            # implement top-k by getting the top-k largest values and setting the rest to 0
            if top_k < 999999:
                modified_logits[prob >= prob.topk(top_k)] = -float('inf')
            # implement top-p by getting indices in the top-p prob mass and setting the rest to 0
            if top_p < 1.:
                sorted_probs, indices = torch.sort(prob, dim=-1, descending=True)
                cumsum_prob = torch.cumsum(sorted_probs, dim=-1)
                nucleus = cumsum_prob < top_p
                nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
                modified_logits[~nucleus] = -float('inf')
            if vocab_nice_mask is not None:
                # add vocab_alpha to the logits of the unmasked vocab items
                modified_logits[:, ~vocab_nice_mask] += vocab_alpha
            prob = (modified_logits/temperature).softmax(dim=-1)
            token_ids = torch.multinomial(prob, num_samples=1)
        token_ids = torch.where(active_seqs.unsqueeze(-1), token_ids, torch.ones_like(token_ids) * eos_token_id)
        logprob = output['logits'][:, -1, :].log_softmax(dim=-1)
        logPF += logprob.gather(-1, token_ids).squeeze(-1)
        actions = torch.cat([actions, token_ids], dim=-1)
        state = torch.cat([state, token_ids], dim=-1)
        if use_tools:
            new_str = []
            for j, tok in enumerate(token_ids):
                if tok[0] == EQUAL_TOK:
                    expr = tokenizer.batch_decode([state[j]])[0]
                    to_eval = expr.split("Answer:")[-1].split(',')[-1].split(".")[-1].split('=')[0]
                    try:
                        val = eval(to_eval)
                    except:
                        new_str.append(state[j])
                        continue
                        # raise ValueError(f'Failed to evaluate {to_eval}')
                    # new_str.append(expr + " " + str(val))
                    new_str.append(tokenizer.batch_encode_plus([expr + " " + str(val)], return_tensors='pt')['input_ids'].cuda()[0])
                else:
                    new_str.append(state[j])
            state, pos, _ = remove_eos_and_pad_left(new_str, eos_token_id, eos_token_id)
            state = state.cuda()
            pos = pos.cuda()
        # check if all sequences have generated eos
        active_seqs = active_seqs * (token_ids != eos_token_id).squeeze(-1)
        if torch.all(~active_seqs):
            break
    return actions, logPF, state

def generate_and_return_eos_logprob(model, encoded_prompt, eos_token_id, reward_fn, vocab_nice_mask=None,
                                    vocab_naughty_mask=None, vocab_alpha=-99, max_len=10, min_len=0,
                                    temperature=1., top_k=999999, top_p=1., action_seq=None, skip_rewards=False,
                                    tokenizer=None, use_tools=False):
    # generate and return the probability of generating eos at every step
    active_seqs = torch.ones(encoded_prompt.size(0)).bool().to(encoded_prompt.device)
    actions = encoded_prompt.clone()
    state = encoded_prompt.clone()
    logpf = encoded_prompt.new_zeros(encoded_prompt.size(0), max_len+1).float()
    logeosprobs = encoded_prompt.new_zeros(encoded_prompt.size(0), max_len+1).float()
    logrewards = encoded_prompt.new_zeros(encoded_prompt.size(0), max_len+1).float()
    if not skip_rewards:
        logrewards[:, 0] = reward_fn(state)
    past_key_values = model(state[:, :-1])['past_key_values']
    for i in range(max_len):
        if use_tools and i > 0:
            output = model(state, attention_mask=state != eos_token_id, position_ids=pos)
        else:
            output = model(state[:, -1:], past_key_values=past_key_values)
        past_key_values = output['past_key_values']
        if action_seq is None:
            with torch.no_grad():
                prob = (output['logits'][:, -1, :]).softmax(dim=-1)
                modified_logits = output['logits'][:, -1, :].clone().detach()
                # implement top-k by getting the top-k largest values and setting the rest to 0
                if top_k < 999999:
                    modified_logits[prob >= prob.topk(top_k)] = -float('inf')
                # implement top-p by getting indices in the top-p prob mass and setting the rest to 0
                if top_p < 1.:
                    sorted_probs, indices = torch.sort(prob, dim=-1, descending=True)
                    cumsum_prob = torch.cumsum(sorted_probs, dim=-1)
                    nucleus = cumsum_prob < top_p
                    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
                    modified_logits[~nucleus] = -float('inf')
                # if we haven't reach the minimum length, set the probability of generating eos to 0
                if i < min_len:
                    modified_logits[:, eos_token_id] = -float('inf')
                if vocab_nice_mask is not None:
                    # add vocab_alpha to the logits of the unmasked vocab items
                    modified_logits[:, ~vocab_nice_mask] += vocab_alpha
                prob = (modified_logits/temperature).softmax(dim=-1)
                token_ids = torch.multinomial(prob, num_samples=1)
        else:
            if i >= action_seq.size(-1):
                token_ids = (torch.ones_like(action_seq[:, 0]) * eos_token_id).unsqueeze(-1)
            else:
                token_ids = action_seq[:, i].unsqueeze(-1)
        token_ids = torch.where(active_seqs.unsqueeze(-1), token_ids, torch.ones_like(token_ids) * eos_token_id)
        modified_logits = output['logits'][:, -1, :]
        if vocab_nice_mask is not None:
            modified_logits[:, ~vocab_nice_mask] += vocab_alpha
        logprob = modified_logits.log_softmax(dim=-1)
        #print(logprob[0, eos_token_id])
        #print(logprob.shape, token_ids.shape)
        #print(logprob.shape, encoded_prompt.shape)
        #print(actions.shape, output['logits'].shape)
        logpf[:, i] = torch.where(active_seqs, logprob.gather(-1, token_ids).squeeze(-1), 0)
        logeosprobs[:, i] = torch.where(active_seqs, logprob[:, eos_token_id], 0)
        actions = torch.cat([actions, token_ids], dim=-1)
        state = torch.cat([state, token_ids], dim=-1)
        if use_tools:
            new_str = []
            for j, tok in enumerate(token_ids):
                if tok[0] == EQUAL_TOK:
                    expr = tokenizer.batch_decode([state[j]])[0]
                    to_eval = expr.split("Answer:")[-1].split(',')[-1].split(".")[-1].split('=')[0]
                    try:
                        val = eval(to_eval)
                    except:
                        new_str.append(state[j])
                        continue
                        # raise ValueError(f'Failed to evaluate {to_eval}')
                    # new_str.append(expr + " " + str(val))
                    new_str.append(tokenizer.batch_encode_plus([expr + " " + str(val)], return_tensors='pt')['input_ids'].cuda()[0])
                else:
                    new_str.append(state[j])
            state, pos, _ = remove_eos_and_pad_left(new_str, eos_token_id, eos_token_id)
            state = state.cuda()
            pos = pos.cuda()
            # state = tokenizer.batch_encode_plus(new_str, return_tensors='pt')['input_ids'].cuda()
        if not skip_rewards:
            if use_tools:
                logrewards[:, i+1] = torch.where(active_seqs, reward_fn(state), 0)
            else:
                logrewards[:, i+1] = torch.where(active_seqs, reward_fn(actions), 0)
        # check if all sequences have generated eos
        active_seqs = active_seqs * (token_ids != eos_token_id).squeeze(-1)
        if torch.all(~active_seqs):
            break
    # add eos to the end of the sequence
    actions = torch.cat([actions, actions.new_ones(actions.size(0), 1)*eos_token_id], dim=-1)
    logpf[:, -1] = 0
    logeosprobs[:, -1] = 0
    return actions, logpf, logeosprobs, logrewards

def generate_and_return_terminate_logprob_v2(model, encoded_prompt, eos_token_id, reward_fn, vocab_nice_mask=None,
                                    vocab_alpha=-99, max_len=10, min_len=0, sep_token_id=None, tokenizer=None,
                                    temperature=1., top_k=999999, top_p=1., action_seq=None, skip_rewards=False):
    # generate and return the probability of generating eos_token_id on every step
    active_seqs = torch.ones(encoded_prompt.size(0)).bool().to(encoded_prompt.device)
    stmt_sets = [set() for _ in range(encoded_prompt.size(0))]
    cursors = [encoded_prompt.size(1) for _ in range(encoded_prompt.size(0))]
    new_input = encoded_prompt.clone()
    logpf = encoded_prompt.new_zeros(encoded_prompt.size(0), max_len+1).float()
    logeosprobs = encoded_prompt.new_zeros(encoded_prompt.size(0), max_len+1).float()
    logrewards = encoded_prompt.new_zeros(encoded_prompt.size(0), max_len+1).float()
    if not skip_rewards:
        logrewards[:, 0] = reward_fn(new_input)
    for i in range(max_len):
        output = model(new_input)
        if action_seq is None:
            with torch.no_grad():
                prob = (output['logits'][:, -1, :]).softmax(dim=-1)
                modified_logits = output['logits'][:, -1, :].clone().detach()
                # implement top-k by getting the top-k largest values and setting the rest to 0
                if top_k < 999999:
                    modified_logits[prob >= prob.topk(top_k)] = -float('inf')
                # implement top-p by getting indices in the top-p prob mass and setting the rest to 0
                if top_p < 1.:
                    sorted_probs, _ = torch.sort(prob, dim=-1, descending=True)
                    cumsum_prob = torch.cumsum(sorted_probs, dim=-1)
                    nucleus = cumsum_prob < top_p
                    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
                    modified_logits[~nucleus] = -float('inf')
                # if we haven't reach the minimum length, set the probability of generating eos to 0
                if i < min_len:
                    modified_logits[:, eos_token_id] = -float('inf')
                if vocab_nice_mask is not None:
                    # add vocab_alpha to the logits of the unmasked vocab items
                    modified_logits[:, ~vocab_nice_mask] += vocab_alpha
                if sep_token_id is not None:
                    # make sure we don't generate duplicate statements
                    for j, token_id in enumerate(token_ids):
                        if token_id == sep_token_id and tuple(new_input[j, cursors[j]:].tolist()) in stmt_sets[j]:
                            modified_logits[:, eos_token_id] = -float('inf')
                            modified_logits[:, sep_token_id] = -float('inf')
                prob = (modified_logits/temperature).softmax(dim=-1)
                token_ids = torch.multinomial(prob, num_samples=1)
        else:
            if i >= action_seq.size(-1):
                token_ids = (torch.ones_like(action_seq[:, 0]) * eos_token_id).unsqueeze(-1)
            else:
                token_ids = action_seq[:, i].unsqueeze(-1)
        token_ids = torch.where(active_seqs.unsqueeze(-1), token_ids, torch.ones_like(token_ids) * eos_token_id)
        if sep_token_id is not None:
            # check if we have generated the separator token
            for j, token_id in enumerate(token_ids):
                if token_id == sep_token_id or token_id == eos_token_id:
                    stmt_sets[j].add(tuple(new_input[j, cursors[j]:].tolist()))
                    cursors[j] = new_input.size(1) + 1
                    # replace the corresponding entry in new_input with the sorted sequences from the set
                    sorted_stmts = tuple([ele for stmt in sorted([x+(sep_token_id,) for x in list(stmt_sets[j])]) for ele in stmt])[:-1]
                    new_stmts = torch.tensor(sorted_stmts, device=encoded_prompt.device)
                    #print(tokenizer.batch_decode([new_stmts]))
                    new_input[j, encoded_prompt.size(1):encoded_prompt.size(1)+new_stmts.size(0)] = new_stmts
            #print(cursors)
        modified_logits = output['logits'][:, -1, :]
        if vocab_nice_mask is not None:
            modified_logits[:, ~vocab_nice_mask] += vocab_alpha
        logprob = modified_logits.log_softmax(dim=-1)
        logpf[:, i] = logprob.gather(-1, token_ids).squeeze(-1)
        logeosprobs[:, i] = logprob[:, eos_token_id]
        new_input = torch.cat([new_input, token_ids], dim=-1)
        if not skip_rewards:
            logrewards[:, i+1] = torch.where(active_seqs, reward_fn(new_input), 0)
        # check if all sequences have generated eos
        active_seqs = active_seqs * (token_ids != eos_token_id).squeeze(-1)
        if torch.all(~active_seqs):
            break
    # add eos to the end of the sequence
    new_input = torch.cat([new_input, new_input.new_ones(new_input.size(0), 1)*eos_token_id], dim=-1)
    logpf[:, -1] = 0
    logeosprobs[:, -1] = 0
    return new_input, logpf, logeosprobs, logrewards

def score(model, encoded_input, skip_first=1):
    raise NotImplementedError
    logPF = 0.
    for i in range(skip_first, encoded_input.size(-1)):
        output = model(encoded_input[:, :i])
        logprob = output['logits'][:, -1, :].log_softmax(dim=-1)
        token_id = encoded_input[:, i]
        logPF += logprob.gather(-1, token_id.unsqueeze(-1)).squeeze(-1)
    return logPF

def score_fast(model, encoded_input, eos_token_id, min_len=0,
               skip_first=1, solution_len=0, solution_beta=1., cot_beta=1.,
               vocab_nice_mask=None, vocab_naughty_mask=None,
               vocab_alpha=-99, len_beta=0, reduction='sum', tokenizer=None,
               prompt_cache=None):
    if prompt_cache is None:
        logits = model(encoded_input, attention_mask=encoded_input!=eos_token_id).logits[:,:-1,:]
    else:
        # prompt_cache[1] contains past_key_values which need to be reshaped to the right batch size from encoded_input
        batched_prompt_cache = tuple(tuple([prompt_cache[1][i][j].repeat(encoded_input.shape[0], 1, 1, 1) for j in range(len(prompt_cache[1][i]))]) for i in range(len(prompt_cache[1])))
        concat_input = torch.cat([prompt_cache[0].repeat(encoded_input.shape[0], 1), encoded_input], dim=-1)
        attention_mask = concat_input != eos_token_id
        logits = model(encoded_input, past_key_values=batched_prompt_cache, attention_mask=attention_mask).logits[:,:-1,:]
    # get rid of the first few tokens
    logits = logits[:, skip_first-1:]
    non_eos_mask = (encoded_input != eos_token_id)[:, skip_first:]
    # score the log probability of the input sequence while ignoring trailing eos
    if vocab_nice_mask is not None:
        # add vocab_alpha to the logits of the unmasked vocab items
        # get a mask for the rationale
        rat_mask = torch.concat([non_eos_mask[:, solution_len:],
                                 encoded_input.new_zeros(encoded_input.size(0), solution_len)], dim=-1)
        #print(tokenizer.batch_decode(encoded_input[:, skip_first:][rat_mask.bool()]))
        logits[(rat_mask.float().unsqueeze(-1) @ (~vocab_nice_mask).float().unsqueeze(0)).bool()] += vocab_alpha
        logprob = logits.log_softmax(-1)
    elif vocab_naughty_mask is not None:
        # add vocab_alpha to the logits of the masked vocab items
        # get a mask for the rationale
        rat_mask = torch.concat([non_eos_mask[:, solution_len:],
                                 encoded_input.new_zeros(encoded_input.size(0), solution_len)], dim=-1)
        logits[(rat_mask.float().unsqueeze(-1) @ vocab_naughty_mask.float().unsqueeze(0)).bool()] += vocab_alpha
        logprob = logits.log_softmax(-1)
    else:
        # all the vocab items are allowed
        logprob = logits.log_softmax(-1)
    token_ids = encoded_input[:, skip_first:].unsqueeze(-1)
    logPF = logprob.gather(-1, token_ids).squeeze(-1)
    #print(tokenizer.batch_decode(token_ids[0]))
    #print(logPF)
    #for tok, logp in zip(tokenizer.batch_decode(token_ids[0]), logPF[0]):
    #for tok, logp in zip(token_ids[0], logPF[0]):
        #pass
        #print(f'{tok}\t\t{logp.item():.3f}')
    # change the log probability of eos to 0
    logPF[encoded_input[:, skip_first:] == eos_token_id] = 0.
    if (solution_len > 0 and solution_beta != 1.) or cot_beta != 1.:
        # get a mask for the last solution_len tokens excluding eos
        mask = torch.concat([(logPF == 0.)[:, solution_len:], logPF.new_ones(logPF.size(0), solution_len)], dim=-1)
        # get the log probability of the solutions
        logPF_solution = (logPF*mask).sum(dim=-1)
        # get the log probability of the chain-of-thought
        logPF_cot = (logPF*(1-mask)).sum(dim=-1) #* solution_beta
        logPF_cot = logPF_cot / (non_eos_mask.sum(dim=-1) - solution_len + 1)**len_beta
        return logPF_solution*(1/solution_beta) + logPF_cot*(1/cot_beta)
    if reduction == 'sum':
        res = logPF.sum(dim=-1)
        res = torch.where((non_eos_mask.sum(dim=-1) - solution_len) < min_len, -99, res)
    else:
        res = logPF
    return res

@torch.no_grad()
def reward_keywords(model, encoded_input, keywords, reward, skip_first=1, gamma=1.):
    has_keywords = torch.isin(encoded_input[:, skip_first-1:], keywords).sum(dim=-1).bool()
    #if torch.any(has_keywords):
    #    import pdb; pdb.set_trace()
    return torch.where(has_keywords, reward + math.log(gamma), reward)


def append_eos(tokenizer, encoded_input):
    return torch.cat([encoded_input, encoded_input.new_ones(encoded_input.size(0), 1)*tokenizer.eos_token_id], dim=-1)

def append_seq(tokenizer, encoded_input, to_append):
    encoded_to_append = tokenizer(to_append, return_tensors='pt')['input_ids'].cuda()
    return torch.cat([encoded_input, encoded_to_append.repeat(encoded_input.size(0), 1)], dim=-1)

def append_sol_and_remove_eos(text : torch.Tensor, result : torch.Tensor, eos_token_id : int, pad_token_id : int):
    # remove anything after the first eos token and append the result
    # if there is no eos token, append the result
    # text is a torch tensor with the first dimension being the batch
    # result is a torch tensor with the first dimension being the batch
    # this is a vectorized implementation
    # returns a torch tensor with the first dimension being the batch
    # and the second dimension being the length of the sequence
    new_text = []
    for t, r in zip(text, result[:text.size(0)]):
        if eos_token_id not in t:
            new_text.append(t if r is None else torch.cat([t, r]))
            continue
        # find the first eos token
        t[(t == eos_token_id).cumsum(dim=-1) >= 1] = eos_token_id
        eos_ind = ((t == eos_token_id).cumsum(dim=-1) == 1).nonzero()[0]
        # remove the eos tokens from the result and shift the result to the left
        if r is not None:
            new_text.append(torch.cat([t[:eos_ind], r]))
        else:
            new_text.append(t[:eos_ind])
    return torch.nn.utils.rnn.pad_sequence(new_text, batch_first=True, padding_value=pad_token_id)


def left_pad_generate(model, encoded_prompt, position_ids, eos_token_id, pad_token_id, max_len=10, temperature=1., top_k=999999, top_p=1.):
    active_seqs = torch.ones(encoded_prompt.size(0)).bool().to(encoded_prompt.device)
    logPF = encoded_prompt.new_zeros(encoded_prompt.size(0)).float()
    new_input = encoded_prompt.clone()
    for _ in range(max_len):
        output = model(new_input, attention_mask=(new_input != pad_token_id).long(), position_ids=position_ids)
        with torch.no_grad():
            prob = (output['logits'][:, -1, :]).softmax(dim=-1)
            modified_logits = output['logits'][:, -1, :].clone()
            # implement top-k by getting the top-k largest values and setting the rest to 0
            if top_k < 999999:
                modified_logits[prob >= prob.topk(top_k)] = -float('inf')
            # implement top-p by getting indices in the top-p prob mass and setting the rest to 0
            if top_p < 1.:
                sorted_probs, indices = torch.sort(prob, dim=-1, descending=True)
                cumsum_prob = torch.cumsum(sorted_probs, dim=-1)
                nucleus = cumsum_prob < top_p
                nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
                modified_logits[~nucleus] = -float('inf')
            prob = (modified_logits/temperature).softmax(dim=-1)
            token_ids = torch.multinomial(prob, num_samples=1)
        logprob = output['logits'][:, -1, :].log_softmax(dim=-1)
        logPF += logprob.gather(-1, token_ids).squeeze(-1)
        new_input = torch.cat([new_input, token_ids], dim=-1)
        position_ids = torch.cat([position_ids, position_ids[:, -1:] + 1], dim=-1)
        active_seqs = active_seqs * (token_ids != eos_token_id).squeeze(-1)
        if torch.all(~active_seqs):
            break
    return new_input, logPF


def remove_eos_and_pad_left(text : torch.Tensor, eos_token_id : int, pad_token_id : int):
    """
    remove anything after the first eos token, and left pad sequences"""
    stripped_text = []
    lens = []
    position_ids = []
    for t in text:
        if eos_token_id not in t:
            stripped_text.append(t)
            position_ids.append(torch.arange(t.size(-1)))
            lens.append(t.size(-1))
            continue
        # find the first eos token
        t[(t == eos_token_id).cumsum(dim=-1) >= 1] = eos_token_id
        eos_ind = ((t == eos_token_id).cumsum(dim=-1) == 1).nonzero()[0]
        stripped_text.append(t[:eos_ind])
        lens.append(eos_ind)
        position_ids.append(torch.arange(eos_ind.item()))
    left_pad_seqs = torch.nn.utils.rnn.pad_sequence([i.flip(0) for i in stripped_text], batch_first=True, padding_value=pad_token_id).flip(dims=[1]) 
    left_pad_position_ids = torch.nn.utils.rnn.pad_sequence([i.flip(0) for i in position_ids], batch_first=True, padding_value=0).flip(dims=[1])
    return left_pad_seqs, left_pad_position_ids, torch.tensor(lens)

def remove_left_pad(text : torch.Tensor, eos_token_id : int, pad_token_id : int):
    """
    remove left padding and pad to the right"""
    stripped_text = []
    for t in text:
        # t[(t != eos_token_id).cumsum(dim=-1) >= 1] = eos_token_id
        # neos_ind = ((t != eos_token_id).cumsum(dim=-1) == 1).nonzero()[0]
        stripped_text.append(t[(t != eos_token_id).cumsum(dim=-1) >= 1])
        # stripped_text.append(torch.cat([t[-l:], t.new_ones(t.size(0), t.size(1)-l)*pad_token_id], dim=-1))
    # print(stripped_text)
    return torch.nn.utils.rnn.pad_sequence(stripped_text, batch_first=True, padding_value=pad_token_id)


def evaluate_generation(config, model, encoded_input, eos_token_id, pad_token_id, max_len=10, temperature=1.):
    generated_text, _ = generate(model,
                                    encoded_input.repeat(config.eval_bsz, 1),
                                    eos_token_id=eos_token_id,
                                    max_len=max_len,
                                    temperature=temperature)
    seq, position_ids, _ = remove_eos_and_pad_left(generated_text, eos_token_id, pad_token_id)
    lora_to_base(model)
    completed_text, _ = left_pad_generate(model, seq.to(encoded_input), position_ids.to(encoded_input.device), eos_token_id, pad_token_id, max_len=max_len, temperature=temperature)
    base_to_lora(model)
    completed_text = remove_left_pad(completed_text, eos_token_id, pad_token_id)
    return completed_text

def extract_answer(str_answer):
    try:
        try:
            extracted_answer = int(str_answer.split('=')[-1].strip().strip('.'))
        except:
            extracted_answer = int(str_answer.split()[-1].strip().strip('.'))
    except:
        extracted_answer = -1
    return extracted_answer

def check_answer(str_answer, num_sol):
    extracted_answer = extract_answer(str_answer)
    if num_sol == extracted_answer:
        return True
    return False

def run_evaluation(model, tokenizer, encoded_test_queries, test_num_sols, eos_token_id, pad_token_id, vocab_nice_list, max_eval_len, num_samples, use_tools):
    base_to_lora(model)
    model.eval()
    test_correct = 0
    test_total = 0
    for query_ind in range(100): #len(encoded_test_queries)):
        encoded_input = encoded_test_queries[query_ind]
        generated_text, _, full_state = generate(model,
                                encoded_input.repeat(num_samples, 1),
                                eos_token_id=eos_token_id,
                                vocab_nice_mask=vocab_nice_list,
                                max_len=max_eval_len,
                                temperature=.1)
        decoded_answers = tokenizer.batch_decode(append_sol_and_remove_eos(full_state if use_tools else generated_text, [None,] * generated_text.size(0), eos_token_id, pad_token_id))
        extracted_answers = [extract_answer(i) for i in decoded_answers]
        answer = Counter(extracted_answers).most_common(1)[0][0]
        # if check_answer(decoded_answer, test_num_sols[query_ind]):
        if answer == test_num_sols[query_ind]:
            test_correct += 1
        test_total += 1
        # print(decoded_answer)
    return test_correct / test_total

def parse_argument():
    config = configparser.ConfigParser()
    parser = argparse.ArgumentParser(prog="Train")
    parser.add_argument("-c", "--config", dest="config_file",
                        help="Pass a generation config file", metavar="FILE")
    args = parser.parse_args()
    config.read(args.config_file)
    return config
