import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import score_fast, lora_to_base, base_to_lora


def get_reward(rewards, rewards_args):
    # reward is a list with the names of rewards to consider
    if len(rewards) == 1:
        return globals()[rewards[0]](**rewards_args[0])
    else:
        rews = []
        for r, args in zip(rewards, rewards_args):
            rews.append(globals()[r](**args))
        return MultiObjective(rews)


class Reward:
    def score(self, input_batch):
        # input_batch is shape b x ... 
        # output is shape b
        raise NotImplementedError("")


class FrozenModelWithPrompt(Reward):
    def __init__(self, model, prompt, eos_token_id, temperature=1., min_len=0, vocab_alpha=-50.,
                 solution_beta=1., vocab_nice_mask=None, vocab_naughty_mask=None, len_beta=0):
        self.model = model
        self.model.eval()
        self.temperature = temperature
        self.eos_token_id = eos_token_id
        self.solution_beta = solution_beta
        self.vocab_nice_mask = vocab_nice_mask
        self.vocab_naughty_mask = vocab_naughty_mask
        self.vocab_alpha = vocab_alpha
        self.min_len = min_len
        self.len_beta = len_beta
        self.prompt = prompt
        # self.prompt_cache = None
        self.prompt_cache = self.model(self.prompt.cuda(), use_cache=True).past_key_values


    def score(self, input_batch, skip_first, solution_len):
        # self.model.eval()
        #logprob = self.model(input_batch).logits[:,:-1].log_softmax(-1)
        #token_ids = input_batch[:, 1:].unsqueeze(-1)
        #logPF = logprob.gather(-1, token_ids).squeeze(-1)
        # self.model.train()
        #return logPF[:, self.skip_first-1:].sum(dim=-1) * (1./self.temperature)
        lora_to_base(self.model)
        with torch.no_grad():
            res = score_fast(self.model,
                            input_batch,
                            self.eos_token_id,
                            skip_first=skip_first,
                            solution_len=solution_len,
                            solution_beta=self.solution_beta,
                            vocab_nice_mask=self.vocab_nice_mask,
                            vocab_naughty_mask=self.vocab_naughty_mask,
                            vocab_alpha=self.vocab_alpha,
                            min_len=self.min_len,
                            len_beta=self.len_beta, 
                            prompt_cache=(self.prompt.cuda(), self.prompt_cache)) * (1./self.temperature)
        base_to_lora(self.model)
        return res


class FrozenModel(Reward):
    def __init__(self, model, eos_token_id, temperature=1., min_len=0, vocab_alpha=-50.,
                 solution_beta=1., cot_beta=1., vocab_nice_mask=None, vocab_naughty_mask=None, len_beta=0):
        self.model = model
        self.model.eval()
        self.temperature = temperature
        self.eos_token_id = eos_token_id
        self.solution_beta = solution_beta
        self.cot_beta = cot_beta
        self.vocab_nice_mask = vocab_nice_mask
        self.vocab_naughty_mask = vocab_naughty_mask
        self.vocab_alpha = vocab_alpha
        self.min_len = min_len
        self.len_beta = len_beta
    def score(self, input_batch, skip_first, solution_len):
        # self.model.eval()
        #logprob = self.model(input_batch).logits[:,:-1].log_softmax(-1)
        #token_ids = input_batch[:, 1:].unsqueeze(-1)
        #logPF = logprob.gather(-1, token_ids).squeeze(-1)
        # self.model.train()
        #return logPF[:, self.skip_first-1:].sum(dim=-1) * (1./self.temperature)
        lora_to_base(self.model)
        with torch.no_grad():
            res = score_fast(self.model,
                            input_batch,
                            self.eos_token_id,
                            skip_first=skip_first,
                            solution_len=solution_len,
                            solution_beta=self.solution_beta,
                            cot_beta=self.cot_beta,
                            vocab_nice_mask=self.vocab_nice_mask,
                            vocab_naughty_mask=self.vocab_naughty_mask,
                            vocab_alpha=self.vocab_alpha,
                            min_len=self.min_len,
                            len_beta=self.len_beta) * (1./self.temperature)
        base_to_lora(self.model)
        return res

class HFModel(Reward):
    def __init__(self, name):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name)

    def score(self, input_batch):
        # TODO: Implement
        
        pass 




class MultiObjective:
    def __init__(self, rewards):
        self.rewards = rewards
    
    def score(self, input_batch):
        # input_batch is shape b x ... 
        # output is shape b x d (d is number of rewards)
        return torch.cat([r(input_batch).unsqueeze(-1) for r in self.rewards])
