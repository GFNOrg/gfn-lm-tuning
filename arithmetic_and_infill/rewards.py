import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import score_fast, lora_to_base, base_to_lora
from hydra.utils import to_absolute_path
import json
from omegaconf import OmegaConf

def get_reward(rewards, rewards_args):
    # reward is a list with the names of rewards to consider
    return globals()[rewards[0]](**rewards_args[0])


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

def load_reward(config, model, tokenizer, dataset, min_len=0):
    if config.impose_vocab_constraint:
        def check_str(string, allow_char_set):
            return all([c in allow_char_set for c in string])

        allowed_indices = torch.tensor([item[1] for item in sorted(tokenizer.vocab.items(), key=lambda x: x[1]) if check_str(item[0], '▁Ġ,.1234567890=+-')], device='cuda', dtype=torch.long)
        vocab_nice_list = torch.zeros(len(tokenizer.vocab), device='cuda').bool()
        vocab_nice_list[allowed_indices] = True
        vocab_nice_list[tokenizer.eos_token_id] = True
    else: 
        vocab_nice_list = None

    rew_config = OmegaConf.to_container(config.reward_config, resolve=True)
    rew_config["model"] = model
    rew_config["eos_token_id"] = tokenizer.eos_token_id
    rew_config["min_len"] = min_len
    rew_config["vocab_nice_mask"] = vocab_nice_list

    if config.reward_name=="FrozenModelWithPrompt" and "prompt_data" in rew_config:
        data_path = to_absolute_path(config.reward_config.prompt_data.path)
        data = json.load(open(data_path, 'r'))
        train_queries = []
        train_rationales = []
        train_sols = []
        for sample in data:
            train_queries.append(sample['str_query'])
            train_rationales.append(sample['str_rationale'])
            train_sols.append(sample['str_sol'])
        rew_config.pop("prompt_data")
        prompt_text = '\n'.join([train_queries[i]+ " " + train_rationales[i]+train_sols[i] for i in range(config.reward_config.prompt_data.num_points)]) + '\n'
        print("Prompt text: ", prompt_text)
        rew_config["prompt"] = tokenizer(prompt_text, return_tensors='pt')['input_ids'].cuda()
    
    elif config.reward_name=="FrozenModelWithPrompt":
        rew_config["prompt"] = tokenizer(rew_config["prompt"], return_tensors='pt')['input_ids'].cuda()
    
    rew = get_reward([config.reward_name], [rew_config])
    return rew, vocab_nice_list