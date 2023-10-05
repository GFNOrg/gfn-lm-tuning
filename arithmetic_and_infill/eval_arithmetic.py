import hydra
import torch
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from hydra.utils import to_absolute_path
import wandb
from omegaconf import OmegaConf
import json
import os
from rich.table import Table
from rich import print
from rich.console import Console

from utils import generate, generate_and_return_eos_logprob, append_sol_and_remove_eos, base_to_lora, lora_to_base, run_evaluation, score_fast
from rewards import load_reward
from replay_buffer import load_replay_buffer
from data import load_dataset


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_table(queries, rationales, solutions, rewards):
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Query", width=30)
    table.add_column("Rationale", width=30)
    table.add_column("Solution", width=30) if solutions is not None else None
    table.add_column("Reward", width=10)
    # import pdb; pdb.set_trace();
    for query, rationale, solution, reward in zip(queries, rationales, solutions, rewards):
        table.add_row(query, rationale[len(query):], solution, str(reward))
    print(table)

def generate_examples(model, tokenizer, encoded_input, eos_token_id, pad_token_id, vocab_nice_list, rew, max_eval_len, config):
    generated_tokens, _, full_state = generate(model,
                            encoded_input.repeat(10, 1),
                            eos_token_id=eos_token_id,
                            vocab_nice_mask=vocab_nice_list,
                            max_len=max_eval_len,
                            temperature=config.eval_temp,
                            tokenizer=tokenizer,
                            use_tools=config.use_tools,
                            limit_capability=config.limit_capability,
                            operators=config.operators,
                            use_cache=not config.use_4bit)

    generated_text = tokenizer.batch_decode(append_sol_and_remove_eos(full_state if config.use_tools else generated_tokens, [None,] * generated_tokens.size(0), eos_token_id, pad_token_id))
    reward = rew.score(
                    append_sol_and_remove_eos(full_state if config.use_tools else generated_tokens, [None,] * generated_tokens.size(0), eos_token_id, pad_token_id),
                    skip_first=encoded_input.size(-1),
                    solution_len=0).cpu().float().numpy().tolist()
    return generated_text, reward


def test(config, model, tokenizer, vocab_nice_list, dataset, eos_token_id, pad_token_id, step, max_eval_len):
    # import pdb; pdb.set_trace();
    if dataset["test_num_sols"] is not None:
        accuracy, incorrect_examples = run_evaluation(model, tokenizer, dataset["encoded_test_queries"], 
                                    dataset["test_num_sols"], eos_token_id, pad_token_id, vocab_nice_list, max_eval_len,
                                    config.eval_bsz, config.use_tools, config.limit_capability, config.operators, not config.use_4bit)
        with open(f'incorrect_examples_{step}.txt', 'w') as f:
            for example in incorrect_examples:
                f.write(str(example) + '\n')
    
        return accuracy
    return 0


def load_model(config):
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(config.name,
                                                trust_remote_code=True,
                                                device_map="auto",
                                                quantization_config=bnb_config)
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False
    else:
        model = AutoModelForCausalLM.from_pretrained(config.name,
                                                torch_dtype=torch.bfloat16,
                                                trust_remote_code=True)
        model.to(config.device)
    
    tokenizer = AutoTokenizer.from_pretrained(config.name, add_bos_token=False)
    
    if config.use_lora and not config.test_only:
        lora_config = LoraConfig(
            r=config.lora_config.r,
            lora_alpha=config.lora_config.alpha,
            target_modules=list(config.lora_config.target_modules),
            lora_dropout=config.lora_config.dropout,
            bias="none",
        )
        lora_model = get_peft_model(model, lora_config)
        return lora_model, tokenizer
    if config.test_only and config.load_checkpoint_path is not None:
        model = PeftModel.from_pretrained(model, config.load_checkpoint_path)
    return model, tokenizer


@hydra.main(version_base=None, config_path='./configs/', config_name='rationale_buffer')
def main(config):
    print(os.getcwd())
    if not config.test_only:
        wandb.init(project="rationale_buffer_llm", mode=config.wandb_mode, config=OmegaConf.to_container(config, resolve=True))
    model, tokenizer = load_model(config.model)

    data_path = to_absolute_path(config.reward.reward_config.prompt_data.path)
    data = json.load(open(data_path, 'r'))
    train_queries = []
    train_rationales = []
    train_sols = []
    for sample in data:
        train_queries.append(sample['str_query'])
        train_rationales.append(sample['str_rationale'])
        train_sols.append(sample['str_sol'])
    if config.reward.reward_config.prompt_data.num_points > 0:
        
        idxs = list(range(config.reward.reward_config.prompt_data.num_points))
        random.shuffle(idxs)
        prompt_text = '\n'.join([train_queries[i]+ " " + train_rationales[i] + train_sols[i] for i in idxs]) + '\n'
    else:
        prompt_text = ""
    prompt_text = config.reward.reward_config.prompt_data.extra_text + prompt_text
    print("Prompt text: ", prompt_text)

    dataset = load_dataset(config, tokenizer, append_prompt_test=prompt_text)
    reward, vocab_nice_list = load_reward(config.reward, model, tokenizer, dataset, config.min_len)
    seed_everything(config.seed)


    with torch.no_grad():
        encoded_test_queries = dataset["encoded_test_queries"]
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.eos_token_id
        # print some examples
        for _ in range(10):
            query_ind = random.randint(0, len(encoded_test_queries)-1)
            encoded_input = encoded_test_queries[query_ind]

            text, rewards = generate_examples(model, tokenizer, encoded_input, eos_token_id, pad_token_id, vocab_nice_list, reward, config.max_eval_len, config)
            print(f"Test example: {text}")
            print(f'mean reward: {rewards}')

        accuracy = test(config, model, tokenizer, vocab_nice_list, dataset, tokenizer.eos_token_id, tokenizer.eos_token_id, 0, config.max_eval_len)
        print('accuracy: ', accuracy)

if __name__ == '__main__':
    main()
