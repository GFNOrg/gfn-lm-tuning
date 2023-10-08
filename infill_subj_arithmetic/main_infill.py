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
# from replay_buffer import load_replay_buffer
from data import load_dataset
import replay_buffer


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(config, model, tokenizer, vocab_nice_list, rew, rbuffer, dataset):
    logZ = torch.nn.Parameter(torch.tensor([config.logZ_init], dtype=torch.float, device=config.model.device))
    if config.use_4bit:
        opt = bnb.optim.PagedAdamW8bit([{'params': model.parameters(), 'lr': config.lr},
                                {'params': [logZ,], 'lr': config.lr_logZ}])
    else:
        opt = torch.optim.AdamW([{'params': model.parameters(), 'lr': config.lr},
                                {'params': [logZ,], 'lr': config.lr_logZ}])

    pb = tqdm(range(config.train_steps))
    desc_str = "Eval Reward: {:.3f} | Train Loss: {:.3f}, LogZ: {:.3f}, Reward: {:.3f}"
    pb.set_description(desc_str.format(0, 0, 0, 0))
    eval_reward = 0
    
    bsz = config.batch_size
    grad_acc = config.grad_acc

    pf_temp_high = config.pf_temp_high
    pf_temp_low = config.pf_temp_low

    subtb_lambda = config.subtb_lambda

    max_len = config.max_len
    max_eval_len = config.max_eval_len
    min_len = config.min_len

    loss_type = config.loss_type

    reward_sched_start = config.reward_sched.start
    reward_sched_end = config.reward_sched.end
    reward_sched_horizon = config.reward_sched.horizon

    get_reward_temp = lambda x : reward_sched_start + (reward_sched_end - reward_sched_start) * min(1, x / reward_sched_horizon)

    encoded_train_queries = dataset["encoded_train_queries"]
    encoded_train_sols = dataset["encoded_train_sols"]
    encoded_test_queries = dataset["encoded_test_queries"]
    encoded_train_beginning = dataset["encoded_train_beginning"]
    train_weight = dataset["train_weight"]
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.eos_token_id
    get_lr_at_step = lambda x : min(x/20*config.lr, config.lr)
    
    for step in pb:
        opt.zero_grad()
        loss = 0.
        # change reward temperature
        rew.temperature = get_reward_temp(step)
        for pg in opt.param_groups:
            pg['lr'] = get_lr_at_step(step)
        for _ in range(grad_acc):
            # select an example
            #query_ind = random.randint(0, len(encoded_train_queries)-1)
            query_ind = np.random.choice(np.arange(len(encoded_train_queries)), p=train_weight)
            encoded_input = encoded_train_queries[query_ind]
            encoded_result = encoded_train_sols[query_ind]
            encoded_beginning = encoded_train_beginning[query_ind]
            if loss_type.startswith('modified'):
                # choose a behavior policy
                b_policy_choice = random.randint(0, 3)
                reward_fn = lambda x: rew.score(
                                append_sol_and_remove_eos(torch.cat((encoded_beginning.repeat(bsz, 1), x[:, encoded_input.size(-1):]), axis=1),
                                                        encoded_result.repeat(bsz, 1),
                                                        eos_token_id,
                                                        pad_token_id),
                                infills=x[:, encoded_input.size(-1):],
                                skip_first=encoded_beginning.size(-1),
                                solution_len=encoded_result.size(-1))
                if b_policy_choice == 0:
                    # using the action policy without tempering
                    generated_text, logPF, eos_logprob, logrewards = \
                        generate_and_return_eos_logprob(model, 
                                                        encoded_input.repeat(bsz, 1),
                                                        eos_token_id=eos_token_id,
                                                        vocab_nice_mask=vocab_nice_list,
                                                        reward_fn=reward_fn,
                                                        max_len=max_len,
                                                        min_len=min_len,
                                                        temperature=1,
                                                        tokenizer=tokenizer,
                                                        use_tools=config.use_tools,
                                                        limit_capability=config.limit_capability,
                                                        operators=config.operators,
                                                        use_cache=not config.use_4bit)
                    rbuffer.add_batch(query=encoded_input,
                                    answer=encoded_result,
                                    rationales=generated_text[:, encoded_input.size(-1):],
                                    logrewards=logrewards * rew.temperature, # undo the effect of reward tempering
                                    tokenizer=tokenizer)
                elif b_policy_choice == 1 and rbuffer.sample(bsz, query=encoded_input, answer=encoded_result)[0] is not None: # and step > 10:
                    # using samples from the replay buffer
                    action_seq, logrewards = rbuffer.sample(bsz, query=encoded_input, answer=encoded_result)
                    if action_seq is None:
                        continue
                    logrewards *= (1/rew.temperature) # redo the effect of reward tempering
                    generated_text, logPF, eos_logprob, logrewards_2 = \
                        generate_and_return_eos_logprob(model, 
                                                        encoded_input.repeat(action_seq.size(0), 1),
                                                        eos_token_id=eos_token_id,
                                                        reward_fn=reward_fn,
                                                        vocab_nice_mask=vocab_nice_list,
                                                        max_len=max_len,
                                                        min_len=min_len,
                                                        action_seq=action_seq,
                                                        skip_rewards=True,
                                                        tokenizer=tokenizer,
                                                        use_tools=False,
                                                        limit_capability=config.limit_capability,
                                                        operators=config.operators,
                                                        use_cache=not config.use_4bit)
                else:
                    # using the action policy with tempering
                    generated_text, logPF, eos_logprob, logrewards = \
                        generate_and_return_eos_logprob(model, 
                                                        encoded_input.repeat(bsz, 1),
                                                        eos_token_id=eos_token_id,
                                                        reward_fn=reward_fn,
                                                        vocab_nice_mask=vocab_nice_list,
                                                        max_len=max_len,
                                                        min_len=min_len,
                                                        temperature=random.random()*(pf_temp_high-pf_temp_low)+pf_temp_low,
                                                        tokenizer=tokenizer,
                                                        use_tools=config.use_tools,
                                                        limit_capability=config.limit_capability,
                                                        operators=config.operators,
                                                        use_cache=not config.use_4bit)
                    rbuffer.add_batch(query=encoded_input,
                                    answer=encoded_result,
                                    rationales=generated_text[:, encoded_input.size(-1):],
                                    logrewards=logrewards * rew.temperature, # undo the effect of reward tempering
                                    tokenizer=tokenizer)
                if loss_type == 'modified_db':
                    # modified db loss with logpb=0
                    db_loss = (logrewards[:, :-1] + logPF[:, :-1] + eos_logprob[:, 1:] - logrewards[:, 1:] - eos_logprob[:, :-1])**2
                    # get a mask for newly generated tokens after the first eos in generated_text
                    mask = (generated_text[:, encoded_input.size(-1):] == eos_token_id).cumsum(dim=-1) > 1
                    # if mask is too short, pad it
                    if mask.size(-1) < max_len:
                        mask = torch.cat([mask, torch.ones(mask.size(0), max_len-1-mask.size(-1), dtype=torch.bool, device='cuda')], dim=-1)
                    mask = mask[:, :max_len]
                    # get trajectory lengths by summing the mask
                    traj_len = (~mask).sum(dim=-1)
                    # get rid of the loss for the terminating step
                    db_loss[mask] = 0
                    batch_loss = db_loss.sum(-1) / traj_len
                    #batch_loss = batch_loss.topk(bsz//2, largest=False, sorted=False).values.mean()
                elif loss_type == 'modified_subtb':
                    # modified subTB loss with logpb=0
                    delta = (logrewards[:, :-1] - eos_logprob[:, :-1] + logPF[:, :-1] - (logrewards[:, 1:] - eos_logprob[:, 1:]))
                    #delta = F.huber_loss(logrewards[:, :-1] + logPF[:, :-1] + eos_logprob[:, 1:],
                    #                     logrewards[:, 1:] + eos_logprob[:, :-1],
                    #                     delta=.1,
                    #                     reduction='none')
                    delta_cumsum = torch.cat( [ torch.zeros_like(delta[:, :1]), delta ], 1).cumsum(1)
                    # get a mask for tokens after the first eos in generated_text
                    mask = (generated_text == eos_token_id).cumsum(dim=-1) > 1
                    mask = mask[:, encoded_input.size(-1):]
                    mask = mask[:, :max_len]
                    # if mask is too short, pad it
                    if mask.size(-1) < max_len:
                        mask = torch.cat([mask, torch.ones(mask.size(0), max_len-mask.size(-1), dtype=torch.bool, device='cuda')], dim=-1)
                    # get trajectory lengths by summing the mask

                    batch_loss = 0.
                    total_lambda = 0.
                    for subtraj_len in range(1, max_len+1):
                        subtb_term = (delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len])**2
                        subtb_term[mask[:, subtraj_len - 1:]] = 0
                        batch_loss += subtb_lambda ** (subtraj_len - 1) * subtb_term.sum()
                        total_lambda += subtb_lambda ** (subtraj_len - 1) * (~mask[:, subtraj_len - 1:]).sum()
                    batch_loss /= total_lambda
                elif loss_type == "ppo":
                    # PPO loss
                    # get a mask for tokens after the first eos in generated_text
                    mask = (generated_text == eos_token_id).cumsum(dim=-1) > 1
                    mask = mask[:, encoded_input.size(-1):]
                    mask = mask[:, :max_len]
                    # if mask is too short, pad it
                    if mask.size(-1) < max_len:
                        mask = torch.cat([mask, torch.ones(mask.size(0), max_len-mask.size(-1), dtype=torch.bool, device='cuda')], dim=-1)
                    # get trajectory lengths by summing the mask
                    traj_len = (~mask).sum(dim=-1)
                    # get rid of the loss for the terminating step
                    logrewards[mask] = 0
                    batch_loss = logrewards.sum(-1) / traj_len

            else:
                raise NotImplementedError
            loss += batch_loss.mean()
            batch_loss.mean().backward()
        opt.step()
        
        if step % config.log_interval == 0:
            print(f'logZ: {logZ.item()}, loss: {loss.item()}')
        if step % config.full_eval_interval == 0:
            # accuracy, incorrect_examples = run_evaluation(model, tokenizer, encoded_test_queries, 
            #                           dataset["test_num_sols"], eos_token_id, pad_token_id, vocab_nice_list, max_eval_len,
            #                           config.eval_bsz, config.use_tools, config.limit_capability, config.operators)
            accuracy = test(config, model, tokenizer, vocab_nice_list, dataset, eos_token_id, pad_token_id, step, max_eval_len)
            print('accuracy: ', accuracy)
            wandb.log({"accuracy": accuracy}, commit=False)

        if step % config.eval_interval == 0:
            query_ind = random.randint(0, len(encoded_test_queries)-1)
            encoded_input = encoded_test_queries[query_ind]

            text, rewards = generate_examples(model, tokenizer, encoded_input, eos_token_id, pad_token_id, vocab_nice_list, rew, max_eval_len, config)
            sols = dataset["test_sols"][query_ind] if "test_sols" in dataset and dataset["test_sols"] else "Test" 
            print_table([dataset["test_queries"][query_ind]] * max_eval_len, text, [sols] * max_eval_len, rewards)

            query_ind = random.randint(0, len(encoded_train_queries)-1)
            encoded_input = encoded_train_queries[query_ind]
            text, rewards = generate_examples(model, tokenizer, encoded_input, eos_token_id, pad_token_id, vocab_nice_list, rew, max_eval_len, config)
            print_table([dataset["train_queries"][query_ind]] * max_eval_len, text, [dataset["train_sols"][query_ind]] * max_eval_len, rewards)

            rbuffer.save(f"buffer_{step}.pkl.gz")
            model.save_pretrained(f'{wandb.run.name}_{step}.pt')
        
        wandb.log({"loss": loss.mean().item(), 'reward': logrewards.mean().item()})
        pb.set_description(desc_str.format(eval_reward, loss.mean().item(), logZ.item(), logrewards.mean().item()))


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
                    append_sol_and_remove_eos(generated_tokens, [None,] * generated_tokens.size(0), eos_token_id, pad_token_id),
                    infills=generated_tokens[:, encoded_input.size(-1):],
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
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(config.name,
                                                trust_remote_code=True,
                                                device_map="auto",
                                                torch_dtype=torch.bfloat16,
                                                quantization_config=bnb_config)
        # model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False
    else:
        model = AutoModelForCausalLM.from_pretrained(config.name,
                                                torch_dtype=torch.bfloat16,
                                                trust_remote_code=True)
        model.to(config.device)
        # model.gradient_checkpointing_enable()
    
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
    # model = torch.compile(model)
    return model, tokenizer

def load_replay_buffer(config, model, dataset, tokenizer, rew, max_len, min_len, vocab_nice_list, use_cache):
    rbuffer = replay_buffer.ReplayBuffer(config.size, eos_token_id=tokenizer.eos_token_id)
    base_to_lora(model)
    # add handcrafted rationales to the replay buffer
    for i in range(len(dataset["encoded_train_queries"])):
        if dataset["train_rationales"][i] is None:
            continue
        encoded_rationale = tokenizer(dataset["train_rationales"][i], return_tensors='pt')['input_ids'].cuda()
        if encoded_rationale.size(-1) > max_len:
            continue
        reward_fn = lambda x: rew.score(
                                append_sol_and_remove_eos(torch.cat((dataset["encoded_train_beginning"][i], x[:, dataset["encoded_train_queries"][i].shape[-1]:]), axis=1),
                                                        dataset["encoded_train_sols"][i],
                                                        tokenizer.eos_token_id,
                                                        tokenizer.eos_token_id),
                                infills=x[:, dataset["encoded_train_queries"][i].shape[-1]:],
                                skip_first=dataset["encoded_train_beginning"][i].size(-1),
                                solution_len=dataset["encoded_train_sols"][i].size(-1))
        with torch.no_grad():
            # import pdb; pdb.set_trace();
            logrewards = generate_and_return_eos_logprob(model, 
                                                            dataset["encoded_train_queries"][i],
                                                            eos_token_id=tokenizer.eos_token_id,
                                                            vocab_nice_mask=vocab_nice_list,
                                                            reward_fn=reward_fn,
                                                            max_len=max_len,
                                                            min_len=min_len,
                                                            temperature=1,
                                                            action_seq=encoded_rationale,
                                                            tokenizer=tokenizer,
                                                            use_tools=config.use_tools,
                                                            use_cache=use_cache)[3]
        rbuffer.add_batch(query=dataset["encoded_train_queries"][i],
                        answer=dataset["encoded_train_sols"][i],
                        rationales=encoded_rationale,
                        logrewards=logrewards,
                        tokenizer=tokenizer)
    return rbuffer


@hydra.main(version_base=None, config_path='./configs/', config_name='rationale_buffer')
def main(config):
    print(os.getcwd())
    if not config.test_only:
        wandb.init(project="rationale_buffer_llm", mode=config.wandb_mode, config=OmegaConf.to_container(config, resolve=True))
    model, tokenizer = load_model(config.model)
    dataset = load_dataset(config, tokenizer)
    reward, vocab_nice_list = load_reward(config.reward, model, tokenizer, dataset, config.min_len)
    if config.use_replay_buffer and not config.test_only:
        rbuffer = load_replay_buffer(config.replay_buffer, model, dataset, tokenizer, reward, config.max_len, config.min_len, vocab_nice_list, not config.use_4bit)
    
    seed_everything(config.seed)
    if not config.test_only:
        train(config, model, tokenizer, vocab_nice_list, reward, rbuffer, dataset)
    else:
        
        encoded_test_queries = dataset["encoded_test_queries"]
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.eos_token_id
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
