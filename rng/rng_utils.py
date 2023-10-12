import torch
from tqdm import tqdm
import json


def format_dict(d):
    if isinstance(d, str):
        d = json.loads(d)
    return ", ".join(f"{k}={v}" for k, v in d.items())


def number_from_generated_text(generated_text, tokenizer, type_of_number="integer"):
    eos_string = tokenizer.decode(tokenizer.eos_token_id)
    convert_func = int if type_of_number == "integer" else float
    try:
        return convert_func(tokenizer.decode(generated_text).replace(eos_string, "").rstrip())
    except ValueError:
        return None

def get_distribution(
    inference_model,
    tokenizer,
    prompt,
    eos_token_id=None,
    num_samples=32 * 512,
    num_samples_per_batch=512,
    max_len=30,
    temperature=1,
    type_of_number="integer",
):
    generated_numbers = []
    len_prompt = len(tokenizer.encode(prompt))

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    number_of_NaNs = 0

    inference_model.eval()
    with torch.inference_mode():
        encoded_test_query = tokenizer(prompt, return_tensors="pt").to("cuda")

        for _ in tqdm(range(num_samples // num_samples_per_batch)):
            generated_outputs = inference_model.generate(
                **encoded_test_query,
                max_new_tokens=max_len,
                temperature=temperature,
                pad_token_id=eos_token_id,
                num_return_sequences=num_samples_per_batch,
                do_sample=True,
            )

            generated_numbers_batch = []
            for generated_output in generated_outputs:
                generated_number = number_from_generated_text(
                    generated_output[len_prompt:], 
                    tokenizer,
                    type_of_number=type_of_number,
                )
                if generated_number is None:
                    number_of_NaNs += 1
                else:
                    generated_numbers_batch.append(generated_number)

            generated_numbers.extend(generated_numbers_batch)
    return (generated_numbers, number_of_NaNs)


def generate(model, encoded_prompt, eos_token_id, max_len=10, temperature=1., vocab_nice_mask=None,vocab_alpha=-99):
    active_seqs = torch.ones(encoded_prompt.size(0)).bool().to(encoded_prompt.device)
    logPF = encoded_prompt.new_zeros(encoded_prompt.size(0)).float()
    actions = encoded_prompt.clone()
    state = encoded_prompt.clone()
    past_key_values = model(state[:, :-1])['past_key_values']
    for _ in range(max_len):
        output = model(state[:, -1:], past_key_values=past_key_values)
        past_key_values = output['past_key_values']
        with torch.inference_mode():
            prob = (output['logits'][:, -1, :]).softmax(dim=-1)
            modified_logits = output['logits'][:, -1, :].clone()
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
        # check if all sequences have generated eos
        active_seqs = active_seqs * (token_ids != eos_token_id).squeeze(-1)
        if torch.all(~active_seqs):
            break
    return actions, logPF, state

def generate_and_return_eos_logprob(model, encoded_prompt, eos_token_id, reward_fn, vocab_nice_mask=None,
                                    vocab_alpha=-99, max_len=10, min_len=0,
                                    temperature=1., action_seq=None, skip_rewards=False):
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
        output = model(state[:, -1:], past_key_values=past_key_values)
        past_key_values = output['past_key_values']
        if action_seq is None:
            with torch.inference_mode():
                prob = (output['logits'][:, -1, :]).softmax(dim=-1)
                modified_logits = output['logits'][:, -1, :].clone().detach()
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
        logpf[:, i] = torch.where(active_seqs, logprob.gather(-1, token_ids).squeeze(-1), 0)
        logeosprobs[:, i] = torch.where(active_seqs, logprob[:, eos_token_id], 0)
        actions = torch.cat([actions, token_ids], dim=-1)
        state = torch.cat([state, token_ids], dim=-1)
        if not skip_rewards:
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