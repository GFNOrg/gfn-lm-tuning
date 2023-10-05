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
