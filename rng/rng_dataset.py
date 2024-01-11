import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence


def split_dataframe(df, test_size=0.2, random_state=None):
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    return train_df, test_df


def get_tensors_from_dataframe(df, tokenizer, method="GFN"):
    prompts = df["Prompt"].tolist()

    if method == "SFT":
        labels = df["Value"].tolist()
    else:
        labels = df["Distribution and Parameters Index"].tolist()

    input_ids = []
    output_ids = []

    for prompt, label in tqdm(zip(prompts, labels), desc="Tokenizing dataset..."):
        encoded_prompt = tokenizer.encode(
            prompt, add_special_tokens=True, return_tensors="pt"
        ).to("cuda")

        if method == "SFT":
            encoded_value = tokenizer.encode(
                str(label), add_special_tokens=True, return_tensors="pt"
            ).to("cuda")
            input_id = (
                torch.cat([encoded_prompt, encoded_value], dim=-1).squeeze(0).to("cuda")
            )
            attn = (
                torch.cat(
                    [torch.zeros_like(encoded_prompt), torch.ones_like(encoded_value)],
                    dim=-1,
                )
                .squeeze(0)
                .to("cuda")
            )
            output_id = torch.where(attn == 1, input_id, -100)
        else:
            input_id = encoded_prompt.squeeze(0)
            output_id = torch.tensor(label).unsqueeze(0).to("cuda")

        input_ids.append(input_id)
        output_ids.append(output_id)

    if method == "PPO":
        # Left-padding of all sequences to the maximum length
        # Reverse sequences, pad them, then reverse them back
        input_ids = [x.flip(dims=[0]) for x in input_ids]
        output_ids = [x.flip(dims=[0]) for x in output_ids]

        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.eos_token_id
        )
        output_ids = pad_sequence(
            output_ids, batch_first=True, padding_value=tokenizer.eos_token_id
        )

        input_ids = input_ids.flip(dims=[1])
        output_ids = output_ids.flip(dims=[1])

    return input_ids, output_ids


def get_dataloader_from_dataframe(df, tokenizer, bsz, shuffle=True, method="GFN"):
    def collate_fn_SFT(batch):
        input_ids, target_ids = zip(*batch)
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.eos_token_id
        )
        target_ids = pad_sequence(
            target_ids, batch_first=True, padding_value=tokenizer.eos_token_id
        )
        return input_ids, target_ids

    input_ids, output_ids = get_tensors_from_dataframe(df, tokenizer, method=method)
    if method == "SFT":
        dataset = list(zip(input_ids, output_ids))
        dataloader = DataLoader(
            dataset, batch_size=bsz, shuffle=shuffle, collate_fn=collate_fn_SFT
        )
    else:
        dataset = list(zip(input_ids, output_ids))
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle)

    return dataloader


# def pad_sequences(sequences, padding_value):
#     max_length = max(len(seq) for seq in sequences)
#     return [seq + [padding_value] * (max_length - len(seq)) for seq in sequences]


def get_train_test_dataloaders(df, tokenizer, bsz):
    train_df, test_df = split_dataframe(df)
    train_dataloader = get_dataloader_from_dataframe(
        train_df, tokenizer, bsz, shuffle=True, method="SFT"
    )
    test_dataloader = get_dataloader_from_dataframe(
        test_df, tokenizer, bsz, shuffle=False, method="SFT"
    )
    return train_dataloader, test_dataloader
