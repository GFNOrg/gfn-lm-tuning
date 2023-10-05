import torch

def scalar_reward(distribution, parameters, value):
    if "uniform" in distribution:
        return 1 if parameters["a"] <= value <= parameters["b"] else 0
    elif distribution == "poisson":
        λ = parameters["lam"]
        return λ**value / torch.factorial(value)

    elif distribution == "binomial":
        n = parameters["n"]
        p = parameters["p"]
        binom_coeff = torch.comb(
            torch.tensor(n, dtype=torch.float64),
            torch.tensor(value, dtype=torch.float64),
        )
        return binom_coeff * (p / (1 - p)) ** value

    elif distribution == "geometric":
        p = parameters["p"]
        return (1 - p) ** value

    elif distribution == "gaussian":
        mean = parameters["mean"]
        std_dev = parameters["std_dev"]
        return torch.exp(-0.5 * ((value - mean) / std_dev) ** 2)

    elif distribution == "exponential":
        scale = parameters["scale"]
        return torch.exp(-value / scale)

@torch.inference_mode()
def reward_fn(x, n_max, skip_first=1):
    # Get the token ids after skip_first
    token_ids = x[:, skip_first:]

    # print("x ", x)

    if token_ids.shape[1] == 0:
        return torch.full((x.shape[0],), fill_value=-8).cuda()

    # Initialize rewards to a low value
    rew = torch.full(token_ids.shape, fill_value=-8).cuda() * torch.arange(
        1, token_ids.shape[1] + 1, device="cuda"
    )

    # Create a mask where the first token is in the list of tokenized permitted numbers
    tokenized_permitted_numbers_mask = tokens_of_numbers[:n_max]
    first_token_mask = torch.cat(
        (
            (token_ids[:, 0].unsqueeze(-1) == tokenized_permitted_numbers_mask)
            .any(dim=-1)
            .unsqueeze(-1),
            torch.zeros(
                token_ids.shape[0], token_ids.shape[1] - 1, device="cuda"
            ).bool(),
        ),
        dim=-1,
    )

    # Create a mask where the remaining tokens are the eos_token_id
    remaining_eos_tokens_mask = torch.cat(
        (
            torch.zeros(token_ids.shape[0], 1, device="cuda").bool(),
            token_ids[:, 1:] == tokenizer.eos_token_id,
        ),
        dim=-1,
    )

    # Set the reward to 8 (resp. 0) where
    # the token is the first token and in the list of tokenized permitted numbers,
    # (resp. it is one of the remaining tokens and equal to eos_token_id)
    rew[first_token_mask] = 8
    rew[remaining_eos_tokens_mask] = 0

    # return torch.cat((res[:, :1], res), dim=-1)
    # return torch.cat((rew, rew.new_full((rew.shape[0], 1), 0)), dim=-1).cumsum(dim=-1)
    return rew.sum(dim=-1)