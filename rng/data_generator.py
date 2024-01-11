import hydra
import numpy as np
import pandas as pd
import random
from omegaconf import DictConfig, OmegaConf, ListConfig
import json
from tqdm import tqdm
from rng.rng_utils import format_dict
import os


def generate_data_for_distribution(distribution, params, data_type, prompts):
    if distribution == "uniform discrete":
        data = np.random.randint(int(params["a"]), int(params["b"]) + 1)
    elif distribution == "poisson":
        data = np.random.poisson(params["lam"])
    elif distribution == "binomial":
        data = np.random.binomial(params["n"], params["p"])
    elif distribution == "geometric":
        data = np.random.geometric(params["p"])
    elif distribution == "gaussian":
        data = np.random.normal(params["mean"], params["std_dev"])
    elif distribution == "exponential":
        data = np.random.exponential(params["scale"])
    elif distribution == "uniform continuous":
        data = np.random.uniform(params["a"], params["b"])
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    type_of_number = "integer" if data_type == "int" else "real number"

    instantiated_prompt = random.choice(prompts).format(
        type_of_number=type_of_number,
        distribution=distribution,
        parameters=format_dict(params),
    )

    df = pd.DataFrame(
        {
            "Prompt": instantiated_prompt,
            "Value": data,
            "Distribution and Parameters": json.dumps({'distribution': distribution, 'data_type': data_type} | params),
            "Distribution": distribution,
            "Parameters": json.dumps(params),
            "Data Type": data_type
        },
        index=[0],
    )

    return df


# hydra.initialize(config_path="configs/")

def build_distribution_and_params_list(df):
    return [json.loads(d) for d in df["Distribution and Parameters"].unique().tolist()]


@hydra.main(config_path="configs/", config_name="config", version_base=None)
def main(cfg):
    all_data_train = []
    distribution_and_params_dict = {}
    distribution_and_params_count = 0

    for dist in cfg.distributions.target_distributions:
        params = cfg.distributions[dist].parameters
        data_type = cfg.distributions[dist].type
        n_samples = cfg.distributions[dist].get("n_samples", cfg.hparams.n_samples)
        if n_samples is None:
            n_samples = 100
        df_train = []
        for _ in tqdm(range(n_samples), desc=f"Generating {dist} data"):
            sampled_params = {
                k: random.choice(v) if isinstance(v, ListConfig) else v
                for k, v in params.items()
            }
            df = generate_data_for_distribution(
                    dist, sampled_params, data_type, cfg.prompts
                )
            if df["Distribution and Parameters"][0] not in distribution_and_params_dict:
                distribution_and_params_dict[df["Distribution and Parameters"][0]] = distribution_and_params_count
                distribution_and_params_count += 1
            df["Distribution and Parameters Index"] = distribution_and_params_dict[df["Distribution and Parameters"][0]]
            df_train.append(df)
        all_data_train.extend(df_train)

    final_df_train = pd.concat(all_data_train, ignore_index=True)
    
    directory = os.path.dirname(cfg.file_name.train)
    if not os.path.exists(directory):
        os.makedirs(directory)

    final_df_train.to_csv(cfg.file_name.train, index=False)
    # visualize(final_df)
    # plt.savefig('images/data_visualization.png')

    return final_df_train


if __name__ == "__main__":
    main()
