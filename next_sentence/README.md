# GFlowNet fine-tuning for next sentence continuation

This file gives instructions for how to setup and run the experiment on next sentence continuation. We will go over:

1. How to create the dataset of prompts that the model is trained on.
2. How to train the GFlowNet fine-tuned model on the dataset of prompts.
3. The specific configuration used in our paper.

The requirements can be installed using `pip`:

```pip install -r requirements.txt```

Note that this experiment has to be run from the `next_sentence` directory for all imports and relative paths to function correctly.

## 1. Create the dataset

The dataset consists of 1000 prompts obtained from Open Web Text. Each prompt is guaranteed to have between 1-3 sentences (each limited to a minimum of 5 and maximum of 30 tokens using the GPT-2 tokenizer). The data is already included in the repository under `data/openwebtext/prompts.txt`, but to see how it was generated (and regenerate it yourself) run:

```python data/openwebtext/data_generator.py```

The resulting prompts will be saved within `data/openwebtext/prompts.txt`, and some statistics about them will be saved in `data/openwebtext/prompt_stats.png`.

## 2. Train the GFlowNet to sample the next sentence

To train the GFlowNet, run:

```python train.py task=open_webtext_gpt2 device=gpu```

The code uses `PyTorch Lightning` to train the model, `wandb` to log results during training, and `hydra` to manage configurations.

## 3. Train with the configuration used in our paper

We train with a specific set of arguments in our paper. These arguments can be specified at the command line, and modify the defaults from `configs/task/openwebtext_gpt2.yaml`. To replicate our results, run:

```python train.py task=open_webtext_gpt2 device=gpu task.training.n_samples=8 task.training.accumulate_grad_batches=32 task.reward.temp_end=[temp_end]```

Where `[temp_end]` is a placeholder that must be specified. In our paper, we sweep over (0.8, 0.95) in increments of 0.25.

Note that this configuration requires approximately 40GB of GPU RAM.
