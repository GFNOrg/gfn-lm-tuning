# Experiment - Next sentence continuation

This file gives instructions for how to setup and run the experiment on next sentence continuation. We will go over:

1. How to create the dataset of prompts that the model is trained on.
2. How to train the GFlowNet fine-tuned model on the dataset of prompts.
3. The specific configuration used in our paper.

We do not provide package requirements here, but they include standard python packages that can easily be installed using `pip` on Linux or macOS. Just keep trying to run the code and install packages along the way as you encounter import errors.

## 1. Create the dataset

The dataset consists of 1000 prompts obtained from Open Web Text. Each prompt is guaranteed to have between 1-3 sentences (each limited to a minimum of 5 and maximum of 30 tokens using the GPT-2 tokenizer). To generate this dataset, run:

```python data/next_sentence/openwebtext/data_generator.py```

The resulting prompts will be saved within `data/next_sentence/openwebtext/prompts.txt`, and some statistics about them will be saved in `data/next_sentence/openwebtext/prompt_stats.png`.

## 2. Train the GFlowNet to sample the next sentence

To train the GFlowNet, run:

```python train_next_sentence.py task=open_webtext_gpt2 device=gpu```

The code uses `PyTorch Lightning` to train the model, `wandb` to log results during training, and `hydra` to manage configurations. To modify the `wandb` logging arguments (e.g., to log to your personal account), you must modify the relevant `logger` object in the `next_sentence/configs/train.yaml`.

## 3. Train with the configuration used in our paper

We train with a specific set of arguments in our paper. These arguments can be specified at the command line, and modify the defaults from `next_sentence/configs/task/openwebtext_gpt2.yaml`. To replicate our results, run:

```python train_next_sentence.py task=open_webtext_gpt2 device=gpu task.training.n_samples=8 task.training.accumulate_grad_batches=32 task.reward.temp_end=[temp_end]```

Where `[temp_end]` is a placeholder that must be specified. In our paper, we try sweep over (0.8, 0.95) in increments of 0.25.

Note that this configuration requires approximately 40GB of GPU RAM.