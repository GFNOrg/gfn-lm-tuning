from types import MethodType
import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import get_peft_model, prepare_model_for_kbit_training
from utils import (
    FrozenModelSentenceGivenPrompt,
    RuleSentenceValidator,
    ModelSentenceValidator,
    ReplayBuffer,
)
from lightning_module import NextSentenceGFNTask
from lightning_data import PromptDataModule


@hydra.main(version_base=None, config_path="./configs/", config_name="train")
def train(config: DictConfig):
    pl.seed_everything(config.seed, workers=True)

    model, tokenizer = get_model(config)

    try:  # Some tokenizers encode a "." differently when it is the first token
        end_of_sentence_token_id = tokenizer.encode(
            "A sentence.", add_special_tokens=False
        )[-1]
    except:
        end_of_sentence_token_id = tokenizer.convert_tokens_to_ids(".")
    illegal_token_mask = torch.zeros(tokenizer.vocab_size, dtype=torch.bool)
    illegal_tokens = OmegaConf.to_container(config.task.constraints.illegal_tokens)
    illegal_tokens = [
        [t] if isinstance(t, int) else tokenizer.encode(t, add_special_tokens=False)
        for t in illegal_tokens
    ]
    assert all(len(t) == 1 for t in illegal_tokens)
    illegal_tokens = [t[0] for t in illegal_tokens]
    illegal_token_mask[illegal_tokens] = True
    illegal_token_mask = illegal_token_mask.numpy()

    reward = get_reward(config, end_of_sentence_token_id, illegal_token_mask)
    reward_buffer = ReplayBuffer(
        buffer_size=config.task.reward.buffer_size,
        termination_token_id=end_of_sentence_token_id,
    )

    data = PromptDataModule(
        data_path=config.task.data.path,
        tokenizer=tokenizer,
        train_size=config.task.data.train_size,
        limit_prompts=config.task.data.limit_prompts,
    )
    data.setup("fit")
    train_probes = [data.train_data[i][0] for i in range(config.task.eval.n_probes)]
    val_probes = [data.val_data[i][0] for i in range(config.task.eval.n_probes)]

    task = NextSentenceGFNTask(
        model=model,
        tokenizer=tokenizer,
        reward=reward,
        reward_buffer=reward_buffer,
        n_samples=config.task.training.n_samples,
        lr=config.task.training.lr,
        subtb_lambda=config.task.training.subtb_lambda,
        pf_temp_high=config.task.training.pf_temp_high,
        pf_temp_low=config.task.training.pf_temp_low,
        pf_temp_prob=config.task.training.pf_temp_prob,
        use_buffer_prob=config.task.training.use_buffer_prob,
        min_sentence_len=config.task.constraints.min_sentence_len,
        max_sentence_len=config.task.constraints.max_sentence_len,
        reward_temp_start=config.task.reward.temp_start,
        reward_temp_end=config.task.reward.temp_end,
        reward_temp_horizon=config.task.reward.temp_horizon,
        illegal_token_mask=illegal_token_mask,
        train_probes=train_probes,
        val_probes=val_probes,
        diversity_metric=config.task.eval.diversity_metric,
        use_4bit=config.task.training.use_4bit,
    )

    trainer = pl.Trainer(
        accelerator=config.device.accelerator,
        max_epochs=config.task.training.epochs,
        accumulate_grad_batches=config.task.training.accumulate_grad_batches,
        logger=config.logger
        if isinstance(config.logger, bool)
        else hydra.utils.instantiate(config.logger),
        callbacks=[hydra.utils.instantiate(c) for c in config.task.callbacks],
    )

    # Fix a bug that arises when using 4-bit quantized models.
    # It's caused by different operations being on different devices,
    # so we'll just deactivate lightning's automatic device placement
    # and let huggingface handle the dynamic device placement
    if config.task.training.use_4bit:
        task.to = MethodType(lambda s, _: s, task)
        task.cuda = MethodType(lambda s: s, task)

    trainer.fit(model=task, datamodule=data)


def get_model(config: DictConfig):
    # Use 4-bit quantization for lower memory use
    if config.task.training.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # Get the model
    tokenizer = AutoTokenizer.from_pretrained(
        config.task.model.name, add_bos_token=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.task.model.name, device_map="auto", quantization_config=bnb_config
    )

    # Prepare model for k-bit training
    if config.task.training.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,  # Doesn't save memory when generating autoregressively compared to caching
        )

    # Wrap using Lora
    model = get_peft_model(
        model, hydra.utils.instantiate(config.task.model.lora_config)
    )

    # Remove dropout
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0

    return model, tokenizer


def get_reward(config: DictConfig, sentence_token_id, illegal_token_mask):
    if config.task.reward.sentence_validator is None:
        sentence_validator, valid_sentence_alpha = None, None
    elif config.task.reward.sentence_validator == "rule":
        sentence_validator, valid_sentence_alpha = (
            RuleSentenceValidator(sentence_token_id=sentence_token_id),
            config.task.reward.valid_sentence_alpha,
        )
    elif config.task.reward.sentence_validator == "model":
        sentence_validator, valid_sentence_alpha = (
            ModelSentenceValidator(sentence_token_id=sentence_token_id),
            config.task.reward.valid_sentence_alpha,
        )
    else:
        raise ValueError(
            f"Invalid sentence validator: {config.task.reward.sentence_validator}"
        )

    reward = FrozenModelSentenceGivenPrompt(
        sentence_token_id=sentence_token_id,
        min_len=config.task.constraints.min_sentence_len,
        vocab_alpha=config.task.reward.vocab_alpha,
        vocab_naughty_mask=illegal_token_mask,
        sentence_validator=sentence_validator,
        valid_sentence_alpha=valid_sentence_alpha,
    )

    return reward


if __name__ == "__main__":
    train()
