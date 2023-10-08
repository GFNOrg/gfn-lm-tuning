import random
from functools import partial
import pandas as pd
import numpy as np
import torch
from pytorch_lightning import LightningModule
from utils import (
    generate_and_return_termination_logprob,
    modified_subtb_loss,
    get_termination_vals,
    SequenceDiversity,
)
from utils import lora_to_base, base_to_lora


class NextSentenceGFNTask(LightningModule):
    def __init__(
        self,
        model,
        tokenizer,
        reward,
        reward_buffer,
        n_samples,
        lr,
        subtb_lambda,
        pf_temp_high,
        pf_temp_low,
        pf_temp_prob,
        use_buffer_prob,
        min_sentence_len,
        max_sentence_len,
        reward_temp_start,
        reward_temp_end,
        reward_temp_horizon,
        illegal_token_mask,
        train_probes=None,
        val_probes=None,
        diversity_metric=None,
        use_4bit=False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "tokenizer"])

        self.model = model
        self.tokenizer = tokenizer
        self.reward = reward
        self.reward_buffer = reward_buffer

        self.diversity_metric_name = f"diversity ({diversity_metric})"
        self.diversity_metric = SequenceDiversity(diversity_metric)

        self.get_lr_at_step = lambda step: min(step / 20 * lr, lr)
        self.get_reward_temp_at_step = lambda step: reward_temp_start + (
            reward_temp_end - reward_temp_start
        ) * min(1, step / reward_temp_horizon)

        try:  # Some tokenizers encode a "." differently when it is the first token
            self.end_of_sentence_token_id = tokenizer.encode(
                "A sentence.", add_special_tokens=False
            )[-1]
        except:
            self.end_of_sentence_token_id = tokenizer.convert_tokens_to_ids(".")

    def forward(self, prompt, n_samples=None, pf_temperature=1.0, action_seq=None):
        assert prompt.ndim == 1
        n_samples = self.hparams.n_samples if n_samples is None else n_samples
        prompt = prompt.unsqueeze(0).expand(n_samples, -1)
        reward_fn = partial(
            self.reward.score,
            prompt_length=prompt.shape[1],
            model=self.model,
            tokenizer=self.tokenizer,
        )
        (
            generated_text,
            log_pf,
            log_pterm,
            log_r,
            log_r_unpenalized,
        ) = generate_and_return_termination_logprob(
            self.model,
            prompt,
            reward_fn=reward_fn,
            termination_token_id=self.end_of_sentence_token_id,
            vocab_naughty_mask=self.hparams.illegal_token_mask,
            min_len=self.hparams.min_sentence_len,
            max_len=self.hparams.max_sentence_len,
            temperature=pf_temperature,
            skip_rewards=False,
            action_seq=action_seq,
        )
        return generated_text, log_pf, log_pterm, log_r, log_r_unpenalized

    def training_step(self, prompt, batch_idx):
        # Should always be (1, prompt_len)
        prompt = prompt[0]

        # Sample a sentence and get the reward
        if (
            random.random() < self.hparams.use_buffer_prob
            and self.reward_buffer.sample(self.hparams.n_samples, prompt)[0] is not None
        ):
            # Using a sample from the reward buffer
            action_seq, log_r = self.reward_buffer.sample(
                self.hparams.n_samples, prompt
            )
            generated_text, log_pf, log_pterm, _, log_r_unpenalized = self.forward(
                prompt, action_seq=action_seq
            )
            log_r = log_r[
                :, : generated_text.shape[1] - len(prompt)
            ]  # Undo padding from buffer
            log_r *= 1 / self.reward.temperature  # redo the effect of reward tempering
        else:
            # Using the forward policy
            if random.random() < self.hparams.pf_temp_prob:  # With tempering
                pf_temp = (
                    random.random()
                    * (self.hparams.pf_temp_high - self.hparams.pf_temp_low)
                    + self.hparams.pf_temp_low
                )
            else:  # Without tempering
                pf_temp = 1.0
            generated_text, log_pf, log_pterm, log_r, log_r_unpenalized = self.forward(
                prompt, pf_temperature=pf_temp
            )
            self.reward_buffer.add_batch(
                prompt=prompt,
                sentences=generated_text[:, len(prompt) :],
                logrewards=log_r
                * self.reward.temperature,  # undo the effect of reward tempering
                tokenizer=self.tokenizer,
            )

        # Get the GFN loss
        loss = modified_subtb_loss(
            log_pf=log_pf,
            log_r=log_r,
            log_pterm=log_pterm,
            generated_text=generated_text,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=len(prompt),
            subtb_lambda=self.hparams.subtb_lambda,
        )

        # Log metrics
        _, last_log_r, last_log_r_unpenalized, sentence_len = get_termination_vals(
            generated_text=generated_text,
            log_pf=log_pf,
            log_pterm=log_pterm,
            log_r=log_r,
            log_r_unpenalized=log_r_unpenalized,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=len(prompt),
        )
        log_ps = last_log_r * self.reward.temperature
        log_ps_unpenalized = last_log_r_unpenalized * self.reward.temperature
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "train/logR",
            last_log_r.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/logP(s) (avg)",
            log_ps.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/logP(s) (max)",
            log_ps.max(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/logP(s) unpenalized (avg)",
            log_ps_unpenalized.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/logP(s) unpenalized (max)",
            log_ps_unpenalized.max(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/sentence_len",
            sentence_len.float().mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, prompt, batch_idx):
        # Should always be (1, prompt_len)
        prompt = prompt[0]

        # Sample a sentence and get the reward
        generated_text, log_pf, log_pterm, log_r, log_r_unpenalized = self.forward(
            prompt
        )

        # Get the GFN loss
        loss = modified_subtb_loss(
            log_pf=log_pf,
            log_r=log_r,
            log_pterm=log_pterm,
            generated_text=generated_text,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=len(prompt),
            subtb_lambda=self.hparams.subtb_lambda,
        )

        # Log metrics
        _, last_log_r, last_log_r_unpenalized, sentence_len = get_termination_vals(
            generated_text=generated_text,
            log_pf=log_pf,
            log_pterm=log_pterm,
            log_r=log_r,
            log_r_unpenalized=log_r_unpenalized,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=len(prompt),
        )
        log_ps = last_log_r * self.reward.temperature
        log_ps_unpenalized = last_log_r_unpenalized * self.reward.temperature
        self.log(
            "val/loss",
            loss,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "val/logR",
            last_log_r.mean(),
            sync_dist=True,
        )
        self.log(
            "val/logP(s) (avg)",
            log_ps.mean(),
            sync_dist=True,
        )
        self.log(
            "val/logP(s) (max)",
            log_ps.max(),
            sync_dist=True,
        )
        self.log(
            "val/logP(s) unpenalized (avg)",
            log_ps_unpenalized.mean(),
            sync_dist=True,
        )
        self.log(
            "val/logP(s) unpenalized (max)",
            log_ps_unpenalized.max(),
            sync_dist=True,
        )
        self.log(
            "val/sentence_len",
            sentence_len.float().mean(),
            sync_dist=True,
        )
        if self.diversity_metric.method is not None:
            generated_sentences = self.tokenizer.batch_decode(
                generated_text[:, len(prompt) :]
            )
            generated_sentences = [
                text.replace(".", "") for text in generated_sentences
            ]
            diversity = self.diversity_metric(generated_sentences)
            self.log(f"val/{self.diversity_metric_name}", diversity, sync_dist=True)

    def on_train_batch_start(self, prompt, batch_idx):
        # Update scheduled quantities
        reward_temp = self.get_reward_temp_at_step(self.global_step)
        lr = self.get_lr_at_step(self.global_step)
        self.reward.temperature = reward_temp
        for pg in self.optimizers().param_groups:
            pg["lr"] = lr

    def on_train_epoch_start(self):
        # Log scheduled quantities
        self.log("scheduled/R_temperature", self.reward.temperature, sync_dist=True)
        self.log("scheduled/lr", self.get_lr_at_step(self.global_step), sync_dist=True)

        # Log probe samples
        if (
            self.hparams.train_probes is not None
            and self.logger is not None
            and self.trainer.current_epoch % 5 == 0
        ):
            samples_table = self.sample_probes(self.hparams.train_probes)
            self.logger.log_table("samples/train_probes", dataframe=samples_table)

    def on_validation_epoch_start(self):
        # Log variance of (logR - logP(s)) using exploration, which should be 0.0
        log_rs, log_pfss = [], []
        val_data = self.trainer.datamodule.val_dataloader().dataset
        for prompt in val_data:
            prompt = prompt[0]
            generated_text, log_pf, log_pterm, log_r, log_r_unpenalized = self.forward(
                prompt.to(self.device), pf_temperature=2.0
            )
            log_pfs, log_r, _, _ = get_termination_vals(
                generated_text=generated_text,
                log_pf=log_pf,
                log_pterm=log_pterm,
                log_r=log_r,
                log_r_unpenalized=log_r_unpenalized,
                termination_token_id=self.end_of_sentence_token_id,
                prompt_len=len(prompt),
            )
            log_rs.append(log_r)
            log_pfss.append(log_pfs)
        log_rs, log_pfss = torch.cat(log_rs), torch.cat(log_pfss)
        self.log("val/Var(logR - logPf(s))", (log_rs - log_pfss).var(), sync_dist=True)

        # Log probe samples
        if (
            self.hparams.val_probes is not None
            and self.logger is not None
            and self.trainer.current_epoch % 5 == 0
        ):
            samples_table = self.sample_probes(self.hparams.val_probes)
            self.logger.log_table("samples/val_probes", dataframe=samples_table)

    def on_train_start(self):
        # Log baseline metrics
        val_data = self.trainer.datamodule.val_dataloader().dataset
        baseline_performance = None
        for prompt in val_data:
            prompt = prompt[0]
            samples = self.sample_baselines(
                prompt.to(self.device), n_samples=self.hparams.n_samples
            )
            if baseline_performance is None:
                baseline_performance = pd.DataFrame(
                    data=np.zeros((6, len(samples))),
                    columns=samples.keys(),
                    index=[
                        "logP(s) (avg)",
                        "logP(s) (max)",
                        "logP(s) unpenalized (avg)",
                        "logP(s) unpenalized (max)",
                        self.diversity_metric_name,
                        "sentence length",
                    ],
                )
            for baseline in samples:
                baseline_performance.loc["logP(s) (avg)", baseline] += samples[
                    baseline
                ]["logP(s)"].mean().item() / len(val_data)
                baseline_performance.loc["logP(s) (max)", baseline] += samples[
                    baseline
                ]["logP(s)"].max().item() / len(val_data)
                baseline_performance.loc[
                    "logP(s) unpenalized (avg)", baseline
                ] += samples[baseline]["logP(s) unpenalized"].mean().item() / len(
                    val_data
                )
                baseline_performance.loc[
                    "logP(s) unpenalized (max)", baseline
                ] += samples[baseline]["logP(s) unpenalized"].max().item() / len(
                    val_data
                )
                if samples[baseline][self.diversity_metric_name] is None:
                    baseline_performance.loc[
                        self.diversity_metric_name, baseline
                    ] = None
                else:
                    baseline_performance.loc[
                        self.diversity_metric_name, baseline
                    ] += samples[baseline][self.diversity_metric_name] / len(val_data)
                baseline_performance.loc["sentence length", baseline] += samples[
                    baseline
                ]["sentence length"].float().mean().item() / len(val_data)
        baseline_performance = baseline_performance.reset_index(names="metric")
        if self.logger is not None:
            self.logger.log_table(
                "val/baseline performance", dataframe=baseline_performance
            )

        # Log baseline probes
        if self.hparams.val_probes is not None and self.logger is not None:
            samples_table = self.sample_probes_baselines(self.hparams.val_probes)
            self.logger.log_table(
                "samples/val_probes (baselines)", dataframe=samples_table
            )

    def sample_probes(self, probes, n_samples=4):
        assert isinstance(probes, list) and probes[0].ndim == 1
        samples = []
        for probe in probes:
            probe_str = self.tokenizer.decode(probe)
            with torch.no_grad():
                generated_text, _, _, log_r, log_r_unpenalized = self.forward(
                    probe.to(self.device), n_samples=n_samples
                )
            log_ps, log_ps_unpenalized = get_termination_vals(
                generated_text=generated_text,
                log_pf=None,
                log_pterm=None,
                log_r=log_r,
                log_r_unpenalized=log_r_unpenalized,
                termination_token_id=self.end_of_sentence_token_id,
                prompt_len=len(probe),
            )[1:3]
            log_ps *= self.reward.temperature
            log_ps_unpenalized *= self.reward.temperature
            generated_text = generated_text[:, len(probe) :]
            generated_text = self.tokenizer.batch_decode(generated_text)
            generated_text = [text.replace(".", "") for text in generated_text]
            for i in range(len(generated_text)):
                samples.append(
                    {
                        "Prompt": probe_str,
                        "Sampled sentence": generated_text[i],
                        "logP(s)": log_ps[i].item(),
                        "logP(s) unpenalized": log_ps_unpenalized[i].item(),
                    }
                )
        samples = pd.DataFrame(samples)
        samples = samples.sort_values(by=["Prompt", "logP(s)"], ascending=False)
        return samples

    def sample_probes_baselines(self, probes, n_samples=4):
        assert isinstance(probes, list) and probes[0].ndim == 1
        samples = []
        for probe in probes:
            probe_str = self.tokenizer.decode(probe)
            probe_samples = self.sample_baselines(
                probe.to(self.device), n_samples=n_samples
            )
            for i in range(n_samples):
                sample = {"Prompt": probe_str}
                for baseline in probe_samples:
                    sample[f"Sampled sentence ({baseline})"] = probe_samples[baseline][
                        "sample"
                    ][i]
                    sample[f"logP(s) ({baseline})"] = probe_samples[baseline][
                        "logP(s)"
                    ][i].item()
                    sample[f"logP(s) unpenalized ({baseline})"] = probe_samples[
                        baseline
                    ]["logP(s) unpenalized"][i].item()
                samples.append(sample)

        samples = pd.DataFrame(samples)
        samples = samples.sort_values(by=["Prompt"], ascending=False)

        return samples

    def sample_baselines(self, prompt, n_samples=4):
        # https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/text_generation#transformers.GenerationMixin.generate
        # https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/text_generation#transformers.GenerationConfig
        assert prompt.ndim == 1
        prompt = prompt.unsqueeze(0)

        def generate(prompt, **kwargs):
            with torch.no_grad():
                lora_to_base(self.model)
                generated_text = self.model.generate(
                    prompt,
                    min_new_tokens=self.hparams.min_sentence_len,
                    max_new_tokens=self.hparams.max_sentence_len + 1,
                    eos_token_id=self.end_of_sentence_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    forced_eos_token_id=self.end_of_sentence_token_id,
                    suppress_tokens=torch.from_numpy(self.hparams.illegal_token_mask)
                    .nonzero()
                    .squeeze(-1),
                    **kwargs,
                )
                base_to_lora(self.model)

                log_r, log_r_unpenalized = self.reward.score(
                    generated_text,
                    prompt_length=prompt.shape[1],
                    model=self.model,
                    tokenizer=self.tokenizer,
                )
                (
                    _,
                    last_log_r,
                    last_log_r_unpenalized,
                    sentence_len,
                ) = get_termination_vals(
                    generated_text=generated_text,
                    log_pf=None,
                    log_pterm=None,
                    log_r=log_r,
                    log_r_unpenalized=log_r_unpenalized,
                    termination_token_id=self.end_of_sentence_token_id,
                    prompt_len=prompt.shape[1],
                )
                log_ps = last_log_r * self.reward.temperature
                log_ps_unpenalized = last_log_r_unpenalized * self.reward.temperature

            generated_text = generated_text[:, prompt.shape[1] :]
            generated_text = torch.where(
                generated_text == self.tokenizer.eos_token_id,
                self.end_of_sentence_token_id,
                generated_text,
            )
            generated_text = self.tokenizer.batch_decode(generated_text)
            generated_text = [text.replace(".", "") for text in generated_text]

            if len(generated_text) > 1:
                diversity = self.diversity_metric(generated_text)
            else:
                diversity = None

            if len(generated_text) == 1:
                generated_text = generated_text * n_samples
                log_ps = log_ps.expand(n_samples, -1)
                log_ps_unpenalized = log_ps_unpenalized.expand(n_samples, -1)

            return {
                "sample": generated_text,
                "logP(s)": log_ps,
                "logP(s) unpenalized": log_ps_unpenalized,
                "sentence length": sentence_len,
                self.diversity_metric_name: diversity,
            }

        samples = {}

        # Beam search
        samples["beam"] = generate(
            prompt=prompt,
            do_sample=False,
            num_beams=n_samples * 5,
            length_penalty=0.0,
        )
        samples["beam [fair]"] = generate(
            prompt=prompt,
            do_sample=False,
            num_beams=n_samples,
            length_penalty=0.0,
        )

        # Diverse beam search
        samples["diverse beam"] = generate(
            prompt=prompt,
            num_beams=n_samples * 5,
            num_beam_groups=n_samples,
            num_return_sequences=n_samples,
            diversity_penalty=1.0,
            length_penalty=0.0,
        )
        samples["diverse beam [fair]"] = generate(
            prompt=prompt,
            num_beams=n_samples,
            num_beam_groups=n_samples,
            num_return_sequences=n_samples,
            diversity_penalty=1.0,
            length_penalty=0.0,
        )

        # Nucleaus sampling
        samples["nucleus"] = generate(
            prompt=prompt,
            do_sample=True,
            num_return_sequences=n_samples,
            top_k=0,
            top_p=0.95,
        )

        # LM
        samples["LM"] = generate(
            prompt=prompt,
            do_sample=True,
            num_return_sequences=n_samples,
            top_k=0,
        )

        # LM with temperature
        samples["LM tempered"] = generate(
            prompt=prompt,
            do_sample=True,
            num_return_sequences=n_samples,
            top_k=0,
            temperature=self.hparams.reward_temp_end,
        )

        # Greedy
        samples["greedy"] = generate(
            prompt=prompt,
            do_sample=False,
        )

        return samples

    def configure_optimizers(self):
        if self.hparams.use_4bit:
            import bitsandbytes as bnb  # fmt: skip
            return bnb.optim.PagedAdamW8bit(self.model.parameters(), lr=self.hparams.lr)
        else:
            return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
