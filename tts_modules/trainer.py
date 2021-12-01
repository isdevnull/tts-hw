import PIL
import torch.nn as nn
import torch
import logging
from pathlib import Path
import wandb
import numpy as np
import matplotlib.pyplot as plt
import sys

from tts_modules.utils import plot_image_to_buf
from tts_modules.Vocoder.waveglow import Vocoder

logger = logging.getLogger(__name__)

root = Path(__name__).parent.parent

sys.path.append(root / "tts_modules" / "waveglow")


class FastSpeechTrainer:
    def __init__(self, model, optimizer, scheduler, aligner, featurizer, config, resume, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.aligner = aligner
        self.featurizer = featurizer
        self.mel_loss = nn.MSELoss()
        self.duration_loss = nn.MSELoss()
        self.step = 0
        self.epoch = 0
        self.config = config
        self.params = config["train_params"]
        self.device = device

        if self.params["log_audio"]:
            self.Vocoder = Vocoder().to(device).eval()

        if resume is not None:
            self._load_checkpoint(resume)

    def _load_checkpoint(self, checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]

    def _save_checkpoint(self):
        logger.info(f"step {self.step}: Saving checkpoint...")
        state = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step + 1,
            "epoch": self.epoch
        }
        checkpoint_dir = root / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state, checkpoint_dir / f"checkpoint-step-{self.step}")

    def step(self):

        reference_mel_specs = self.featurizer(batch.waveform)
        max_timeframe_length = reference_mel_specs.size(-1)
        mel_durations = batch.durations * max_timeframe_length,
        batch.durations = self.aligner(wavs=batch.waveform.to(self.device), wav_lengths=batch.waveform_length,
                                       texts=batch.transcript)
        pred_mel_specs, pred_log_durations = self.model(batch.tokens.to(self.device),
                                                        teacher_durations=mel_durations,
                                                        mel_spec_length=max_timeframe_length)
        mel_loss = self.mel_loss(pred_mel_specs, reference_mel_specs)
        dur_loss = self.duration_loss(pred_log_durations, mel_durations)
        loss = mel_loss + dur_loss
        self.step += 1

        return {
            "predicted_spectrogram": pred_mel_specs,
            "predicted_log_durations": pred_log_durations,
            "mel_loss": mel_loss,
            "dur_loss": dur_loss,
            "loss": mel_loss + dur_loss
        }

    def train_epoch(self, train_dataloader):
        self.model.train()

        for batch in train_dataloader:
            step_results = self.step(batch)

        if self.step % self.params["logging_step"] == 0:
            if self.scheduler is not None:
                step_results["cur_lr"] = self.scheduler.get_lr()
            logging.info(
                f"step {self.step}: loss = {step_results['loss']} | "
                f"mel_loss = {step_results['mel_loss']} | "
                f"dur_loss = {step_results['dur_loss']} "
            )
            wandb.log(
                {
                    "loss": step_results["loss"],
                    "mel_loss": step_results["mel_loss"],
                    "dur_loss": step_results["dur_loss"],
                    "learning_rate": step_results["cur_lr"]
                }
            )
            if self.config["return_attention"]:
                idx = np.random.choice(self.config["batch_size"], replace=False)
                for i, attention_score in enumerate(self.model.attention_scores):
                    for head in range(2):
                        image = PIL.Image(plot_image_to_buf(attention_score[idx, head, :, :].cpu().numpy()))
                        wandb.log({
                            f"Attention-{i}-head-{head}": wandb.Image(image)
                        })

        if self.step % self.params["checkpoint_interval"] == 0:
            self._save_checkpoint()

        step_results["loss"].backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step()

    @torch.no_grad()
    def validation_epoch(self, val_dataloader):
        self.model.eval()

        metric_tracker = {
            "loss": [],
            "mel_loss": [],
            "dur_loss": []
        }
        if self.params["log_audio"]:
            random_idx = np.random.choice(self.config["batch_size"])
            predicted_spectrogram = None

        for i, batch in enumerate(val_dataloader):
            step_results = self.step(batch)

            if self.params["log_audio"] and i == random_idx:
                predicted_spectrogram = step_results["predicted_spectrogram"]
            metric_tracker["loss"].append(step_results["loss"])
            metric_tracker["mel_loss"].append(step_results["mel_loss"])
            metric_tracker["dur_loss"].append(step_results["dur_loss"])

        loss_avg = metric_tracker["loss"].sum() / len(metric_tracker["loss"])
        mel_loss_avg = metric_tracker["mel_loss"].sum() / len(metric_tracker["mel_loss"])
        dur_loss_avg = metric_tracker["dur_loss"].sum() / len(metric_tracker["dur_loss"])

        logger.info(
            f"val_loss = {loss_avg} | "
            f"val_mel_loss = {mel_loss_avg} | "
            f"val_dur_loss = {dur_loss_avg}"
        )

        wandb.log({
            "val_loss": loss_avg,
            "val_mel_loss": mel_loss_avg,
            "val_dur_loss": dur_loss_avg
        })

        if self.params["log_audio"]:
            reconstructed_wav = self.Vocoder.inference(predicted_spectrogram).cpu()
            original_waveform = batch.waveform[random_idx]
            plt.plot(reconstructed_wav.squeeze(), label='reconstructed', alpha=.5)
            plt.plot(original_waveform.squeeze(), label='GT', alpha=.5)
            plt.grid()
            plt.legend()
            wandb.log({"Waveform Comparison", plt})
            plt.clf()
            sample_rate = self.featurizer.get_config.sample_rate
            wandb.log({
                "Original Audio": wandb.Audio(original_waveform.squeeze().cpu().numpy(), sample_rate=sample_rate),
                "Reconstructed Audio": wandb.Audio(reconstructed_wav.squeeze().cpu().numpy(), sample_rate=sample_rate)
            })

    def train(self, train_dataloader, valid_dataloader):
        for epoch in range(self.config["epochs"]):
            self.epoch = epoch
            logger.info(f"Starting training on epoch {self.epoch}...")
            self.train_epoch(train_dataloader)
            self.validation_epoch(valid_dataloader)
            logger.info(f"Finished training on epoch {self.epoch}...")
