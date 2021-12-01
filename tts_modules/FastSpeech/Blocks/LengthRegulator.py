import torch.nn as nn
import torch
import torch.nn.functional as F

from tts_modules.FastSpeech.Blocks.DurationPredictor import ConvNetDurationPredictor


class LengthRegulator(nn.Module):
    def __init__(self, d_model: int, p_dropout: float, alpha: float = 1.0, n_mels: int = 80, *args, **kwargs):
        super().__init__()
        self.alpha = alpha  # stretching parameter
        self.n_mels = n_mels
        self.duration_predictor = ConvNetDurationPredictor(d_model=d_model, p_dropout=p_dropout, *args, **kwargs)

    def forward(self, x, teacher_durations: torch.Tensor = None, mel_spec_length: int = 80):
        log_pred = self.duration_predictor(x)
        if teacher_durations is not None:
            pred_num_timeframes = torch.round(teacher_durations * self.alpha).int().view(x.size(0),
                                                                                                           -1)
        else:
            pred_num_timeframes = torch.round(torch.exp(log_pred) * self.alpha).int().view(x.size(0), -1)
        x = torch.stack(
            [F.pad(x[i, ...].repeat_interleave(pred_num_timeframes[i, ...], dim=0),
                   (0, 0, 0, mel_spec_length - pred_num_timeframes[i].sum()), value=0.0) for i in range(x.size(0))],
            dim=0
        )
        return x, log_pred
