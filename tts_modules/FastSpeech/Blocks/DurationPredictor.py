import torch.nn as nn


class ConvolutionWrapper(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, *args, **kwargs)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class ConvNetDurationPredictor(nn.Module):
    def __init__(self, d_model: int, p_dropout: float, *args, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            ConvolutionWrapper(in_channels=d_model, out_channels=d_model, padding=1, *args, **kwargs),
            nn.LayerNorm(d_model),
            nn.Dropout(p=p_dropout),
            nn.ReLU(inplace=True),
            ConvolutionWrapper(in_channels=d_model, out_channels=d_model, padding=1, *args, **kwargs),
            nn.LayerNorm(d_model),
            nn.Dropout(p=p_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=d_model, out_features=1)
        )

    def forward(self, x):
        return self.net(x)
