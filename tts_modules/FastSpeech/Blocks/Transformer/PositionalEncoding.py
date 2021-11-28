import math
import torch.nn as nn
import torch


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, p_dropout: float = 0.1, pos_size: int = 5000):
        super().__init__()

        positional_encoding = torch.empty(pos_size, d_model)
        positions = torch.arange(0, pos_size, dtype=torch.float).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000) / d_model))
        positional_encoding[:, 0::2] = torch.sin(positions * divisor)
        positional_encoding[:, 1::2] = torch.cos(positions * divisor)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer("positional_encoding", positional_encoding)

        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        return self.dropout(x + self.positional_encoding[:, :x.size(1)])
