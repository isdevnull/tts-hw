import torch.nn as nn
import math


class GraphemeEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, padding_idx: int, *args, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, self.d_model, padding_idx=padding_idx, *args, **kwargs)

    def forward(self, x):
        return self.embed(x) + math.sqrt(self.d_model)
