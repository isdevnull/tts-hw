import torch
import torch.nn.functional as F
import torch.nn as nn
import math


def attention(key: torch.Tensor, value: torch.Tensor, query: torch.Tensor, mask=None):
    d_k = query.size(-1)
    score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask:
        score = score.masked_fill_(mask == 0, -5e-12)
    attention_score = F.softmax(score, dim=-1)
    output = torch.matmul(attention_score, value)
    return output


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, d_k: int, d_v: int):
        super().__init__()
        self.n_heads, self.d_model, self.d_k, self.d_v = n_heads, d_model, d_k, d_v
        self.kqv_weights = nn.ModuleList(
            [
                nn.Linear(self.d_model, self.d_k * self.n_heads),
                nn.Linear(self.d_model, self.d_k * self.n_heads),
                nn.Linear(self.d_model, self.d_v * self.n_heads)
            ]
        )
        self.final_proj = nn.Linear(self.n_heads * self.d_v, self.d_model)

    def forward(self, key: torch.Tensor, value: torch.Tensor, query: torch.Tensor, mask=None):
        batch_size = query.size(0)
        key_proj, query_proj, value_proj = \
            [
                weight_proj(param).view(batch_size, self.n_heads, -1, param.size(-1)) for weight_proj, param in
                zip(self.kqv_weights, (key, query, value))
            ]
        output_values = attention(key_proj, value_proj, query_proj, mask=mask)
        outputs = self.kqv_weights[-1](output_values.view(batch_size, -1, self.n_heads * self.d_v))
        return outputs
