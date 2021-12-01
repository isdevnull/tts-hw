from typing import Union, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
import math


def attention(
        key: torch.Tensor,
        value: torch.Tensor,
        query: torch.Tensor,
        mask=None,
        return_attention: bool = False) -> \
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    d_k = query.size(-1)
    score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(mask == 0, -5e-12)
    attention_score = F.softmax(score, dim=-1)
    output = torch.matmul(attention_score, value)
    return output, attention_score if return_attention else None


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, d_k: int, d_v: int, return_attention: bool = False):
        super().__init__()
        self.return_attention = return_attention
        self.attention_score = None
        self.n_heads, self.d_model, self.d_k, self.d_v = n_heads, d_model, d_k, d_v
        self.kqv_weights = nn.ModuleList(
            [
                nn.Linear(self.d_model, self.d_k * self.n_heads),
                nn.Linear(self.d_model, self.d_k * self.n_heads),
                nn.Linear(self.d_model, self.d_v * self.n_heads)
            ]
        )
        self.final_proj = nn.Linear(self.n_heads * self.d_v, self.d_model)

    @property
    def get_attention_matrix(self):
        return self.attention_score

    def forward(self, key: torch.Tensor, value: torch.Tensor, query: torch.Tensor, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        key_proj, query_proj, value_proj = \
            [
                weight_proj(param).view(batch_size, self.n_heads, -1, self.d_k) for weight_proj, param in
                zip(self.kqv_weights, (key, query, value))
            ]
        output_values, self.attention_score = attention(key_proj, value_proj, query_proj, mask=mask,
                                                        return_attention=self.return_attention)
        if self.attention_score is not None:
            self.attention_score.requires_grad = False
        outputs = self.kqv_weights[-1](output_values.view(batch_size, -1, self.n_heads * self.d_v))
        return outputs
