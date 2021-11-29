import torch


def get_mask(values: torch.Tensor, pad_idx: int = 0):
    return values != pad_idx
