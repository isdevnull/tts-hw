import torch


def get_mask(values: torch.Tensor, pad_value=0):
    return values != pad_value
