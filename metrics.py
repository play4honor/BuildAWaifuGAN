import torch

def calc_overfit_metric(scores: torch.Tensor):

    return torch.sign(scores).mean().item()
    