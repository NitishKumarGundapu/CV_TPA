import torch
import torch.nn.functional as F


def accuracy(logits, y):
    return (logits.argmax(-1) == y).float().mean()



