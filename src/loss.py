import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AEL_WEIGHTS, DEVICE


class AsymmetricEndoscopyLoss(nn.Module):
    """Weighted cross-entropy encoding ACG/ESGE clinical cost asymmetry."""
    def __init__(self, weights=None):
        super().__init__()
        w = torch.tensor(weights if weights is not None else AEL_WEIGHTS,
                         dtype=torch.float32).to(DEVICE)
        self.ce = nn.CrossEntropyLoss(weight=w)

    def forward(self, logits, targets):
        return self.ce(logits, targets)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()
