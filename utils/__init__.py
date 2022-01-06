from .architectures import get_architecture
from .my_loader import MyCustomDataset
from .ensemble_model import Input_diversity, MultiEnsemble

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    cross entropy loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        return F.cross_entropy(logits, labels, reduction='none')


class MarginLoss(nn.Module):
    """
    top-5 margin loss
    """

    def __init__(self, kappa=float('inf'), k = 5):
        super().__init__()
        self.kappa = kappa
        self.k = k

    def forward(self, logits, labels, conf=1):
        onehot_label = F.one_hot(labels, num_classes=1000).float()
        true_logit5 = torch.sum(logits * onehot_label, dim=-1, keepdims=True)
        wrong_logit5, _idx = torch.topk(logits * (1-onehot_label) - onehot_label * 1e7, k=self.k, dim = 1)
        target_loss5 = torch.sum(F.relu(true_logit5 - wrong_logit5 + conf), dim = 1)
        return target_loss5