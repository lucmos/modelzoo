import torch.nn.functional as F
from torch import nn


class L2Norm(nn.Module):
    def __init__(self, p: int = 2, dim: int = -1, eps: float = 1e-12):
        super().__init__()
        self.p = p
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=self.dim, eps=self.eps)
