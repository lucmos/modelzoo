import logging
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

log = logging.getLogger(__name__)


class SumAggregation(nn.Module):
    def __init__(self, subspace_dim: int, num_subspaces: int):
        super().__init__()

        self.subspace_dim = subspace_dim
        self.num_subspaces = num_subspaces

        log.info(f"{__class__.__name__}: subspace_dim={self.subspace_dim}, num_subspaces={self.num_subspaces}")

    @property
    def out_dim(self):
        return self.subspace_dim

    def forward(self, concat_subspaces: Sequence[torch.Tensor]) -> torch.Tensor:
        concat_subspaces = concat_subspaces.split(self.subspace_dim, dim=1)
        out = [norm_layer(x) for norm_layer, x in zip(self.norm_layers, concat_subspaces)]
        return torch.stack(out, dim=1).sum(dim=1)


class NonLinearSumAggregation(SumAggregation):
    def __init__(self, subspace_dim: int, num_subspaces: int):
        super().__init__(subspace_dim, num_subspaces)

        self.norm_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(subspace_dim),
                    nn.Linear(subspace_dim, subspace_dim),
                    nn.Tanh(),
                )
                for _ in range(num_subspaces)
            ]
        )
