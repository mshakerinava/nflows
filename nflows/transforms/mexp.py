"""Implementations of Matrix Exponential Flow."""

import torch
from torch import nn

from nflows.transforms.base import Transform


class ExpLinear(Transform):
    def __init__(self, d):
        super().__init__()
        self.d = d
        dummy = nn.Linear(d, d)
        self.A = nn.Parameter(torch.tensor(sp.linalg.logm(dummy.weight.detach().numpy()), dtype=torch.float32))
        self.b = nn.Parameter(dummy.bias.detach())

    def forward(self, x, context=None):
        W = torch.matrix_exp(self.A)
        z = torch.matmul(x, W) + self.b
        logabsdet = torch.trace(self.A) * x.new_ones(z.shape[0])
        return z, logabsdet

    def inverse(self, z, context=None):
        W_inv = torch.matrix_exp(-self.A)
        x = torch.matmul(Z - self.b, W_inv)
        logabsdet = -torch.trace(self.A) * z.new_ones(x.shape[0])
        return x, logabsdet
