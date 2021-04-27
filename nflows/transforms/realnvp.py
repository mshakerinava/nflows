"""Implementations of Real NVP."""

import torch
from torch import nn

from nflows.transforms.base import Transform
from nflows.utils import torchutils


class RealNVP(Transform):
    def __init__(self, D, d, hidden):
        assert d > 0
        assert D > d
        assert hidden > 0
        super().__init__()
        self.D = D
        self.d = d
        self.hidden = hidden
        self.s_net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, D - d)
        )
        self.t_net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, D - d)
        )

    def forward(self, x, context=None):
        x1, x2 = x[:, :self.d], x[:, self.d:] 
        s = self.s_net(x1)
        z1, z2 = x1, x2 * torch.exp(s) + self.t_net(x1)
        z = torch.cat([z1, z2], dim=-1)
        logabsdet = torchutils.sum_except_batch(s, num_batch_dims=1)
        return z, logabsdet

    def inverse(self, z, context=None):
        z1, z2 = z[:, :self.d], z[:, self.d:] 
        x1 = z1
        s = self.s_net(z1)
        x2 = (z2 - self.t_net(z1)) * torch.exp(-s)
        logabsdet = -torchutils.sum_except_batch(s, num_batch_dims=1)
        return torch.cat([x1, x2], -1), logabsdet
