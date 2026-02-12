from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from .numerics import finite_or_zero


class SharedPolicyValue(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.7))
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.backbone(obs)
        mu = torch.tanh(self.mu(h))
        log_std = self.log_std.clamp(-4.0, 1.0)
        std = torch.exp(log_std).clamp(1e-4, 2.0)
        val = self.v(h).squeeze(-1)
        return finite_or_zero(mu), finite_or_zero(std), finite_or_zero(val)

    def sample_action(self, obs: torch.Tensor):
        mu, std, val = self(obs)
        dist = Normal(mu, std)
        a = dist.rsample()
        logp = dist.log_prob(a).sum(-1)
        return a.clamp(-1.0, 1.0), finite_or_zero(logp), val
