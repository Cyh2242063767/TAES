from __future__ import annotations

import torch


EPS = 1e-8


def finite_or_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(x), x, torch.zeros_like(x))


def safe_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = finite_or_zero(x)
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True).clamp_min(EPS)
    out = (x - mean) / std
    return finite_or_zero(out)


def safe_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logits = finite_or_zero(logits)
    logits = logits - logits.amax(dim=dim, keepdim=True)
    exp = torch.exp(logits).clamp_min(EPS)
    denom = exp.sum(dim=dim, keepdim=True).clamp_min(EPS)
    return exp / denom


def clip_obs(obs: torch.Tensor, min_val: float = -20.0, max_val: float = 20.0) -> torch.Tensor:
    return finite_or_zero(obs).clamp(min_val, max_val)
