from __future__ import annotations

import torch

from .numerics import EPS, finite_or_zero, safe_softmax


class HiGSController:
    """Three-layer controller: grouping -> assignment -> coordination."""

    def __init__(self, n_groups: int = 3, assignment_iters: int = 5):
        self.n_groups = n_groups
        self.assignment_iters = assignment_iters

    def strategic_grouping(self, obs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Graph-aware simple clustering by spectral proxy: degree-weighted anchors
        deg = adj.sum(dim=-1)
        score = deg + 0.1 * obs[:, 0]
        anchors = torch.topk(score, k=min(self.n_groups, obs.shape[0])).indices
        dist = torch.cdist(obs[:, :2], obs[anchors, :2])
        groups = dist.argmin(dim=-1)
        return groups

    def tactical_assignment(self, obs: torch.Tensor, groups: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_agents = obs.shape[0]
        n_tasks = targets.shape[0]
        pos = obs[:, :2]
        payoff = -torch.cdist(pos, targets)
        assign = torch.randint(0, n_tasks, (n_agents,), device=obs.device)

        for _ in range(self.assignment_iters):
            for g in groups.unique():
                idx = (groups == g).nonzero(as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    continue
                g_payoff = payoff[idx]
                congestion = torch.bincount(assign, minlength=n_tasks).float().to(obs.device)
                utility = g_payoff - 0.1 * congestion[None, :]
                probs = safe_softmax(utility, dim=-1)
                new_assign = torch.multinomial(probs, 1).squeeze(-1)
                assign[idx] = new_assign
        return assign

    def execution_guidance(self, obs: torch.Tensor, assign: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target = targets[assign]
        pos = obs[:, :2]
        vel = obs[:, 2:4]
        desired = target - pos
        norm = torch.linalg.norm(desired, dim=-1, keepdim=True).clamp_min(EPS)
        desired_dir = desired / norm
        accel = desired_dir - 0.3 * vel
        return finite_or_zero(accel).clamp(-1.0, 1.0)
