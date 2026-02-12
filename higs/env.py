from __future__ import annotations

from dataclasses import dataclass

import torch

from .numerics import clip_obs, finite_or_zero


@dataclass
class EnvConfig:
    n_agents: int = 12
    n_tasks: int = 4
    comm_radius: float = 3.0
    world_size: float = 10.0
    dt: float = 0.1
    max_speed: float = 1.2
    target_noise: float = 0.1


class DroneSwarmEnv:
    def __init__(self, config: EnvConfig | None = None, device: str = "cpu"):
        self.cfg = config or EnvConfig()
        self.device = torch.device(device)
        self.n_agents = self.cfg.n_agents
        self.n_tasks = self.cfg.n_tasks
        self.reset()

    def reset(self) -> torch.Tensor:
        self.pos = (torch.rand(self.n_agents, 2, device=self.device) - 0.5) * self.cfg.world_size
        self.vel = torch.zeros(self.n_agents, 2, device=self.device)
        self.targets = (torch.rand(self.n_tasks, 2, device=self.device) - 0.5) * self.cfg.world_size
        return self.get_obs()

    def communication_graph(self) -> torch.Tensor:
        delta = self.pos[:, None, :] - self.pos[None, :, :]
        dist = torch.linalg.norm(delta, dim=-1)
        adj = (dist <= self.cfg.comm_radius).float()
        adj.fill_diagonal_(1.0)
        return adj

    def get_obs(self) -> torch.Tensor:
        target_center = self.targets.mean(dim=0, keepdim=True)
        rel = self.pos - target_center
        obs = torch.cat([self.pos, self.vel, rel], dim=-1)
        return clip_obs(obs)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions = finite_or_zero(actions).clamp(-1.0, 1.0)
        self.vel = (self.vel + actions * self.cfg.dt).clamp(-self.cfg.max_speed, self.cfg.max_speed)
        self.pos = self.pos + self.vel * self.cfg.dt
        self.pos = self.pos.clamp(-self.cfg.world_size, self.cfg.world_size)

        noise = torch.randn_like(self.targets) * self.cfg.target_noise
        self.targets = (self.targets + noise).clamp(-self.cfg.world_size, self.cfg.world_size)

        d = torch.cdist(self.pos, self.targets)
        min_d = d.min(dim=1).values
        reward = -min_d.mean()
        collision_penalty = (torch.cdist(self.pos, self.pos) < 0.3).float().sum() / (self.n_agents**2)
        reward = reward - 0.2 * collision_penalty
        reward = finite_or_zero(reward)
        return self.get_obs(), reward
