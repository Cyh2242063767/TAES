from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .controller import HiGSController
from .env import DroneSwarmEnv
from .model import SharedPolicyValue
from .numerics import EPS, clip_obs, finite_or_zero, safe_normalize


@dataclass
class TrainConfig:
    steps: int = 3000
    horizon: int = 64
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 3e-4
    clip_ratio: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5


class PPOTrainer:
    def __init__(self, env: DroneSwarmEnv, cfg: TrainConfig | None = None, device: str = "cpu"):
        self.env = env
        self.cfg = cfg or TrainConfig()
        self.device = torch.device(device)

        obs_dim = env.get_obs().shape[-1]
        self.model = SharedPolicyValue(obs_dim=obs_dim, act_dim=2).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.controller = HiGSController()

    def _gae(self, rewards: torch.Tensor, values: torch.Tensor, next_v: torch.Tensor):
        adv = torch.zeros_like(rewards)
        gae = torch.zeros((), device=rewards.device)
        for t in reversed(range(rewards.shape[0])):
            delta = rewards[t] + self.cfg.gamma * next_v - values[t]
            gae = delta + self.cfg.gamma * self.cfg.lam * gae
            adv[t] = gae
            next_v = values[t]
        ret = adv + values
        return safe_normalize(adv, dim=0), ret

    def train(self, steps: int | None = None) -> list[float]:
        total_steps = steps or self.cfg.steps
        obs = self.env.reset().to(self.device)
        rewards_log = []

        for _ in range(total_steps // self.cfg.horizon):
            obs_b, act_b, logp_b, val_b, rew_b = [], [], [], [], []

            for _ in range(self.cfg.horizon):
                obs = clip_obs(obs)
                adj = self.env.communication_graph().to(self.device)
                groups = self.controller.strategic_grouping(obs, adj)
                assign = self.controller.tactical_assignment(obs, groups, self.env.targets.to(self.device))
                guide = self.controller.execution_guidance(obs, assign, self.env.targets.to(self.device))

                a_rl, logp, v = self.model.sample_action(obs)
                action = 0.7 * a_rl + 0.3 * guide

                next_obs, reward = self.env.step(action.detach())
                next_obs = next_obs.to(self.device)
                reward = reward.to(self.device)

                if not torch.isfinite(reward):
                    reward = torch.tensor(0.0, device=self.device)

                obs_b.append(obs)
                act_b.append(action)
                logp_b.append(logp.mean())
                val_b.append(v.mean())
                rew_b.append(reward)
                rewards_log.append(float(reward.item()))

                obs = next_obs

            with torch.no_grad():
                _, _, next_v = self.model(obs)
                next_v = next_v.mean()

            rewards = torch.stack(rew_b)
            values = torch.stack(val_b)
            old_logp = torch.stack(logp_b)
            actions = torch.stack(act_b)
            obss = torch.stack(obs_b)

            adv, ret = self._gae(rewards, values, next_v)
            adv, ret = finite_or_zero(adv), finite_or_zero(ret)

            flat_obs = obss.reshape(-1, obss.shape[-1])
            flat_act = actions.reshape(-1, actions.shape[-1])

            mu, std, v_pred = self.model(flat_obs)
            dist = torch.distributions.Normal(mu, std)
            new_logp = dist.log_prob(flat_act).sum(-1).reshape(self.cfg.horizon, -1).mean(-1)
            entropy = dist.entropy().sum(-1).mean()
            v_pred = v_pred.reshape(self.cfg.horizon, -1).mean(-1)

            ratio = torch.exp((new_logp - old_logp).clamp(-10.0, 10.0))
            clipped = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio)
            policy_loss = -(torch.min(ratio * adv, clipped * adv)).mean()
            value_loss = ((v_pred - ret) ** 2).mean()

            loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy
            if not torch.isfinite(loss):
                continue

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)

            bad_grad = False
            for p in self.model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_grad = True
                    break
            if bad_grad:
                self.opt.zero_grad(set_to_none=True)
                continue

            self.opt.step()

        return rewards_log
