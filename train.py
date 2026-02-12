from __future__ import annotations

import argparse

import torch

from higs.env import DroneSwarmEnv, EnvConfig
from higs.trainer import PPOTrainer, TrainConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    torch.manual_seed(42)

    env = DroneSwarmEnv(EnvConfig(), device=args.device)
    trainer = PPOTrainer(env, TrainConfig(steps=args.steps), device=args.device)
    rewards = trainer.train(args.steps)

    print(f"Train done. reward_mean={sum(rewards)/len(rewards):.4f}, reward_last={rewards[-1]:.4f}")


if __name__ == "__main__":
    main()
