import torch

from higs.env import DroneSwarmEnv, EnvConfig
from higs.trainer import PPOTrainer, TrainConfig


def test_training_no_nan_short_run():
    torch.manual_seed(0)
    env = DroneSwarmEnv(EnvConfig(n_agents=8, n_tasks=3))
    trainer = PPOTrainer(env, TrainConfig(steps=256, horizon=32))
    rewards = trainer.train(256)

    t = torch.tensor(rewards)
    assert torch.isfinite(t).all()
