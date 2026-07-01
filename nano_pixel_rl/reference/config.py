from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LearnerConfig:
    model_size: str = "tiny"
    learning_rate: float = 3e-4
    prediction_weight: float = 0.05
    hidden_dim: int = 64
    context_frames: int = 1


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    steps: int = 200
    num_envs: int = 128
    rollout_steps: int = 32
    eval_episodes: int = 64
    max_episode_steps: int = 256


def learner_config_for_size(model_size: str) -> LearnerConfig:
    sizes = {
        "tiny": LearnerConfig(model_size="tiny", hidden_dim=64),
        "small": LearnerConfig(model_size="small", hidden_dim=128),
    }
    if model_size not in sizes:
        raise ValueError(f"Unknown model_size {model_size!r}; expected one of {sorted(sizes)}")
    return sizes[model_size]
