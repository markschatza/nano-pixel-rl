from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from nano_pixel_rl.benchmark.evaluate import evaluate
from nano_pixel_rl.benchmark.logging import git_revision, hardware_summary, write_artifacts
from nano_pixel_rl.benchmark.rollout import make_random_batch
from nano_pixel_rl.env.pixelpong import EnvConfig
from nano_pixel_rl.learner import Learner
from nano_pixel_rl.reference.config import TrainConfig, learner_config_for_size


def run_speedrun(train_config: TrainConfig, model_size: str, out_dir: str):
    env_config = EnvConfig(max_steps=train_config.max_episode_steps)
    learner = Learner(learner_config_for_size(model_size), env_config)
    key = jax.random.PRNGKey(train_config.seed)
    learner_state = learner.init(key)
    update_seconds = 0.0
    train_metrics = {}

    for step_idx in range(train_config.steps):
        key, batch_key = jax.random.split(key)
        batch = make_random_batch(batch_key, train_config.num_envs, env_config)
        start = time.perf_counter()
        learner_state, metrics = learner.update(learner_state, batch)
        jax.block_until_ready(learner_state.step)
        update_seconds += time.perf_counter() - start
        if step_idx == train_config.steps - 1:
            train_metrics = {name: float(jnp.asarray(value)) for name, value in metrics.items()}

    eval_result = evaluate(
        learner,
        learner_state,
        episodes=train_config.eval_episodes,
        max_steps=train_config.max_episode_steps,
        seed=train_config.seed + 10_000,
        env_config=env_config,
    )
    result = {
        "git_revision": git_revision(),
        "seed": train_config.seed,
        "hardware": hardware_summary(),
        "config": {
            "steps": train_config.steps,
            "num_envs": train_config.num_envs,
            "rollout_steps": train_config.rollout_steps,
            "eval_episodes": train_config.eval_episodes,
            "max_episode_steps": train_config.max_episode_steps,
            "model_size": model_size,
        },
        "update_time_seconds": update_seconds,
        "threshold_reached": bool(eval_result["threshold_reached"]),
        "eval": eval_result,
        "train_metrics": train_metrics,
    }
    json_path, md_path = write_artifacts(out_dir, result)
    return result, json_path, md_path
