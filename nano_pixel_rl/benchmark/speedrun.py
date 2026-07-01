from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from nano_pixel_rl.benchmark.evaluate import evaluate
from nano_pixel_rl.benchmark.logging import git_revision, hardware_summary, write_artifacts
from nano_pixel_rl.benchmark.rollout import make_training_batch
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
    last_progress_seconds = 0.0

    step_idx = 0
    while True:
        if train_config.duration_seconds is None:
            if step_idx >= train_config.steps:
                break
        elif update_seconds >= train_config.duration_seconds and step_idx > 0:
            break
        key, batch_key = jax.random.split(key)
        batch = make_training_batch(batch_key, train_config.num_envs, env_config)
        start = time.perf_counter()
        learner_state, metrics = learner.update(learner_state, batch)
        jax.block_until_ready(learner_state.step)
        update_seconds += time.perf_counter() - start
        train_metrics = {name: float(jnp.asarray(value)) for name, value in metrics.items()}
        step_idx += 1
        if update_seconds - last_progress_seconds >= 300:
            print(
                "progress "
                f"steps={step_idx} update_seconds={update_seconds:.1f} "
                f"loss={train_metrics.get('loss', 0.0):.4f} "
                f"paddle_window_loss={train_metrics.get('paddle_window_loss', 0.0):.4f}",
                flush=True,
            )
            last_progress_seconds = update_seconds

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
            "completed_steps": step_idx,
            "duration_seconds": train_config.duration_seconds,
            "num_envs": train_config.num_envs,
            "rollout_steps": train_config.rollout_steps,
            "eval_episodes": train_config.eval_episodes,
            "max_episode_steps": train_config.max_episode_steps,
            "model_size": model_size,
        },
        "update_time_seconds": update_seconds,
        "completed_run": train_config.duration_seconds is None or update_seconds >= train_config.duration_seconds,
        "threshold_reached": bool(eval_result["threshold_reached"]),
        "eval": eval_result,
        "train_metrics": train_metrics,
    }
    json_path, md_path = write_artifacts(out_dir, result)
    return result, json_path, md_path
