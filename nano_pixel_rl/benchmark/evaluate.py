from __future__ import annotations

import jax
import jax.numpy as jnp

from nano_pixel_rl.benchmark.rollout import rollout_once
from nano_pixel_rl.env.opponent import DELAYED_TRACKER, RANDOM_LEGAL
from nano_pixel_rl.env.pixelpong import EnvConfig, reset


def evaluate(learner, learner_state, episodes: int, max_steps: int, seed: int, env_config: EnvConfig = EnvConfig()):
    results = {}
    for name, kind in [("random_legal", RANDOM_LEGAL), ("delayed_tracker", DELAYED_TRACKER)]:
        states = jax.vmap(lambda k: reset(k, env_config))(jax.random.split(jax.random.PRNGKey(seed + kind), episodes))
        invalid_sum = 0.0
        pred_sum = 0.0
        for t in range(max_steps):
            keys = jax.random.split(jax.random.PRNGKey(seed + kind * 1000 + t), episodes)
            batch = rollout_once(learner, learner_state, states, keys, kind, env_config)
            states = batch.final_states
            invalid_sum += float(jnp.mean(batch.invalid.astype(jnp.float32)))
            pred_sum += float(jnp.mean(batch.prediction_loss))
        wins = states.player_score > states.opponent_score
        results[f"win_rate_{name}"] = float(jnp.mean(wins.astype(jnp.float32)))
        results[f"invalid_proposal_rate_{name}"] = invalid_sum / max_steps
        results[f"prediction_loss_{name}"] = pred_sum / max_steps
    results["threshold_reached"] = (
        results["win_rate_random_legal"] >= 0.90 and results["win_rate_delayed_tracker"] >= 0.50
    )
    return results
