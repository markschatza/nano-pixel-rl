from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from nano_pixel_rl.benchmark.metrics import cross_entropy
from nano_pixel_rl.env.interpreter import interpret_proposal
from nano_pixel_rl.env.opponent import opponent_delta
from nano_pixel_rl.env.pixelpong import EnvConfig, reset, render_frame, step
from nano_pixel_rl.env.rewards import point_reward


class RolloutBatch(NamedTuple):
    obs: jnp.ndarray
    target: jnp.ndarray
    rewards: jnp.ndarray
    invalid: jnp.ndarray
    prediction_loss: jnp.ndarray
    final_states: object


def make_random_batch(key, batch_size: int, env_config: EnvConfig = EnvConfig()):
    keys = jax.random.split(key, batch_size)
    states = jax.vmap(lambda k: reset(k, env_config))(keys)
    obs = jax.vmap(lambda s: render_frame(s, env_config))(states)
    next_states = jax.vmap(lambda s: step(s, 0, 0, env_config))(states)
    target = jax.vmap(lambda s: render_frame(s, env_config))(next_states)
    return {"obs": obs, "target": target}


def rollout_once(learner, learner_state, states, keys, opponent_kind: int, env_config: EnvConfig = EnvConfig()):
    obs = jax.vmap(lambda s: render_frame(s, env_config))(states)
    proposal = learner.propose(learner_state, obs)
    actions = jax.vmap(lambda p, s: interpret_proposal(p, s, env_config))(proposal, states)
    opp = jax.vmap(lambda s, k: opponent_delta(s, k, opponent_kind, env_config))(states, keys)
    next_states = jax.vmap(lambda s, a, o: step(s, a, o, env_config))(states, actions.delta, opp)
    target = jax.vmap(lambda s: render_frame(s, env_config))(next_states)
    logits = learner.logits(learner_state, obs)
    losses = jax.vmap(cross_entropy)(logits, target)
    rewards = jax.vmap(point_reward)(next_states)
    return RolloutBatch(
        obs=obs,
        target=target,
        rewards=rewards,
        invalid=~actions.valid,
        prediction_loss=losses,
        final_states=next_states,
    )
