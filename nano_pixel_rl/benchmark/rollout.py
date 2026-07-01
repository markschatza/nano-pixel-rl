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
    return make_training_batch(key, batch_size, env_config)


def _track_player_delta(state, env_config: EnvConfig):
    center = state.player_y + env_config.paddle_height // 2
    return jnp.sign(state.ball_y - center).astype(jnp.int32)


def _randomize_state(key, env_config: EnvConfig):
    state = reset(key, env_config)
    keys = jax.random.split(key, 12)

    def body(s, k):
        player_delta = jax.random.randint(k, (), -1, 2, dtype=jnp.int32)
        opponent = jnp.sign(s.ball_y - (s.opponent_y + env_config.paddle_height // 2)).astype(jnp.int32)
        return step(s, player_delta, opponent, env_config), None

    state, _ = jax.lax.scan(body, state, keys)
    return state


def make_heuristic_batch(key, batch_size: int, env_config: EnvConfig = EnvConfig()):
    return make_training_batch(key, batch_size, env_config)


def make_training_batch(key, batch_size: int, env_config: EnvConfig = EnvConfig()):
    keys = jax.random.split(key, batch_size)
    states = jax.vmap(lambda k: _randomize_state(k, env_config))(keys)
    obs = jax.vmap(lambda s: render_frame(s, env_config))(states)
    player_deltas = jax.vmap(lambda s: _track_player_delta(s, env_config))(states)
    opponent_deltas = jax.vmap(lambda s: jnp.sign(s.ball_y - (s.opponent_y + env_config.paddle_height // 2)).astype(jnp.int32))(states)
    next_states = jax.vmap(lambda s, p, o: step(s, p, o, env_config))(states, player_deltas, opponent_deltas)
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
