from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from nano_pixel_rl.benchmark.metrics import cross_entropy, token_accuracy
from nano_pixel_rl.env.tokens import PADDLE
from nano_pixel_rl.learner.model import ModelParams, forward
from nano_pixel_rl.learner.proposal import legal_candidate_tops, paddle_window_scores


class LearnerState(NamedTuple):
    params: ModelParams
    opt_state: optax.OptState
    step: jnp.ndarray


def _paddle_window_loss(logits, obs, target):
    player_x = 1
    paddle_height = 3
    current = obs[:, -1, :, :] if obs.ndim == 4 else obs
    current_top = jnp.argmax(current[:, :, player_x] == PADDLE, axis=1)
    target_top = jnp.argmax(target[:, :, player_x] == PADDLE, axis=1)
    candidates = legal_candidate_tops(current, player_x, paddle_height)
    scores = paddle_window_scores(logits, player_x, paddle_height)
    candidate_scores = jnp.take_along_axis(scores, candidates, axis=1)
    target_delta = jnp.clip(target_top - current_top, -1, 1) + 1
    return optax.softmax_cross_entropy_with_integer_labels(candidate_scores, target_delta).mean()


def _policy_loss(logits, obs, action_label, reward):
    player_x = 1
    paddle_height = 3
    current = obs[:, -1, :, :] if obs.ndim == 4 else obs
    candidates = legal_candidate_tops(current, player_x, paddle_height)
    scores = paddle_window_scores(logits, player_x, paddle_height)
    candidate_scores = jnp.take_along_axis(scores, candidates, axis=1)
    log_probs = jax.nn.log_softmax(candidate_scores, axis=1)
    chosen_log_prob = jnp.take_along_axis(log_probs, action_label[:, None].astype(jnp.int32), axis=1)[:, 0]
    centered_reward = reward.astype(jnp.float32) - jnp.mean(reward.astype(jnp.float32))
    return -jnp.mean(chosen_log_prob * centered_reward)


def loss_fn(params, batch, paddle_window_weight: float = 1.0):
    logits = forward(params, batch["obs"])
    prediction_loss = cross_entropy(logits, batch["target"], class_weights=jnp.asarray([0.05, 4.0, 6.0], dtype=jnp.float32))
    paddle_loss = _paddle_window_loss(logits, batch["obs"], batch["target"])
    policy_loss = jnp.asarray(0.0, dtype=jnp.float32)
    if "action_label" in batch and "reward" in batch:
        policy_loss = _policy_loss(logits, batch["obs"], batch["action_label"], batch["reward"])
    loss = prediction_loss + paddle_window_weight * paddle_loss + policy_loss
    acc = token_accuracy(logits, batch["target"])
    return loss, {
        "prediction_loss": prediction_loss,
        "paddle_window_loss": paddle_loss,
        "policy_loss": policy_loss,
        "token_accuracy": acc,
    }


def update_state(state: LearnerState, optimizer, batch, paddle_window_weight: float = 1.0):
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch, paddle_window_weight)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    metrics = dict(metrics)
    metrics["loss"] = loss
    return LearnerState(params=params, opt_state=opt_state, step=state.step + 1), metrics
