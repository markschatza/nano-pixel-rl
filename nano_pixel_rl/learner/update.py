from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from nano_pixel_rl.benchmark.metrics import cross_entropy, token_accuracy
from nano_pixel_rl.env.tokens import PADDLE
from nano_pixel_rl.learner.model import ModelParams, forward


class LearnerState(NamedTuple):
    params: ModelParams
    opt_state: optax.OptState
    step: jnp.ndarray


def _paddle_window_loss(logits, obs, target):
    player_x = 1
    paddle_height = 3
    max_top = logits.shape[1] - paddle_height + 1
    paddle_logits = logits[:, :, player_x, int(PADDLE)]
    all_scores = jnp.stack(
        [jnp.sum(paddle_logits[:, top : top + paddle_height], axis=1) for top in range(max_top)],
        axis=1,
    )
    current_top = jnp.argmax(obs[:, :, player_x] == PADDLE, axis=1)
    target_top = jnp.argmax(target[:, :, player_x] == PADDLE, axis=1)
    candidates = jnp.clip(
        current_top[:, None] + jnp.asarray([-1, 0, 1], dtype=jnp.int32)[None, :],
        0,
        max_top - 1,
    )
    candidate_scores = jnp.take_along_axis(all_scores, candidates, axis=1)
    target_delta = jnp.clip(target_top - current_top, -1, 1) + 1
    return optax.softmax_cross_entropy_with_integer_labels(candidate_scores, target_delta).mean()


def loss_fn(params, batch):
    logits = forward(params, batch["obs"])
    prediction_loss = cross_entropy(logits, batch["target"], class_weights=jnp.asarray([0.05, 4.0, 6.0], dtype=jnp.float32))
    paddle_loss = _paddle_window_loss(logits, batch["obs"], batch["target"])
    loss = prediction_loss + paddle_loss
    acc = token_accuracy(logits, batch["target"])
    return loss, {"prediction_loss": prediction_loss, "paddle_window_loss": paddle_loss, "token_accuracy": acc}


def update_state(state: LearnerState, optimizer, batch):
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    metrics = dict(metrics)
    metrics["loss"] = loss
    return LearnerState(params=params, opt_state=opt_state, step=state.step + 1), metrics
