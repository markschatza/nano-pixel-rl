from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from nano_pixel_rl.benchmark.metrics import cross_entropy, token_accuracy
from nano_pixel_rl.learner.model import ModelParams, forward


class LearnerState(NamedTuple):
    params: ModelParams
    opt_state: optax.OptState
    step: jnp.ndarray


def loss_fn(params, batch):
    logits = forward(params, batch["obs"])
    loss = cross_entropy(logits, batch["target"])
    acc = token_accuracy(logits, batch["target"])
    return loss, {"prediction_loss": loss, "token_accuracy": acc}


def update_state(state: LearnerState, optimizer, batch):
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    metrics = dict(metrics)
    metrics["loss"] = loss
    return LearnerState(params=params, opt_state=opt_state, step=state.step + 1), metrics
