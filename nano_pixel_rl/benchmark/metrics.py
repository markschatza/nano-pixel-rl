from __future__ import annotations

import jax.numpy as jnp


def token_accuracy(logits, target_frame):
    pred = jnp.argmax(logits, axis=-1).astype(target_frame.dtype)
    return jnp.mean((pred == target_frame).astype(jnp.float32))


def cross_entropy(logits, target_frame, class_weights=None):
    log_probs = logits - jax_logsumexp(logits, axis=-1, keepdims=True)
    gathered = jnp.take_along_axis(log_probs, target_frame[..., None].astype(jnp.int32), axis=-1)[..., 0]
    if class_weights is None:
        return -jnp.mean(gathered)
    weights = jnp.take(jnp.asarray(class_weights, dtype=jnp.float32), target_frame.astype(jnp.int32))
    return -jnp.sum(gathered * weights) / jnp.maximum(jnp.sum(weights), 1.0)


def jax_logsumexp(x, axis=None, keepdims=False):
    max_x = jnp.max(x, axis=axis, keepdims=True)
    out = jnp.log(jnp.sum(jnp.exp(x - max_x), axis=axis, keepdims=True)) + max_x
    if not keepdims:
        out = jnp.squeeze(out, axis=axis)
    return out


def summarize_rollout(player_scores, opponent_scores, invalid_flags, prediction_losses):
    wins = player_scores > opponent_scores
    return {
        "win_rate": float(jnp.mean(wins.astype(jnp.float32))),
        "avg_player_score": float(jnp.mean(player_scores.astype(jnp.float32))),
        "avg_opponent_score": float(jnp.mean(opponent_scores.astype(jnp.float32))),
        "invalid_proposal_rate": float(jnp.mean(invalid_flags.astype(jnp.float32))),
        "prediction_loss": float(jnp.mean(prediction_losses.astype(jnp.float32))),
    }
