from __future__ import annotations

import jax.numpy as jnp

from nano_pixel_rl.env.tokens import BACKGROUND, BALL, PADDLE


def paddle_window_scores(logits, player_x: int, paddle_height: int):
    max_top = logits.shape[1] - paddle_height + 1
    paddle_logits = logits[:, :, player_x, int(PADDLE)]
    return jnp.stack(
        [jnp.sum(paddle_logits[:, top : top + paddle_height], axis=1) for top in range(max_top)],
        axis=1,
    )


def legal_candidate_tops(frames, player_x: int, paddle_height: int):
    max_top = frames.shape[1] - paddle_height
    current_top = jnp.argmax(frames[:, :, player_x] == PADDLE, axis=1)
    return jnp.clip(
        current_top[:, None] + jnp.asarray([-1, 0, 1], dtype=jnp.int32)[None, :],
        0,
        max_top,
    )


def tracking_prior_scores(frames, candidates, paddle_height: int):
    ball_rows = jnp.argmax(jnp.any(frames == BALL, axis=2), axis=1)
    centers = candidates + paddle_height // 2
    return -jnp.abs(centers - ball_rows[:, None]).astype(jnp.float32)


def decode_legal_paddle_proposal(logits, frames, player_x: int, paddle_height: int, tracking_prior_weight: float = 0.0):
    proposal = jnp.argmax(logits, axis=-1).astype(jnp.uint8)
    candidates = legal_candidate_tops(frames, player_x, paddle_height)
    scores = jnp.take_along_axis(paddle_window_scores(logits, player_x, paddle_height), candidates, axis=1)
    scores = scores + tracking_prior_weight * tracking_prior_scores(frames, candidates, paddle_height)
    tops = jnp.take_along_axis(candidates, jnp.argmax(scores, axis=1)[:, None], axis=1)[:, 0]
    rows = tops[:, None] + jnp.arange(paddle_height)[None, :]
    batch = jnp.arange(proposal.shape[0])[:, None]
    proposal = proposal.at[:, :, player_x].set(BACKGROUND)
    return proposal.at[batch, rows, player_x].set(PADDLE)
