from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from nano_pixel_rl.env.pixelpong import EnvConfig, EnvState
from nano_pixel_rl.env.tokens import PADDLE


class InterpretedAction(NamedTuple):
    delta: jnp.ndarray
    valid: jnp.ndarray


def interpret_proposal(proposed_frame, state: EnvState, config: EnvConfig = EnvConfig()) -> InterpretedAction:
    column = proposed_frame[:, config.player_x]
    rows = jnp.where(column == PADDLE, size=config.height, fill_value=-1)[0]
    count = jnp.sum(rows >= 0)
    first = rows[0]
    expected = first + jnp.arange(config.paddle_height, dtype=jnp.int32)
    contiguous = jnp.all(rows[: config.paddle_height] == expected)
    right_height = count == config.paddle_height
    proposed_top = first
    current_top = state.player_y
    raw_delta = proposed_top - current_top
    valid = (
        right_height
        & contiguous
        & (proposed_top >= 0)
        & (proposed_top <= config.height - config.paddle_height)
        & (jnp.abs(raw_delta) <= 1)
    )
    delta = jnp.where(valid, raw_delta, 0).astype(jnp.int32)
    return InterpretedAction(delta=delta, valid=valid)
