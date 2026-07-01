from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from nano_pixel_rl.env.tokens import BACKGROUND, BALL, PADDLE


@dataclass(frozen=True)
class EnvConfig:
    height: int = 16
    width: int = 16
    paddle_height: int = 3
    player_x: int = 1
    opponent_x: int = 14
    max_steps: int = 256


class EnvState(NamedTuple):
    ball_y: jnp.ndarray
    ball_x: jnp.ndarray
    vel_y: jnp.ndarray
    vel_x: jnp.ndarray
    player_y: jnp.ndarray
    opponent_y: jnp.ndarray
    player_score: jnp.ndarray
    opponent_score: jnp.ndarray
    steps: jnp.ndarray
    last_player_point: jnp.ndarray
    last_opponent_point: jnp.ndarray


def _center_start(config: EnvConfig):
    return jnp.asarray((config.height - config.paddle_height) // 2, dtype=jnp.int32)


def reset(key, config: EnvConfig = EnvConfig()) -> EnvState:
    del key
    center = _center_start(config)
    return EnvState(
        ball_y=jnp.asarray(config.height // 2, dtype=jnp.int32),
        ball_x=jnp.asarray(config.width // 2, dtype=jnp.int32),
        vel_y=jnp.asarray(1, dtype=jnp.int32),
        vel_x=jnp.asarray(1, dtype=jnp.int32),
        player_y=center,
        opponent_y=center,
        player_score=jnp.asarray(0, dtype=jnp.int32),
        opponent_score=jnp.asarray(0, dtype=jnp.int32),
        steps=jnp.asarray(0, dtype=jnp.int32),
        last_player_point=jnp.asarray(False),
        last_opponent_point=jnp.asarray(False),
    )


def render_frame(state: EnvState, config: EnvConfig = EnvConfig()):
    frame = jnp.full((config.height, config.width), BACKGROUND, dtype=jnp.uint8)
    paddle_offsets = jnp.arange(config.paddle_height, dtype=jnp.int32)
    player_rows = state.player_y + paddle_offsets
    opponent_rows = state.opponent_y + paddle_offsets
    frame = frame.at[player_rows, config.player_x].set(PADDLE)
    frame = frame.at[opponent_rows, config.opponent_x].set(PADDLE)
    frame = frame.at[state.ball_y, state.ball_x].set(BALL)
    return frame


def _clip_paddle(y, config: EnvConfig):
    return jnp.clip(y, 0, config.height - config.paddle_height).astype(jnp.int32)


def _reset_after_point(state: EnvState, vx: jnp.ndarray, config: EnvConfig):
    return state._replace(
        ball_y=jnp.asarray(config.height // 2, dtype=jnp.int32),
        ball_x=jnp.asarray(config.width // 2, dtype=jnp.int32),
        vel_y=jnp.asarray(1, dtype=jnp.int32),
        vel_x=vx.astype(jnp.int32),
        player_y=_center_start(config),
        opponent_y=_center_start(config),
    )


def step(state: EnvState, player_delta, opponent_delta, config: EnvConfig = EnvConfig()) -> EnvState:
    player_y = _clip_paddle(state.player_y + jnp.asarray(player_delta, dtype=jnp.int32), config)
    opponent_y = _clip_paddle(state.opponent_y + jnp.asarray(opponent_delta, dtype=jnp.int32), config)

    next_y = state.ball_y + state.vel_y
    hit_top_bottom = (next_y <= 0) | (next_y >= config.height - 1)
    vel_y = jnp.where(hit_top_bottom, -state.vel_y, state.vel_y).astype(jnp.int32)
    next_y = jnp.clip(state.ball_y + vel_y, 0, config.height - 1).astype(jnp.int32)

    next_x = (state.ball_x + state.vel_x).astype(jnp.int32)
    player_hit = (
        (next_x == config.player_x)
        & (next_y >= player_y)
        & (next_y < player_y + config.paddle_height)
        & (state.vel_x < 0)
    )
    opponent_hit = (
        (next_x == config.opponent_x)
        & (next_y >= opponent_y)
        & (next_y < opponent_y + config.paddle_height)
        & (state.vel_x > 0)
    )
    vel_x = jnp.where(player_hit | opponent_hit, -state.vel_x, state.vel_x).astype(jnp.int32)
    next_x = jnp.where(player_hit | opponent_hit, state.ball_x + vel_x, next_x).astype(jnp.int32)

    player_point = next_x >= config.width
    opponent_point = next_x < 0

    updated = state._replace(
        ball_y=next_y,
        ball_x=next_x,
        vel_y=vel_y,
        vel_x=vel_x,
        player_y=player_y,
        opponent_y=opponent_y,
        player_score=state.player_score + player_point.astype(jnp.int32),
        opponent_score=state.opponent_score + opponent_point.astype(jnp.int32),
        steps=state.steps + 1,
        last_player_point=player_point,
        last_opponent_point=opponent_point,
    )
    updated = jax.lax.cond(player_point, lambda s: _reset_after_point(s, jnp.asarray(-1), config), lambda s: s, updated)
    updated = jax.lax.cond(opponent_point, lambda s: _reset_after_point(s, jnp.asarray(1), config), lambda s: s, updated)
    return updated
