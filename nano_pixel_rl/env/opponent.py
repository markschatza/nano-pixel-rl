from __future__ import annotations

import jax
import jax.numpy as jnp

from nano_pixel_rl.env.pixelpong import EnvConfig, EnvState

RANDOM_LEGAL = 0
DELAYED_TRACKER = 1
NEAR_PERFECT = 2


def _track_delta(paddle_y, ball_y, config: EnvConfig):
    center = paddle_y + config.paddle_height // 2
    return jnp.sign(ball_y - center).astype(jnp.int32)


def opponent_delta(state: EnvState, key, opponent_kind: int, config: EnvConfig = EnvConfig()):
    random_delta = jax.random.randint(key, (), -1, 2, dtype=jnp.int32)
    delayed_delta = jnp.where(state.ball_x >= config.width // 2, _track_delta(state.opponent_y, state.ball_y, config), 0)
    perfect_delta = _track_delta(state.opponent_y, state.ball_y, config)
    return jnp.select(
        [opponent_kind == RANDOM_LEGAL, opponent_kind == DELAYED_TRACKER, opponent_kind == NEAR_PERFECT],
        [random_delta, delayed_delta, perfect_delta],
        default=delayed_delta,
    ).astype(jnp.int32)
