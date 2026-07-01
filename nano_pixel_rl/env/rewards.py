from __future__ import annotations

import jax.numpy as jnp

from nano_pixel_rl.env.pixelpong import EnvState


def point_reward(state: EnvState):
    return jnp.asarray(state.last_player_point, dtype=jnp.float32) - jnp.asarray(
        state.last_opponent_point, dtype=jnp.float32
    )
