from __future__ import annotations

import jax.numpy as jnp

BACKGROUND = jnp.uint8(0)
BALL = jnp.uint8(1)
PADDLE = jnp.uint8(2)
VOCAB_SIZE = 3

PUBLIC_TOKEN_VALUES = {
    0: 0.0,
    1: 0.5,
    2: 1.0,
}


def to_public(frame):
    values = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float32)
    return values[frame.astype(jnp.int32)]
