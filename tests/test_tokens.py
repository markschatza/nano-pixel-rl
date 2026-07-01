import jax.numpy as jnp

from nano_pixel_rl.env.tokens import BACKGROUND, BALL, PADDLE, PUBLIC_TOKEN_VALUES, to_public


def test_public_token_contract_is_immutable():
    assert int(BACKGROUND) == 0
    assert int(BALL) == 1
    assert int(PADDLE) == 2
    assert PUBLIC_TOKEN_VALUES == {0: 0.0, 1: 0.5, 2: 1.0}


def test_to_public_maps_internal_ids_to_public_values():
    frame = jnp.asarray([[0, 1, 2]], dtype=jnp.uint8)
    assert to_public(frame).tolist() == [[0.0, 0.5, 1.0]]
