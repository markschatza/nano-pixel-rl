import jax
import jax.numpy as jnp

from nano_pixel_rl.env.opponent import DELAYED_TRACKER, NEAR_PERFECT, RANDOM_LEGAL, opponent_delta
from nano_pixel_rl.env.pixelpong import EnvConfig, reset


def test_opponents_emit_legal_deltas():
    config = EnvConfig()
    state = reset(jax.random.PRNGKey(0), config)
    for kind in [RANDOM_LEGAL, DELAYED_TRACKER, NEAR_PERFECT]:
        delta = opponent_delta(state, jax.random.PRNGKey(1), kind, config)
        assert int(delta) in [-1, 0, 1]


def test_near_perfect_tracks_ball():
    config = EnvConfig()
    state = reset(jax.random.PRNGKey(0), config)._replace(ball_y=jnp.asarray(config.height - 1))
    assert int(opponent_delta(state, jax.random.PRNGKey(1), NEAR_PERFECT, config)) == 1
