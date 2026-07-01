import jax
import jax.numpy as jnp

from nano_pixel_rl.env.pixelpong import EnvConfig, reset, render_frame, step
from nano_pixel_rl.env.tokens import BALL, PADDLE


def test_render_frame_contains_ball_and_two_paddles():
    config = EnvConfig()
    state = reset(jax.random.PRNGKey(0), config)
    frame = render_frame(state, config)
    assert int(jnp.sum(frame == BALL)) == 1
    assert int(jnp.sum(frame == PADDLE)) == config.paddle_height * 2


def test_wall_bounce_inverts_vertical_velocity():
    config = EnvConfig()
    state = reset(jax.random.PRNGKey(0), config)._replace(ball_y=jnp.asarray(0), vel_y=jnp.asarray(-1))
    next_state = step(state, 0, 0, config)
    assert int(next_state.vel_y) == 1
    assert int(next_state.ball_y) == 1


def test_player_miss_awards_opponent_point_and_resets_ball():
    config = EnvConfig()
    state = reset(jax.random.PRNGKey(0), config)._replace(ball_x=jnp.asarray(0), vel_x=jnp.asarray(-1))
    next_state = step(state, 0, 0, config)
    assert int(next_state.opponent_score) == 1
    assert int(next_state.ball_x) == config.width // 2


def test_step_vmaps_over_states():
    config = EnvConfig()
    states = jax.vmap(lambda k: reset(k, config))(jax.random.split(jax.random.PRNGKey(0), 4))
    next_states = jax.vmap(lambda s: step(s, 0, 0, config))(states)
    assert next_states.ball_x.shape == (4,)
