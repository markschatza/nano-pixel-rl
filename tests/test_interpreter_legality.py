import jax
import jax.numpy as jnp

from nano_pixel_rl.env.interpreter import interpret_proposal
from nano_pixel_rl.env.pixelpong import EnvConfig, reset, render_frame
from nano_pixel_rl.env.tokens import BACKGROUND, BALL, PADDLE


def test_legal_paddle_move_is_interpreted_as_delta():
    config = EnvConfig()
    state = reset(jax.random.PRNGKey(0), config)
    proposal = jnp.full((config.height, config.width), BACKGROUND, dtype=jnp.uint8)
    top = state.player_y + 1
    rows = top + jnp.arange(config.paddle_height)
    proposal = proposal.at[rows, config.player_x].set(PADDLE)
    action = interpret_proposal(proposal, state, config)
    assert bool(action.valid)
    assert int(action.delta) == 1


def test_teleported_paddle_is_rejected_as_noop():
    config = EnvConfig()
    state = reset(jax.random.PRNGKey(0), config)
    proposal = jnp.full((config.height, config.width), BACKGROUND, dtype=jnp.uint8)
    proposal = proposal.at[jnp.arange(config.paddle_height), config.player_x].set(PADDLE)
    action = interpret_proposal(proposal, state, config)
    assert not bool(action.valid)
    assert int(action.delta) == 0


def test_ball_edits_do_not_affect_interpreted_action():
    config = EnvConfig()
    state = reset(jax.random.PRNGKey(0), config)
    proposal = render_frame(state, config).at[0, 0].set(BALL)
    action = interpret_proposal(proposal, state, config)
    assert bool(action.valid)
    assert int(action.delta) == 0
