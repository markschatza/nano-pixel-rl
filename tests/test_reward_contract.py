import jax

from nano_pixel_rl.env.pixelpong import reset
from nano_pixel_rl.env.rewards import point_reward


def test_point_reward_is_large_sparse_signal():
    state = reset(jax.random.PRNGKey(0))._replace(last_player_point=True)
    assert float(point_reward(state)) == 1.0
    state = state._replace(last_player_point=False, last_opponent_point=True)
    assert float(point_reward(state)) == -1.0
