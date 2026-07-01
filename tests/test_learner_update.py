import jax

from nano_pixel_rl.benchmark.rollout import make_random_batch
from nano_pixel_rl.env.pixelpong import EnvConfig
from nano_pixel_rl.learner import Learner
from nano_pixel_rl.reference.config import LearnerConfig


def test_learner_forward_and_update_shapes():
    env_config = EnvConfig()
    learner = Learner(LearnerConfig(hidden_dim=16), env_config)
    state = learner.init(jax.random.PRNGKey(0))
    batch = make_random_batch(jax.random.PRNGKey(1), 4, env_config)
    logits = learner.logits(state, batch["obs"])
    assert logits.shape == (4, env_config.height, env_config.width, 3)
    next_state, metrics = learner.update(state, batch)
    assert int(next_state.step) == 1
    assert float(metrics["loss"]) > 0.0
