from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from nano_pixel_rl.env.pixelpong import EnvConfig
from nano_pixel_rl.learner.model import forward, init_params
from nano_pixel_rl.learner.proposal import decode_legal_paddle_proposal
from nano_pixel_rl.learner.update import LearnerState, update_state
from nano_pixel_rl.reference.config import LearnerConfig


class Learner:
    def __init__(self, config: LearnerConfig, env_config: EnvConfig = EnvConfig()):
        self.config = config
        self.env_config = env_config
        self.optimizer = optax.adam(config.learning_rate)
        self._jit_update = jax.jit(lambda state, batch: update_state(state, self.optimizer, batch))
        self._jit_logits = jax.jit(lambda params, frames: forward(params, frames))

    def init(self, key):
        params = init_params(key, self.env_config.height, self.env_config.width, self.config.hidden_dim)
        return LearnerState(params=params, opt_state=self.optimizer.init(params), step=jnp.asarray(0, dtype=jnp.int32))

    def logits(self, state: LearnerState, frames):
        return self._jit_logits(state.params, frames)

    def propose(self, state: LearnerState, frames):
        logits = self.logits(state, frames)
        return decode_legal_paddle_proposal(logits, frames, self.env_config.player_x, self.env_config.paddle_height)

    def update(self, state: LearnerState, batch):
        return self._jit_update(state, batch)
