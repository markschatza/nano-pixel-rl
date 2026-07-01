from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from nano_pixel_rl.env.tokens import BACKGROUND, PADDLE
from nano_pixel_rl.env.pixelpong import EnvConfig
from nano_pixel_rl.learner.model import forward, init_params
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
        proposal = jnp.argmax(logits, axis=-1).astype(jnp.uint8)
        paddle_logits = logits[:, :, self.env_config.player_x, int(PADDLE)]
        all_windows = jnp.stack(
            [
                jnp.sum(paddle_logits[:, top : top + self.env_config.paddle_height], axis=1)
                for top in range(self.env_config.height - self.env_config.paddle_height + 1)
            ],
            axis=1,
        )
        player_column = frames[:, :, self.env_config.player_x]
        current_rows = jnp.argmax(player_column == PADDLE, axis=1)
        candidates = jnp.clip(
            current_rows[:, None] + jnp.asarray([-1, 0, 1], dtype=jnp.int32)[None, :],
            0,
            self.env_config.height - self.env_config.paddle_height,
        )
        scores = jnp.take_along_axis(all_windows, candidates, axis=1)
        tops = jnp.take_along_axis(candidates, jnp.argmax(scores, axis=1)[:, None], axis=1)[:, 0]
        rows = tops[:, None] + jnp.arange(self.env_config.paddle_height)[None, :]
        batch = jnp.arange(proposal.shape[0])[:, None]
        proposal = proposal.at[:, :, self.env_config.player_x].set(BACKGROUND)
        proposal = proposal.at[batch, rows, self.env_config.player_x].set(PADDLE)
        return proposal

    def update(self, state: LearnerState, batch):
        return self._jit_update(state, batch)
