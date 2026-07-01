from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from nano_pixel_rl.env.tokens import VOCAB_SIZE


class ModelParams(NamedTuple):
    token_embed: jnp.ndarray
    pos_embed: jnp.ndarray
    q: jnp.ndarray
    k: jnp.ndarray
    v: jnp.ndarray
    proj: jnp.ndarray
    mlp_w1: jnp.ndarray
    mlp_b1: jnp.ndarray
    mlp_w2: jnp.ndarray
    mlp_b2: jnp.ndarray
    out_w: jnp.ndarray
    out_b: jnp.ndarray


def init_params(key, height: int, width: int, hidden_dim: int, context_frames: int = 1) -> ModelParams:
    seq = context_frames * height * width
    keys = jax.random.split(key, 10)

    def normal(k, shape, scale=0.02):
        return jax.random.normal(k, shape, dtype=jnp.float32) * scale

    return ModelParams(
        token_embed=normal(keys[0], (VOCAB_SIZE, hidden_dim)),
        pos_embed=normal(keys[1], (seq, hidden_dim)),
        q=normal(keys[2], (hidden_dim, hidden_dim)),
        k=normal(keys[3], (hidden_dim, hidden_dim)),
        v=normal(keys[4], (hidden_dim, hidden_dim)),
        proj=normal(keys[5], (hidden_dim, hidden_dim)),
        mlp_w1=normal(keys[6], (hidden_dim, hidden_dim * 2)),
        mlp_b1=jnp.zeros((hidden_dim * 2,), dtype=jnp.float32),
        mlp_w2=normal(keys[7], (hidden_dim * 2, hidden_dim)),
        mlp_b2=jnp.zeros((hidden_dim,), dtype=jnp.float32),
        out_w=normal(keys[8], (hidden_dim, VOCAB_SIZE)),
        out_b=jnp.zeros((VOCAB_SIZE,), dtype=jnp.float32),
    )


def forward(params: ModelParams, frames):
    if frames.ndim == 3:
        batch, height, width = frames.shape
        frame_seq = height * width
        context = params.pos_embed.shape[0] // frame_seq
        frames = jnp.repeat(frames[:, None, :, :], context, axis=1)
    batch, context, height, width = frames.shape
    frame_seq = height * width
    seq = context * frame_seq
    tokens = frames.reshape(batch, seq).astype(jnp.int32)
    x = params.token_embed[tokens] + params.pos_embed[None, :, :]

    q = x @ params.q
    k = x @ params.k
    v = x @ params.v
    scale = jnp.sqrt(jnp.asarray(q.shape[-1], dtype=jnp.float32))
    attn = jax.nn.softmax((q @ jnp.swapaxes(k, -1, -2)) / scale, axis=-1)
    x = x + (attn @ v) @ params.proj
    x = x + (jax.nn.gelu(x @ params.mlp_w1 + params.mlp_b1) @ params.mlp_w2 + params.mlp_b2)
    logits = x[:, -frame_seq:, :] @ params.out_w + params.out_b
    return logits.reshape(batch, height, width, VOCAB_SIZE)
