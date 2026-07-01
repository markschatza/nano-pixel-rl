import jax.numpy as jnp

from nano_pixel_rl.benchmark.metrics import cross_entropy, token_accuracy


def test_prediction_metrics_read_frame_logits():
    logits = jnp.asarray([[[0.0, 5.0, 0.0], [5.0, 0.0, 0.0]]])
    target = jnp.asarray([[1, 0]], dtype=jnp.uint8)
    assert float(token_accuracy(logits, target)) == 1.0
    assert float(cross_entropy(logits, target)) < 0.02
