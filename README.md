# Nano Pixel RL

Nano Pixel RL is a nanochat-style RL speedrun benchmark where the model plays PixelPong by proposing the entire next frame.
The observation and proposal share one immutable token space: `0` is background, `0.5` is ball, and `1` is paddle.

The v1 benchmark keeps the environment frozen and makes the learner the editable surface.
Contributors should improve how `nano_pixel_rl/learner/` transforms input frame tokens into output frame-token proposals without changing token meanings, physics, reward definitions, evaluator semantics, or speedrun accounting.

## Quick Start

```bash
uv sync --extra dev
bash runs/smoke.sh
```

The canonical speedrun entrypoint is:

```bash
bash runs/speedrun.sh
```

V1 signs of life require both:

- At least `90%` win rate versus the random/legal opponent.
- At least `50%` win rate versus the delayed-tracker opponent.

The leaderboard score counts learner training/update time only.
Setup, evaluation, validation, and report writing are recorded as metadata but are not scored time.

## Editable Surface

Edit learner algorithms in `nano_pixel_rl/learner/`.
The intended entry point is `Learner.update()`, which receives frame-token batches and returns an updated learner plus metrics.

Frozen benchmark surfaces live under `nano_pixel_rl/env/` and `nano_pixel_rl/benchmark/`.
Changing them can be useful for local experiments, but those changes are not leaderboard-valid.

## Hardware

The accelerated path targets native Linux with CUDA for JAX.
WSL2 CUDA can work as a secondary path.
Native Windows is treated as CPU smoke-test only for v1.
