# Nano Pixel RL Leaderboard

The v1 leaderboard ranks valid runs by wall-clock learner update time to signs of life.

## Validity Rules

- Token meanings are immutable: `0` background, `0.5` ball, `1` paddle.
- The learner proposes a full next frame; the interpreter alone converts coherent controlled-paddle edits into movement.
- Ball physics, opponent behavior, rewards, evaluator seeds, and threshold checks are frozen.
- Scored time includes learner training/update work only.
- Run artifacts must include git revision, seeds, hardware summary, update time, win rates, prediction loss, invalid proposal rate, and threshold status.

## V1 Threshold

A run reaches signs of life when fixed-seed evaluation reports:

- `win_rate_random_legal >= 0.90`
- `win_rate_delayed_tracker >= 0.50`

Prediction loss is diagnostic only and cannot qualify a run without point performance.
