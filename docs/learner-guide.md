# Learner Guide

The normal editable surface is `nano_pixel_rl/learner/`.

You may change:

- Model shape and size.
- Optimizer and update rules.
- Loss weighting.
- Learner memory/state.
- Proposal sampling or decoding inside the learner contract.

Do not change these for leaderboard-valid runs:

- Token meanings.
- PixelPong physics.
- Proposal legality rules.
- Opponent ladder definitions.
- Reward definitions.
- Evaluation threshold.
- Scored-time semantics.

The baseline learner is intentionally small and direct so algorithm changes are easy to inspect.
