# Benchmark Contract

PixelPong v1 is a fixed `16x16` image-grid game.
The learner-controlled paddle is on the left at `x=1`; the opponent paddle is on the right at `x=14`.
Paddles are vertical bars of height `3`, and the ball is one cell.

## Immutable Tokens

| Public value | Internal ID | Meaning |
|---|---:|---|
| `0` | `0` | Background |
| `0.5` | `1` | Ball |
| `1` | `2` | Paddle |

Leaderboard-valid runs must keep these meanings fixed.

## Proposal Interpretation

The learner proposes a full next frame in the same token space as the observation.
The environment reads only the controlled paddle column from that proposal.
If the proposal contains one contiguous paddle of the correct height and its center moves at most one cell from the current paddle, the movement is accepted.
Otherwise the proposal is rejected as a no-op.

Ball pixels and opponent pixels in the proposal affect dense prediction loss but never directly edit environment state.

## Rewards

Point wins dominate the benchmark reward.
Dense next-frame prediction loss is a smaller learning signal and a required diagnostic because the project is testing shared-token next-pixel prediction plus control.
