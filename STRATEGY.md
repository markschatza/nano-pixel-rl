---
name: Nano Pixel RL
last_updated: 2026-06-30
---

# Nano Pixel RL Strategy

## Target problem

Small RL benchmarks are often either too toy-like to reveal useful learning improvements or too entangled with environment engineering to make learner changes easy to compare. Contributors need a fast, editable benchmark where predictable dynamics, controllable dynamics, and wall-clock learning speed can be measured without touching a complex backend.

## Our approach

Nano Pixel RL copies the nanoGPT/nanochat speedrun philosophy: keep the environment prebaked and stable, expose one user-facing `learner.update()` surface, and make time-to-threshold the organizing score. The scaling bet is that next-pixel prediction and control should share the same small token space, so progress on PixelPong feels like progress on a tiny version of the sequence-modeling problem rather than a one-off RL control trick.

PixelPong makes the input and output the same simple image vocabulary, so improvements have to separate predictable world dynamics from controllable action rather than optimize a hand-written UP/DOWN policy head. The dense next-frame prediction loss is intentionally not just auxiliary bookkeeping: it is the pressure that makes the learner model what can be predicted, discover what can be influenced, and expose small architecture or update-rule improvements that could plausibly scale beyond this game.

## Who it's for

**Primary:** RL benchmark contributors - They're hiring Nano Pixel RL to test small learner ideas quickly against a clear speedrun target and compare whether those ideas improve time-to-threshold.

**Secondary:** ML tinkerers - They're hiring Nano Pixel RL to play with one editable learner file and see concrete learning-speed changes without rebuilding the environment.

## Key metrics

- **Time-to-threshold** - Wall-clock time required to reach the reference PixelPong capability threshold; measured by the benchmark runner and leaderboard submission logs.
- **Point-win performance** - Win rate or average point differential against the baked opponent after training; measured in evaluation rollouts.
- **Next-pixel prediction loss** - Dense next-frame prediction error over legal environment transitions in the same token space used for action proposals; measured during training as the small auxiliary objective.
- **Invalid proposal rate** - Share of proposed next frames rejected for impossible edits or incoherent paddle movement; measured by the environment interpreter.
- **Reference reproducibility** - Variance of the reference run across seeds and machines; measured by repeated `speedrun` runs before leaderboard updates.

## Tracks

### Speedrun Benchmark

Define the canonical training command, threshold, logs, and leaderboard format.

_Why it serves the approach:_ A public time-to-threshold target turns small learner changes into comparable progress instead of isolated experiments.

### Editable Learner Surface

Keep all participant-facing model and update logic concentrated in a small `learner.update()` path.

_Why it serves the approach:_ Contributors can focus on learning improvements while the environment, scoring, and data plumbing stay stable.

### PixelPong Dynamics Contract

Lock down the image-token environment, proposal interpreter, legality checks, rewards, and evaluation protocol around a shared input/output vocabulary.

_Why it serves the approach:_ A stable shared-token dynamics contract forces learners to distinguish predictable ball physics from controllable paddle influence while keeping the benchmark aligned with nanochat-style sequence modeling.

### Multi-Game Tokenization Path

Preserve a future path toward a learned tokenizer and a single model trained across a suite of small visual games.

_Why it serves the approach:_ Nanochat trains its tokenizer as part of the system, but PixelPong v1 freezes the token space to isolate learner improvements; a learned tokenizer only becomes strategically useful once it has to serve multiple environments instead of overfitting one game.

### Reference Run Quality

Maintain a simple baseline run that starts around one hour to threshold and can be optimized down over time.

_Why it serves the approach:_ The benchmark needs a credible baseline that is slow enough to leave room for progress but fast enough for repeated local iteration.

## Not working on

- A broad RL environment suite; PixelPong is the first tight benchmark.
- A learned tokenizer for v1; tokenization becomes interesting when it is shared across a suite of games.
- A new backend research platform; the backend should be fast, prebaked, and mostly untouched.
- A general game-playing API with discrete action heads; the defining constraint is same-token-space frame proposal.

## Marketing

**One-liner:** A nanoGPT-style RL speedrun benchmark where the model plays PixelPong by proposing the next frame.

**Key message:** Fast learner iteration, stable environment rules, and a leaderboard that rewards real reductions in time-to-threshold. The core bet is that next-pixel prediction in a shared action/observation token space can become a scalable nanochat-like playground for RL ideas.
