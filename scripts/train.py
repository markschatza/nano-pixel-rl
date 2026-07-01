from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--out", default="artifacts/runs/latest")
    args = parser.parse_args()
    raise SystemExit(
        "Training loop is not implemented yet. "
        f"Received steps={args.steps}, num_envs={args.num_envs}, eval_episodes={args.eval_episodes}, out={args.out}."
    )


if __name__ == "__main__":
    main()
