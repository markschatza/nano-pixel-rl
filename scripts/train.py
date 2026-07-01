from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_pixel_rl.benchmark.speedrun import run_speedrun
from nano_pixel_rl.reference.config import TrainConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--max-episode-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-size", default="tiny")
    parser.add_argument("--out", default="artifacts/runs/latest")
    args = parser.parse_args()
    result, json_path, md_path = run_speedrun(
        TrainConfig(
            seed=args.seed,
            steps=args.steps,
            num_envs=args.num_envs,
            eval_episodes=args.eval_episodes,
            max_episode_steps=args.max_episode_steps,
        ),
        model_size=args.model_size,
        out_dir=args.out,
    )
    print(f"threshold_reached={result['threshold_reached']}")
    print(f"result_json={json_path}")
    print(f"report_md={md_path}")


if __name__ == "__main__":
    main()
