#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/train.py --steps 2 --num-envs 8 --eval-episodes 8 --out artifacts/smoke
