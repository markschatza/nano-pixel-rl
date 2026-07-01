#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  uv run --extra dev python scripts/train.py --steps 2 --num-envs 8 --eval-episodes 8 --out artifacts/smoke
else
  .venv/bin/python scripts/train.py --steps 2 --num-envs 8 --eval-episodes 8 --out artifacts/smoke
fi
