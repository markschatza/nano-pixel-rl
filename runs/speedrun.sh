#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  uv run --extra cuda python scripts/train.py "$@"
else
  .venv/bin/python scripts/train.py "$@"
fi
