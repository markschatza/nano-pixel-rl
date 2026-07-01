from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Evaluation is currently run as part of scripts/train.py.")
    parser.parse_args()
    raise SystemExit("Standalone checkpoint evaluation will be added after checkpoint persistence lands.")


if __name__ == "__main__":
    main()
