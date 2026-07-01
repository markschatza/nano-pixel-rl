from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_pixel_rl.benchmark.validate_run import validate_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", required=True)
    args = parser.parse_args()
    ok, message = validate_file(args.validate)
    print(message)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
