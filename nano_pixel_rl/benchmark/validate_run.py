from __future__ import annotations

import json
from pathlib import Path


REQUIRED_KEYS = {
    "git_revision",
    "seed",
    "hardware",
    "update_time_seconds",
    "threshold_reached",
    "eval",
    "train_metrics",
}


def validate_result(result):
    missing = sorted(REQUIRED_KEYS - set(result))
    if missing:
        return False, f"Missing required keys: {missing}"
    eval_result = result["eval"]
    threshold = eval_result.get("win_rate_random_legal", 0.0) >= 0.90 and eval_result.get(
        "win_rate_delayed_tracker", 0.0
    ) >= 0.50
    if result["threshold_reached"] != threshold:
        return False, "threshold_reached does not match win-rate gates"
    return True, "ok"


def validate_file(path):
    result = json.loads(Path(path).read_text())
    return validate_result(result)
