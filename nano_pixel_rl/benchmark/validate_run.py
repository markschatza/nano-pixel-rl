from __future__ import annotations

import json
from pathlib import Path


REQUIRED_KEYS = {
    "config",
    "eval",
    "git_revision",
    "hardware",
    "completed_run",
    "seed",
    "threshold_reached",
    "update_time_seconds",
    "train_metrics",
}

REQUIRED_CONFIG_KEYS = {
    "completed_steps",
    "duration_seconds",
    "eval_episodes",
    "max_episode_steps",
    "model_size",
    "num_envs",
}

REQUIRED_EVAL_KEYS = {
    "invalid_proposal_rate_delayed_tracker",
    "invalid_proposal_rate_random_legal",
    "prediction_loss_delayed_tracker",
    "prediction_loss_random_legal",
    "win_rate_delayed_tracker",
    "win_rate_random_legal",
}


def validate_result(result):
    missing = sorted(REQUIRED_KEYS - set(result))
    if missing:
        return False, f"Missing required keys: {missing}"
    missing_config = sorted(REQUIRED_CONFIG_KEYS - set(result["config"]))
    if missing_config:
        return False, f"Missing config keys: {missing_config}"
    missing_eval = sorted(REQUIRED_EVAL_KEYS - set(result["eval"]))
    if missing_eval:
        return False, f"Missing eval keys: {missing_eval}"
    duration = result["config"].get("duration_seconds")
    if duration is not None and result["update_time_seconds"] < duration:
        return False, "update_time_seconds is below requested duration"
    eval_result = result["eval"]
    threshold = eval_result.get("win_rate_random_legal", 0.0) >= 0.90 and eval_result.get(
        "win_rate_delayed_tracker", 0.0
    ) >= 0.50
    if result["threshold_reached"] != threshold:
        return False, "threshold_reached does not match win-rate gates"
    if not result["completed_run"]:
        return False, "run did not complete its requested training budget"
    return True, "ok"


def validate_file(path):
    result = json.loads(Path(path).read_text())
    return validate_result(result)
