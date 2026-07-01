from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path


def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def hardware_summary():
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }


def write_artifacts(out_dir, result):
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    json_path = path / "result.json"
    md_path = path / "report.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    md_path.write_text(
        "\n".join(
            [
                "# Nano Pixel RL Run Report",
                "",
                f"- Threshold reached: `{result['threshold_reached']}`",
                f"- Completed run: `{result['completed_run']}`",
                f"- Update time seconds: `{result['update_time_seconds']:.6f}`",
                f"- Random/legal win rate: `{result['eval']['win_rate_random_legal']:.3f}`",
                f"- Delayed tracker win rate: `{result['eval']['win_rate_delayed_tracker']:.3f}`",
                f"- Prediction loss: `{result['train_metrics'].get('prediction_loss', 0.0):.6f}`",
            ]
        )
        + "\n"
    )
    return json_path, md_path
