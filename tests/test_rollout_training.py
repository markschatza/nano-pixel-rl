import json
from pathlib import Path

from nano_pixel_rl.benchmark.speedrun import run_speedrun
from nano_pixel_rl.reference.config import TrainConfig


def test_smoke_speedrun_writes_artifacts(tmp_path):
    result, json_path, md_path = run_speedrun(
        TrainConfig(seed=0, steps=1, num_envs=4, eval_episodes=4, max_episode_steps=4),
        model_size="tiny",
        out_dir=str(tmp_path),
    )
    assert result["update_time_seconds"] > 0
    assert json_path.exists()
    assert md_path.exists()
    loaded = json.loads(Path(json_path).read_text())
    assert "win_rate_random_legal" in loaded["eval"]
