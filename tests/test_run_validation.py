from nano_pixel_rl.benchmark.validate_run import validate_result


def valid_result():
    return {
        "git_revision": "abc",
        "seed": 0,
        "hardware": {},
        "update_time_seconds": 1.0,
        "completed_run": True,
        "threshold_reached": True,
        "config": {
            "completed_steps": 1,
            "duration_seconds": 1.0,
            "eval_episodes": 1,
            "max_episode_steps": 1,
            "model_size": "tiny",
            "num_envs": 1,
        },
        "eval": {
            "invalid_proposal_rate_delayed_tracker": 0.0,
            "invalid_proposal_rate_random_legal": 0.0,
            "prediction_loss_delayed_tracker": 1.0,
            "prediction_loss_random_legal": 1.0,
            "win_rate_random_legal": 0.9,
            "win_rate_delayed_tracker": 0.5,
        },
        "train_metrics": {"prediction_loss": 1.0, "policy_loss": 0.0},
    }


def test_valid_result_passes():
    ok, message = validate_result(valid_result())
    assert ok, message


def test_prediction_loss_cannot_override_win_rates():
    result = valid_result()
    result["threshold_reached"] = True
    result["eval"]["win_rate_delayed_tracker"] = 0.49
    ok, message = validate_result(result)
    assert not ok
    assert "threshold" in message


def test_duration_must_match_update_time():
    result = valid_result()
    result["update_time_seconds"] = 0.5
    ok, message = validate_result(result)
    assert not ok
    assert "duration" in message
