from pathlib import Path


def test_docs_name_frozen_and_editable_surfaces():
    readme = Path("README.md").read_text()
    contract = Path("docs/benchmark-contract.md").read_text()
    learner = Path("docs/learner-guide.md").read_text()
    assert "runs/speedrun.sh" in readme
    assert "`0.5` is ball" in readme
    assert "Proposal Interpretation" in contract
    assert "nano_pixel_rl/learner/" in learner
