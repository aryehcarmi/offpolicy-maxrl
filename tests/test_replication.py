"""The published comparison uses three matched training seeds and fixed evaluation draws."""
import json
from pathlib import Path
import runpy

import pytest


def test_published_summary_uses_three_paired_seeds(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    wrapper = runpy.run_path(str(repo / "scripts/li_replication.py"))
    spec = json.loads((repo / "configs/replication/li_frozen.json").read_text())
    for seed in spec["training_seeds"]:
        for method, value in (("is", 0.1), ("is_sc", 0.2)):
            path = tmp_path / wrapper["run_name"](method, seed)
            path.mkdir()
            record = {"training_seed": seed, "method": method, "upstream_commit": spec["upstream_commit"],
                      "mean_accuracy": value + seed / 16}
            (path / "evaluation.json").write_text(json.dumps(record))
    summary = wrapper["summarize"](tmp_path, spec)
    assert summary["is"]["mean"] == pytest.approx(0.1625)
    assert summary["is"]["published_mean"] == 0.209
    assert summary["is_sc_minus_is"]["seed_t_interval_95"] == pytest.approx([0.1, 0.1])
    path = tmp_path / "li_is_sc_seed2/evaluation.json"
    path.write_text(path.read_text().replace('"training_seed": 2', '"training_seed": 1'))
    with pytest.raises(ValueError, match="provenance"):
        wrapper["summarize"](tmp_path, spec)
