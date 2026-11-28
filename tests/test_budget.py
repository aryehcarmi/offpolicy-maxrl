import json
import sys
from pathlib import Path

import pytest

from maxrl.budget import main, projection, workload


def test_frozen_workload_and_projection(tmp_path):
    directory = Path(__file__).resolve().parents[1] / "configs"
    plan = workload(directory)
    assert [r["runs"] for r in plan["controlled"]] == [1, 3, 12, 16]
    assert plan["diagnostic"] == {"midpoint": 2048, "discovery": 40960, "confirmation": 40960}
    assert plan["total_responses"] == 1_368_576
    assert workload(directory, True)["controlled"][-1]["runs"] == 13
    run = {"config": {"method": "count"}, "runtime": {"hourly_usd": 2.0}, "hardware": {"gpu": "fixture"}}
    (tmp_path / "run.json").write_text(json.dumps(run))
    costs = ("phase,seconds,responses,generated_tokens\ntraining_generation,3600,4096,8192\n"
             "evaluation,1800,2048,4096\ntraining_update,900,0,0\nsetup_io_other,60,0,0\n")
    (tmp_path / "resources.csv").write_text(costs)
    result = projection(plan, tmp_path, None, None, None)
    assert result["pilot_mean_training_tokens"] == 2
    assert result["projected_core_usd"] is None
    complete = projection(plan, tmp_path, None, 40, 60)
    assert complete["projected_core_usd"] == pytest.approx(complete["controlled_usd"] + 100)
    run["runtime"]["hourly_usd"] = None
    (tmp_path / "run.json").write_text(json.dumps(run))
    unpriced = projection(plan, tmp_path, None, 40, 60)
    assert unpriced["controlled_phase_hours"] == complete["controlled_phase_hours"]
    assert unpriced["controlled_usd"] is None and unpriced["projected_core_usd"] is None


def test_budget_expanded_tuning_grid(monkeypatch, capsys):
    directory = Path(__file__).resolve().parents[1] / "configs"
    monkeypatch.setattr(sys, "argv", ["maxrl.budget", "--configs", str(directory), "--tune-trials", "16"])
    main()
    plan = json.loads(capsys.readouterr().out)
    assert [r["runs"] for r in plan["controlled"]] == [1, 3, 16, 16]
    assert plan["total_responses"] == 1_495_552
