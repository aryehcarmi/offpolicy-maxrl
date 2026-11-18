"""Regression tests for the frozen-batch trainer and its reproduction check."""

import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from maxrl.config import Runtime, load_config
from maxrl.data import digest
from maxrl.evaluate import record_groups
from maxrl.metrics import dump_json
from maxrl.model import checkpoint_id, load_policy
from maxrl.prompts import prompt_metadata
from maxrl import saved_trajectory as runner
from maxrl.schema import Response
from maxrl.train import update_policy


def test_success_weights_ignore_failures_and_measure_kish_ess():
    groups = [[type("Row", (), {"reward": r}) for r in (1, 0, 1)]]
    stats = runner.success_weight_stats(groups, np.log([1., 100., 3.]))
    assert stats["count"] == 2
    assert stats["mean"] == pytest.approx(2.)
    assert stats["ess"] == pytest.approx(16/10)
    assert stats["quantiles"]["p25"] == pytest.approx(1.5)


def test_compare_detects_relative_errors_at_tiny_gradient_norms(tmp_path):
    path = tmp_path / "weights.csv"
    path.write_text("sample_id,puzzle_id,reward,log_weight\na,p,1,-30\n")
    groups = [[Response("p", [1], [2], [-1.], [1], [0], 1, "base", "a", "batch")]]
    assert runner.compare_update(path, groups, [-30], 1e-15, 1e-15)["passed"]
    assert not runner.compare_update(path, groups, [-30], 2e-15, 1e-15)["passed"]
    assert not runner.compare_update(path, groups, [-30.5], 1e-15, 1e-15)["passed"]
    path.write_text(path.read_text().replace("a,p,1", "a,p,0"))
    with pytest.raises(ValueError, match="Puzzle or reward"):
        runner.compare_update(path, groups, [-30], 1e-15, 1e-15)


@pytest.fixture
def frozen_pilot(tiny, tmp_path):
    pilot = tmp_path / "original"
    (pilot / "responses").mkdir(parents=True)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    initial_eval = tmp_path / "initial-eval"
    initial_eval.mkdir()
    (initial_eval / "responses.jsonl").write_text("test probe fixture")
    (pilot / "responses/validation-000000.jsonl").write_text("test probe fixture")
    base_id = checkpoint_id(tiny)
    base_config = load_config(Path(__file__).resolve().parents[1] / "configs/pilot.json")
    config = replace(base_config, model=tiny, revision=None, prompts_per_step=2, responses_per_prompt=2)
    dump_json(pilot / "run.json", dict(run_id="source", config=asdict(config), initial_checkpoint_id=base_id,
                                       runtime=asdict(Runtime(device="cpu")), prompt=prompt_metadata(),
                                       manifest_sha256=digest(manifest)))
    model, _ = load_policy(tiny, device="cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    events = []
    for step in range(1,17):
        pool = f"source/train/{step}"
        groups = [[Response(str(p), [3,4], [3+(p+i+step)%4], [-2.], [1], [0], int(i == 0),
                            base_id, f"{pool}/{p}/{i}", pool) for i in range(2)] for p in range(2)]
        record_groups(pilot / f"responses/train-{step:06d}.jsonl", groups)
        stats, logs = update_policy(model, optimizer, groups, "count", 1)
        # Deliberately reverse the official CSV's row order: IDs bind every comparison.
        with (pilot / f"responses/weights-{step:06d}.csv").open("w") as f:
            w = csv.writer(f)
            w.writerow(["sample_id", "puzzle_id", "reward", "log_weight"])
            for row, value in reversed(list(zip([r for g in groups for r in g], logs))):
                w.writerow([row.sample_id, row.puzzle_id, row.reward, value])
        events.append(dict(split="train", step=step, gradient_norm=stats["gradient_norm"], sampler_id=base_id,
                           target_id=base_id if step == 1 else f"source/step-{step-1:06d}"))
    (pilot / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
    return pilot, initial_eval, manifest


def test_verify_replays_all_steps_and_rejects_a_late_mismatch(frozen_pilot, tmp_path, monkeypatch):
    pilot, _, _ = frozen_pilot
    def invoke(out):
        monkeypatch.setattr(sys, "argv", ["saved_trajectory", "verify", "--pilot", str(pilot),
                                          "--out", str(out), "--device", "cpu"])
        runner.main()

    invoke(tmp_path / "pass")
    record = json.loads((tmp_path / "pass/verification.json").read_text())
    assert record["passed"] and len(record["steps"]) == 16
    assert max(row["abs_log_weight_error_max"] for row in record["steps"]) == 0.
    path = pilot / "responses/weights-000016.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["log_weight"] = str(float(rows[0]["log_weight"]) + 1.)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError, match="tolerance"):
        invoke(tmp_path / "fail")
    failed = json.loads((tmp_path / "fail/verification.json").read_text())
    assert not failed["passed"]
    assert [r["step"] for r in failed["steps"] if not r["passed"]] == [16]


def test_trajectories_start_fresh_save_selected_checkpoint_and_log_post_update_probes(
        frozen_pilot, tmp_path, monkeypatch):
    pilot, initial_eval, manifest = frozen_pilot
    from maxrl import trajectory_probes as probes
    # The tiny model's vocabulary omits "<". Use a vocabulary token for the opening probe.
    monkeypatch.setattr(probes, "opening_id", lambda tok: 3)
    monkeypatch.setattr(probes, "load_validation", lambda *a: ({}, [[3,4]]))
    monkeypatch.setattr(probes, "validation_estimates", lambda *a: {"fixture": True})
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["saved_trajectory", "run", "--pilot", str(pilot), "--out", str(out),
                                      "--device", "cpu", "--initial-eval", str(initial_eval), "--manifest", str(manifest)])
    runner.main()
    rows = [json.loads(line) for line in (out / "trajectories.jsonl").read_text().splitlines()]
    assert len(rows) == 96
    for method in ("count","snis","uncorrected"):
        for lr in (1.5e-7,3e-7):
            path = [r for r in rows if (r["method"],r["lr"]) == (method,lr)]
            assert [r["step"] for r in path] == list(range(1,17))
            assert [r["step"] for r in path if "validation_is_post_update" in r] == [4,8,16]
            assert all(0 < r["validation_open_probability_post_update"] < 1 for r in path)
            assert all(r["success_weights_pre_update"]["count"] == 2 for r in path)
            pair = [r for r in rows if r["method"] == method and r["step"] == 1]
            assert pair[0]["gradient_norm"] == pair[1]["gradient_norm"]
    selected = json.loads((out / "pilot/run.json").read_text())
    assert (selected["config"]["method"], selected["config"]["lr"]) == ("count",1.5e-7)
    assert selected["config"]["checkpoint_steps"] == [16]
    assert checkpoint_id(out / "pilot/checkpoints/step-000016") == selected["run_id"] + "/step-000016"
    assert selected["source_run_id"] == "source"
