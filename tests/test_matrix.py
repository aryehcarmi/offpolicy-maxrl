from argparse import Namespace
from dataclasses import asdict
import json
from pathlib import Path
import shlex

import pytest

from maxrl.config import Runtime, load_config
from maxrl.data import digest
from maxrl.prompts import prompt_metadata
from maxrl.matrix import METHODS, SEEDS, STEPS, load_freeze, read_trial, run, trials


@pytest.fixture
def args(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n")
    return Namespace(stage="screen", root=tmp_path / "trials with spaces", manifest=manifest,
                     configs=Path(__file__).parents[1] / "configs", device="cpu", dtype="float32",
                     generation_batch=8, score_batch=1, hourly_usd=None)


def populate(args, stage, eta0=None):
    for method, lr, child in trials(stage, eta0):
        config = load_config(args.configs / f"{stage}.json", method=method, lr=lr)
        folder = args.root / child
        folder.mkdir(parents=True)
        (folder / "run.json").write_text(json.dumps({"config": asdict(config),
                                                    "runtime": asdict(Runtime.from_args(args)),
                                                    "prompt": prompt_metadata(),
                                                    "manifest_sha256": digest(args.manifest)}))
        scores = {16: .25, 64: .25, 80: .25, 96: .25}
        if stage == "tune" and lr == eta0:
            scores.update({64: .625, 80: .375, 96: 0.})
        events = [{"split": "validation", "step": step, "pass@1": scores[step],
                   "prompts": 128, "responses": 1024} for step in STEPS[stage]]
        (folder / "metrics.jsonl").write_text("\n".join(map(json.dumps, events)) + "\n")


def test_screen_tune_freeze_and_main_matrix(args, capsys):
    run(args)
    assert len(capsys.readouterr().out.splitlines()) == 3
    populate(args, "screen")
    args.stage = "tune"
    run(args)
    commands = [shlex.split(line) for line in capsys.readouterr().out.splitlines()]
    assert len(commands) == 12
    assert {float(c[c.index("--lr")+1]) for c in commands} == {1e-4/3, 1e-4, 3*1e-4}
    populate(args, "tune", 1e-4)
    # A genuine numerical failure remains in the grid, even with prior metrics.
    (args.root / "tune/count-1/failure.json").write_text('{"category":"numerical"}')
    args.stage = "freeze"
    run(args)
    frozen = json.loads((args.root / "frozen.json").read_text())
    assert frozen["version"] == 3
    assert frozen["prompt"] == prompt_metadata()
    assert frozen["runtime"] == {"dtype": "float32", "generation_batch": 8, "score_batch": 1}
    assert len(frozen["screen"]) == 3 and len(frozen["tune"]) == 12
    assert frozen["rates"] == {m: (1e-4/3 if m == "count" else 1e-4) for m in METHODS}
    assert any(row["status"] == "numerical_failure" for row in frozen["tune"])
    assert 17 not in {seed for seeds in SEEDS.values() for seed in seeds}
    capsys.readouterr()
    for stage in ("main", "test"):
        args.stage = stage
        run(args)
        commands = [shlex.split(line) for line in capsys.readouterr().out.splitlines()]
        pairs = {(c[c.index("--method")+1], int(c[c.index("--seed")+1])) for c in commands}
        assert len(commands) == 16
        assert pairs == {(m, seed) for m, seeds in SEEDS.items() for seed in seeds}
        assert all(("--panel" in c) == (stage == "test") for c in commands)


def test_pending_or_operational_failure_cannot_freeze(args):
    args.stage = "freeze"
    populate(args, "screen")
    with pytest.raises(ValueError, match="missing"):
        run(args)
    populate(args, "tune", 1e-4)
    failure = args.root / "tune/count-1/failure.json"
    failure.write_text('{"type":"OutOfMemoryError"}')
    with pytest.raises(ValueError, match="other_failure"):
        run(args)
    failure.unlink()
    for i in range(3):
        (args.root / f"tune/count-{i}/failure.json").write_text('{"category":"numerical"}')
    with pytest.raises(ValueError, match="shared smaller-rate extension"):
        run(args)
    assert not (args.root / "frozen.json").exists()


def test_mismatched_configuration_and_incomplete_metrics(args):
    populate(args, "screen")
    trial_dir = args.root / "screen/fresh-0"
    expected = load_config(args.configs / "screen.json", lr=1e-4)
    runtime = asdict(Runtime.from_args(args))
    assert read_trial(trial_dir, expected, "another-manifest", [16], runtime)["status"] == "invalid"
    (trial_dir / "metrics.jsonl").write_text("")
    assert read_trial(trial_dir, expected, digest(args.manifest), [16], runtime)["status"] == "incomplete"


@pytest.mark.parametrize("key,value", [("dtype", "bfloat16"), ("generation_batch", 4), ("score_batch", 2)])
def test_trial_selection_rejects_numerical_runtime_drift(args, key, value):
    populate(args, "screen")
    folder = args.root / "screen/fresh-0"
    path = folder / "run.json"
    record = json.loads(path.read_text())
    record["runtime"][key] = value
    path.write_text(json.dumps(record))
    expected = load_config(args.configs / "screen.json", lr=1e-4)
    result = read_trial(folder, expected, digest(args.manifest), [16], asdict(Runtime.from_args(args)))
    assert result["status"] == "invalid" and "runtime" in result["reason"]
    args.stage = "tune"
    with pytest.raises(ValueError, match="runtime"):
        run(args)


def test_missing_runtime_is_invalid_but_device_and_price_may_differ(args):
    populate(args, "screen")
    folder = args.root / "screen/fresh-0"
    path = folder / "run.json"
    record = json.loads(path.read_text())
    expected = load_config(args.configs / "screen.json", lr=1e-4)
    runtime = asdict(Runtime.from_args(args))
    record["runtime"].update(device="cuda:3", hourly_usd=7.5)
    path.write_text(json.dumps(record))
    assert read_trial(folder, expected, digest(args.manifest), [16], runtime)["status"] == "valid"
    del record["runtime"]["dtype"]
    path.write_text(json.dumps(record))
    result = read_trial(folder, expected, digest(args.manifest), [16], runtime)
    assert result["status"] == "invalid" and "runtime" in result["reason"]


def test_generated_jobs_must_use_frozen_numerical_runtime(args, capsys):
    populate(args, "screen")
    populate(args, "tune", 1e-4)
    args.stage = "freeze"
    run(args)
    capsys.readouterr()
    for stage in ("main", "test"):
        args.stage = stage
        for key, value in (("dtype", "bfloat16"), ("generation_batch", 4)):
            original = getattr(args, key)
            setattr(args, key, value)
            with pytest.raises(ValueError, match="runtime differs from freeze"):
                run(args)
            setattr(args, key, original)
    args.stage = "main"
    args.score_batch = 2
    with pytest.raises(ValueError, match="runtime differs from freeze"):
        run(args)
    # Final evaluation does not use score_batch; device indices and prices are logistical.
    args.stage = "test"
    args.device, args.hourly_usd = "cuda:2", 5.0
    run(args)
    assert len(capsys.readouterr().out.splitlines()) == 16


def test_old_freeze_requires_numerical_runtime(args):
    path = args.root / "frozen.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"version": 1}')
    with pytest.raises(ValueError, match="numerical runtime"):
        load_freeze(path, digest(args.manifest), asdict(load_config(args.configs / "main.json")))


def test_sealed_test_commands_require_freeze(args):
    args.stage = "test"
    with pytest.raises(FileNotFoundError, match="frozen.json"):
        run(args)


@pytest.mark.parametrize("metadata", [None, {"id": "legacy-prompt"}])
def test_trial_selection_rejects_missing_or_old_prompt(args, metadata):
    populate(args, "screen")
    folder = args.root / "screen/fresh-0"
    path = folder / "run.json"
    record = json.loads(path.read_text())
    record["prompt"] = metadata
    path.write_text(json.dumps(record))
    cfg = load_config(args.configs / "screen.json", lr=1e-4)
    result = read_trial(folder, cfg, digest(args.manifest), [16], asdict(Runtime.from_args(args)))
    assert result["status"] == "invalid" and "prompt metadata" in result["reason"]
