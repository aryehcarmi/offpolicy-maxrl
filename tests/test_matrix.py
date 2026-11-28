from argparse import Namespace
from dataclasses import asdict
import json
from pathlib import Path
import shlex

import pytest

from maxrl.config import Runtime, load_config
from maxrl.data import digest
from maxrl.prompts import prompt_metadata
from maxrl.matrix import METHODS, SEEDS, load_freeze, read_trial, run, trials


@pytest.fixture
def args(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n")
    return Namespace(stage="screen", root=tmp_path / "trials with spaces", manifest=manifest,
                     configs=Path(__file__).parents[1] / "configs", device="cpu", dtype="float32",
                     generation_batch=8, score_batch=1, hourly_usd=None)


def populate(args, stage, eta0=None, *, overrides=None):
    for method, lr, child in trials(stage, eta0):
        config = load_config(args.configs / f"{stage}.json", method=method, lr=lr)
        folder = args.root / child
        folder.mkdir(parents=True)
        (folder / "run.json").write_text(json.dumps({"config": asdict(config),
                                                    "runtime": asdict(Runtime.from_args(args)),
                                                    "prompt": prompt_metadata(),
                                                    "manifest_sha256": digest(args.manifest)}))
        scores = {0: .125, 16: .25, 32: .25} if stage == "screen" else {64: .25, 80: .25, 96: .25}
        if overrides:
            scores.update(overrides.get(child.name, {}))
        events = [{"split": "validation", "step": step, "pass@1": score,
                   "prompts": 128, "responses": 1024} for step, score in scores.items()]
        (folder / "metrics.jsonl").write_text("\n".join(map(json.dumps, events)) + "\n")


def assert_tune_commands(stdout, expected_rates):
    commands = [shlex.split(line) for line in stdout.splitlines()]
    assert len(commands) == len(METHODS) * len(expected_rates)
    for method in METHODS:
        method_commands = [c for c in commands if c[c.index("--method") + 1] == method]
        assert [float(c[c.index("--lr") + 1]) for c in method_commands] == pytest.approx(expected_rates)
        for index, command in enumerate(method_commands):
            assert command[:3] == ["python", "-m", "maxrl.train"]
            assert Path(command[command.index("--out") + 1]).name == f"{method}-{index}"


def test_screen_commands(args, capsys):
    run(args)
    output = capsys.readouterr()
    commands = [shlex.split(line) for line in output.out.splitlines()]
    assert output.err == ""
    assert [float(c[c.index("--lr") + 1]) for c in commands] == [3e-7, 1e-6, 3e-6]
    for command in commands:
        assert command[:3] == ["python", "-m", "maxrl.train"]
        assert command[command.index("--method") + 1] == "fresh"
        assert command[command.index("--seed") + 1] == "17"


def test_screen_tune_freeze_and_main_matrix(args, capsys):
    populate(args, "screen")
    args.stage = "tune"
    run(args)
    output = capsys.readouterr()
    assert_tune_commands(output.out, [1e-7, 3e-7, 9e-7])
    assert output.err == ""
    populate(args, "tune", 3e-7, overrides={
        f"{method}-1": {64: .625, 80: .375, 96: .125} for method in METHODS})
    # A genuine numerical failure remains in the grid, even with prior metrics.
    (args.root / "tune/count-1/failure.json").write_text('{"category":"numerical"}')
    args.stage = "freeze"
    run(args)
    frozen = json.loads((args.root / "frozen.json").read_text())
    assert frozen["version"] == 3
    assert frozen["prompt"] == prompt_metadata()
    assert frozen["runtime"] == {"dtype": "float32", "generation_batch": 8, "score_batch": 1}
    assert len(frozen["screen"]) == 3 and len(frozen["tune"]) == 12
    assert frozen["eta0"] == 3e-7
    assert frozen["screen_outcome"] == {"gain": .125, "margin": .02, "met_margin": True, "fallback_used": False}
    assert frozen["rates"] == {m: (3e-7 / 3 if m == "count" else 3e-7) for m in METHODS}
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


def test_screen_at_one_micro_adds_lower_rate_for_every_method(args, capsys):
    populate(args, "screen", overrides={"fresh-1": {16: .375}})
    args.stage = "tune"
    run(args)
    assert_tune_commands(capsys.readouterr().out, [1.5e-7, 1e-6 / 3, 1e-6, 3e-6])


@pytest.mark.parametrize("numerical", [(), (0, 2), (0, 1, 2)], ids=["collapsed", "mixed", "numerical"])
def test_screen_without_selectable_rate_uses_replay_fallback(args, capsys, numerical):
    populate(args, "screen", overrides={f"fresh-{i}": {32: 0} for i in range(3)})
    for i in numerical:
        folder = args.root / f"screen/fresh-{i}"
        metrics = folder / "metrics.jsonl"
        baseline = metrics.read_text().splitlines()[0]
        metrics.write_text(baseline + "\n")
        (folder / "failure.json").write_text('{"category":"numerical"}')
    args.stage = "tune"
    run(args)
    output = capsys.readouterr()
    assert_tune_commands(output.out, [5e-8, 1.5e-7, 4.5e-7])
    assert len(output.err.splitlines()) == 1 and "fallback" in output.err.lower()
    populate(args, "tune", 1.5e-7)
    args.stage = "freeze"
    run(args)
    frozen = json.loads((args.root / "frozen.json").read_text())
    assert frozen["eta0"] == 1.5e-7
    assert frozen["screen_outcome"] == {"gain": None, "margin": .02, "met_margin": False, "fallback_used": True}
    assert [r["status"] for r in frozen["screen"]] == [
        "numerical_failure" if i in numerical else "collapsed" for i in range(3)]


def test_screen_excludes_highest_mean_when_last_score_is_zero(args):
    # The first trial's mean is .375, and each selectable trial scores .25.
    populate(args, "screen", overrides={"fresh-0": {16: .75, 32: 0}})
    populate(args, "tune", 1e-6)
    args.stage = "freeze"
    run(args)
    frozen = json.loads((args.root / "frozen.json").read_text())
    assert frozen["eta0"] == 1e-6
    assert frozen["screen"][0]["status"] == "collapsed"


def test_small_screen_gain_proceeds_and_reports_notice_on_stderr(args, capsys):
    # Only steps 16 and 32 count in the ranking; the gain uses the winner's own baseline.
    baseline = 304 / 1024
    populate(args, "screen", overrides={"fresh-0": {0: .5}, "fresh-1": {0: baseline, 16: .375}})
    args.stage = "tune"
    run(args)
    output = capsys.readouterr()
    assert_tune_commands(output.out, [1.5e-7, 1e-6 / 3, 1e-6, 3e-6])
    assert len(output.err.splitlines()) == 1 and "margin" in output.err.lower()
    populate(args, "tune", 1e-6)
    args.stage = "freeze"
    run(args)
    frozen = json.loads((args.root / "frozen.json").read_text())
    assert frozen["eta0"] == 1e-6
    assert frozen["screen"][1]["validation"]["0"] == baseline
    assert frozen["screen_outcome"] == {
        "gain": (.375 + .25) / 2 - baseline, "margin": .02, "met_margin": False, "fallback_used": False}


def test_tune_excludes_highest_mean_when_last_score_is_zero(args):
    populate(args, "screen")
    populate(args, "tune", 3e-7, overrides={"count-1": {64: .625, 80: .375, 96: 0}})
    args.stage = "freeze"
    run(args)
    frozen = json.loads((args.root / "frozen.json").read_text())
    assert frozen["rates"]["count"] == 3e-7 / 3
    collapsed = [r for r in frozen["tune"] if r["status"] == "collapsed"]
    assert [(r["method"], r["lr"]) for r in collapsed] == [("count", 3e-7)]


@pytest.mark.parametrize("method", ["count", "snis"])
def test_collapsed_primary_tuning_grid_requires_smaller_rate_extension(args, method):
    populate(args, "screen")
    populate(args, "tune", 3e-7, overrides={f"{method}-{i}": {96: 0} for i in range(3)})
    args.stage = "freeze"
    with pytest.raises(ValueError, match="shared smaller-rate extension"):
        run(args)
    assert not (args.root / "frozen.json").exists()


@pytest.mark.parametrize("change,status", [
    ("absent", "incomplete"), ("absent-at-numerical-failure", "incomplete"),
    ("duplicate", "invalid"), ("score", "invalid"), ("count", "invalid")])
def test_screen_requires_valid_step_zero_before_tuning(args, change, status):
    populate(args, "screen")
    folder = args.root / "screen/fresh-0"
    metrics = folder / "metrics.jsonl"
    events = [json.loads(line) for line in metrics.read_text().splitlines()]
    baseline = next(e for e in events if e["step"] == 0)
    if change in ("absent", "absent-at-numerical-failure"):
        events.remove(baseline)
        if change == "absent-at-numerical-failure":
            (folder / "failure.json").write_text('{"category":"numerical"}')
    elif change == "duplicate":
        events.append(baseline)
    elif change == "score":
        baseline["pass@1"] = 1.5
    else:
        baseline["responses"] = 1
    metrics.write_text("\n".join(map(json.dumps, events)) + "\n")
    cfg = load_config(args.configs / "screen.json", lr=3e-7)
    trial = read_trial(folder, cfg, digest(args.manifest), "screen", asdict(Runtime.from_args(args)))
    assert trial["status"] == status
    args.stage = "tune"
    with pytest.raises(ValueError):
        run(args)


def test_pending_or_operational_failure_cannot_freeze(args):
    args.stage = "freeze"
    populate(args, "screen")
    with pytest.raises(ValueError, match="missing"):
        run(args)
    populate(args, "tune", 3e-7)
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
    expected = load_config(args.configs / "screen.json", lr=3e-7)
    runtime = asdict(Runtime.from_args(args))
    assert read_trial(trial_dir, expected, "another-manifest", "screen", runtime)["status"] == "invalid"
    (trial_dir / "metrics.jsonl").write_text("")
    assert read_trial(trial_dir, expected, digest(args.manifest), "screen", runtime)["status"] == "incomplete"


@pytest.mark.parametrize("key,value", [("dtype", "bfloat16"), ("generation_batch", 4), ("score_batch", 2)])
def test_trial_selection_rejects_numerical_runtime_drift(args, key, value):
    populate(args, "screen")
    folder = args.root / "screen/fresh-0"
    path = folder / "run.json"
    record = json.loads(path.read_text())
    record["runtime"][key] = value
    path.write_text(json.dumps(record))
    expected = load_config(args.configs / "screen.json", lr=3e-7)
    result = read_trial(folder, expected, digest(args.manifest), "screen", asdict(Runtime.from_args(args)))
    assert result["status"] == "invalid" and "runtime" in result["reason"]
    args.stage = "tune"
    with pytest.raises(ValueError, match="runtime"):
        run(args)


def test_missing_runtime_is_invalid_but_device_and_price_may_differ(args):
    populate(args, "screen")
    folder = args.root / "screen/fresh-0"
    path = folder / "run.json"
    record = json.loads(path.read_text())
    expected = load_config(args.configs / "screen.json", lr=3e-7)
    runtime = asdict(Runtime.from_args(args))
    record["runtime"].update(device="cuda:3", hourly_usd=7.5)
    path.write_text(json.dumps(record))
    assert read_trial(folder, expected, digest(args.manifest), "screen", runtime)["status"] == "valid"
    del record["runtime"]["dtype"]
    path.write_text(json.dumps(record))
    result = read_trial(folder, expected, digest(args.manifest), "screen", runtime)
    assert result["status"] == "invalid" and "runtime" in result["reason"]


def test_generated_jobs_must_use_frozen_numerical_runtime(args, capsys):
    populate(args, "screen")
    populate(args, "tune", 3e-7)
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
    cfg = load_config(args.configs / "screen.json", lr=3e-7)
    result = read_trial(folder, cfg, digest(args.manifest), "screen", asdict(Runtime.from_args(args)))
    assert result["status"] == "invalid" and "prompt metadata" in result["reason"]
