"""Exercise the planner, trial workers and recovery with a filesystem-backed Modal stub."""
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from maxrl import compare
from maxrl.config import Runtime, load_config
from maxrl.data import digest
from maxrl.prompts import prompt_metadata
from maxrl.schema import Response
from test_matrix import populate
from test_modal_countdown import PROVENANCE, REPO, import_launcher, write_data_fixture


RATES = {"gpu_hour_cost_a10g": "3", "cpu_hour_cost": ".25", "mem_gib_hour_cost": ".1"}


class LocalVolume:
    """Expose the small subset of the SDK used by the local entrypoint."""

    def __init__(self, root):
        self.root = root
        self.reads = []
        self.uploads = []

    def read_file(self, name):
        path = Path(str(name).lstrip("/"))
        assert "checkpoints" not in path.parts, "the laptop attempted to download a checkpoint"
        self.reads.append(str(path))
        yield (self.root / path).read_bytes()

    def listdir(self, path):
        directory = self.root / str(path).lstrip("/")
        return [SimpleNamespace(path=str(p.relative_to(self.root))) for p in directory.iterdir()]

    @contextmanager
    def batch_upload(self, *, force=False):
        def put_file(source, name):
            target = self.root / str(name).lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not force:
                raise FileExistsError(target)
            target.write_bytes(source.read() if hasattr(source, "read") else Path(source).read_bytes())
            self.uploads.append((str(name).lstrip("/"), force))
        yield SimpleNamespace(put_file=put_file)


@pytest.mark.filterwarnings("ignore:The MatrixTrials.run function is executing locally:UserWarning")
def test_repeated_worker_handoff_only_runs_the_owning_call(tmp_path, monkeypatch):
    modal = pytest.importorskip("modal")
    app = import_launcher(with_modal=True)
    volume = LocalVolume(tmp_path)
    reloads, executions = [], []
    monkeypatch.setattr(app, "volume", volume)
    monkeypatch.setattr(volume, "reload", lambda: reloads.append(True), raising=False)
    monkeypatch.setattr(volume, "commit", lambda: None, raising=False)

    def attempt(*args, **kwargs):
        executions.append((kwargs["attempt_id"], kwargs["function_call_id"]))
        return {"status": "completed"}

    monkeypatch.setattr(app, "run_attempt", attempt)
    calls = iter(("fc-original", "fc-duplicate", "fc-original", "fc-new", "fc-invalid"))
    monkeypatch.setattr(modal, "current_function_call_id", lambda: next(calls))
    invoke = app.MatrixTrials().run.local
    args = ("screen", "planner command", RATES, PROVENANCE)
    name = app.new_attempt_id("matrix")
    assert invoke(*args, attempt_id=name) == {"status": "completed"}
    assert invoke(*args, attempt_id=name) == {"existing_worker_call_id": "fc-original"}
    assert invoke(*args, attempt_id=name) == {"status": "completed"}
    assert invoke(*args) == {"status": "completed"}
    assert executions[:2] == [(name, "fc-original")] * 2
    assert executions[2][0] != name and executions[2][1] == "fc-new"
    with pytest.raises(ValueError, match="attempt ID"):
        invoke(*args, attempt_id="matrix-../outside/path")
    assert len(reloads) == len(executions) == 3
    assert json.loads((tmp_path / name / "worker.json").read_bytes()) == {"function_call_id": "fc-original"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    app = import_launcher()
    repo = tmp_path / "local checkout"
    repo.mkdir()
    for name in ("configs", "src"):
        (repo / name).symlink_to(REPO / name, target_is_directory=True)
    root = tmp_path / "remote volume"
    pinned = write_data_fixture(root / "data")
    monkeypatch.setattr(app, "MANIFEST_SHA256", pinned)
    return SimpleNamespace(app=app, repo=repo, volume=LocalVolume(root), root=root,
                           matrix=repo / "runs/matrix", manifest_hash=pinned)


def record_attempt(env, trial, *, serial=1, status="completed", files=None, **extra):
    """Write the durable worker record and inventory; preserve earlier attempts."""
    stage = "test" if trial.endswith("/test") else trial.split("/")[0]
    attempt = f"matrix-20000101T0100{serial:02d}000000Z-{trial.replace('/', '-')}"
    directory = env.root / attempt
    directory.mkdir()
    files = dict(files or {})
    if status != "running":
        files.update({"run.log": b"worker log\n", "gpu.csv": b"timestamp,gpu\n"})
    for name, content in files.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    record = {
        "attempt_id": attempt, "stage": "matrix", "matrix_stage": stage, "trial": trial,
        "function_call_id": f"fc-{attempt}", "status": status,
        "started_utc": f"2000-01-01T01:00:{serial:02d}Z",
        "manifest_sha256": env.manifest_hash, "hourly_usd": 5.1,
        "sha256": {name: digest(directory / name) for name in files}, **extra,
    }
    if status != "running":
        record.update(finished_utc=f"2000-01-01T02:00:{serial:02d}Z", estimated_cost_usd=5.1)
    (directory / "attempt.json").write_text(json.dumps(record))
    return record


def training_files():
    return {"output/run.json": b"{}", "output/metrics.jsonl": b'{"split":"validation"}\n',
            "output/resources.csv": b"phase,seconds\ntotal,1\n"}


def collect(env, call_running=lambda call_id: False):
    from scripts.modal_matrix_store import collect_matrix

    return collect_matrix(env.repo, env.volume, call_running, env.app.download_attempt)


def test_collect_polls_claimed_worker_before_it_writes_attempt(env):
    record = record_attempt(env, "screen/fresh-0", status="running")
    folder = env.root / record["attempt_id"]
    (folder / "attempt.json").rename(folder / "dispatch.json")
    (folder / "worker.json").write_text(json.dumps({"function_call_id": "fc-live-owner"}))
    polled = []

    def is_running(call_id):
        polled.append(call_id)
        return call_id == "fc-live-owner"

    state, active = collect(env, is_running)["screen/fresh-0"]
    assert state == "running" and polled == [active["function_call_id"]] == ["fc-live-owner"]


def test_collect_numerical_failure_and_remove_superseded_operational_result(env, capsys):
    numerical = dict(training_files(), **{"output/failure.json": b'{"category":"numerical"}'})
    done = record_attempt(env, "tune/count-0", status="failed", files=numerical)
    old = record_attempt(env, "tune/count-1", files=training_files())
    collect(env)
    assert (env.matrix / "tune/count-1/run.json").is_file()
    failure = dict(training_files(), **{"output/failure.json": b'{"category":"operational"}'})
    failed = record_attempt(env, "tune/count-1", serial=2, status="failed", files=failure)
    status = collect(env)
    assert status["tune/count-0"] == ("done", done)
    assert status["tune/count-1"] == ("failed", failed)
    assert json.loads((env.matrix / "tune/count-0/failure.json").read_text()) == {"category": "numerical"}
    assert json.loads((env.matrix / "tune/count-0/attempt.json").read_text()) == done
    assert (env.repo / "runs/countdown" / done["attempt_id"] / "run.log").is_file()
    assert (env.repo / "runs/countdown" / old["attempt_id"] / "output/run.json").is_file()
    assert not (env.matrix / "tune/count-1/run.json").exists()
    assert not (env.repo / "results").exists()
    table = capsys.readouterr().out
    assert all(value in table for value in ("tune/count-1", "failed", failed["attempt_id"],
                                             failed["started_utc"], failed["finished_utc"], "5.1"))


def test_collect_rejects_corrupt_file_without_publishing_partial_folder(env):
    record = record_attempt(env, "screen/fresh-0", files=training_files())
    (env.root / record["attempt_id"] / "output/resources.csv").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="SHA-256"):
        collect(env)
    assert not (env.matrix / "screen/fresh-0").exists()
    assert not (env.repo / "runs/countdown" / record["attempt_id"]).exists()


def planner_fixtures(env, tmp_path, stage):
    args = SimpleNamespace(root=tmp_path / "planner fixtures", configs=env.repo / "configs",
                           manifest=env.root / "data/manifest.json", **asdict(Runtime()))
    populate(args, stage, 3e-7 if stage == "tune" else None)
    for folder in sorted((args.root / stage).iterdir()):
        files = {f"output/{name}": (folder / name).read_bytes() for name in ("run.json", "metrics.jsonl")}
        files["output/resources.csv"] = b"phase,seconds\ntotal,1\n"
        record_attempt(env, str(folder.relative_to(args.root)), files=files)


def test_collected_final_test_is_readable_by_compare(env):
    method, seed, rate = "count", 101, 3e-7
    trial = f"main/{method}-{seed}"
    cfg = asdict(load_config(env.repo / "configs/main.json", method=method, seed=seed, lr=rate))
    base = dict(config=cfg, manifest_sha256=env.manifest_hash,
                runtime=asdict(Runtime()), prompt=prompt_metadata())
    train = dict(base, run_id="trajectory", initial_checkpoint_id="initial")
    identity = "trajectory/step-000128"
    eval_run = dict(base, checkpoint_id=identity, training_config=cfg, panel="test")
    training = training_files()
    training["output/run.json"] = json.dumps(train).encode()
    parent = record_attempt(env, trial, files=training)
    rows = [asdict(Response("puzzle", [1], [2], [-.5], [1], [1], int(i == 0),
                            identity, f"sample-{i}", f"{identity}/test")) for i in range(cfg["test_samples"])]
    files = {"output/run.json": json.dumps(eval_run).encode(),
             "output/responses.jsonl": "\n".join(map(json.dumps, rows)).encode(),
             "output/resources.csv": b"phase,seconds\ntotal,1\n", "output/metrics.json": b"{}"}
    evaluated = record_attempt(env, f"{trial}/test", files=files, main_attempt=parent["attempt_id"])
    collect(env)
    collect(env)  # A second publication of the parent must preserve the nested test.
    frozen = dict(main_config=cfg, manifest_sha256=env.manifest_hash,
                  rates={method: rate}, runtime=asdict(Runtime()))
    counts, checkpoint, origin = compare.test_counts(env.matrix, frozen, method, seed, {"puzzle": 0}, {})
    assert (counts.tolist(), checkpoint, origin) == ([1], identity, "initial")
    test_folder = env.matrix / trial / "test"
    assert {file.name for file in test_folder.iterdir()} == {
        "run.json", "metrics.json", "resources.csv", "responses.jsonl", "attempt.json"}
    assert json.loads((test_folder / "attempt.json").read_text()) == evaluated


def cli_lines(env, stage):
    """Read the planner's own stdout for an independent comparison with dispatch."""
    import os
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "-m", "maxrl.matrix", stage], cwd=env.repo,
                            env={**os.environ, "PYTHONPATH": str(env.repo / "src")},
                            capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


@pytest.mark.parametrize("stage,original,replacement", [
    ("screen", "python -m maxrl.train", "python -m maxrl.replay"),
    ("tune", "python -m maxrl.train", "sh -c maxrl.train"),
    ("main", "python -m maxrl.train", "python -m maxrl.evaluate"),
    ("test", "python -m maxrl.evaluate", "python -m maxrl.train"),
    ("test", "--panel test", "--panel validation"),
])
def test_worker_refuses_nonplanner_command(env, stage, original, replacement):
    from maxrl.matrix import command

    args = SimpleNamespace(configs=Path("configs"), manifest=Path("data/manifest.json"),
                           **asdict(Runtime()))
    training = "main" if stage == "test" else stage
    line = command(args, training, "count", 1e-6, 101,
                   Path("runs/matrix") / training / "count-101", test=stage == "test")
    assert env.app.parse_matrix_command(stage, line)["matrix_stage"] == stage
    with pytest.raises(ValueError):
        env.app.parse_matrix_command(stage, line.replace(original, replacement))


def fake_workload(argv, repo, emit, commands):
    """Model-free replacement for maxrl.train / maxrl.evaluate in an otherwise real worker."""
    argv = [str(value) for value in argv]
    if argv[1:2] == ["-c"]:
        return  # The GPU was stubbed and is checked separately by the existing launcher tests.
    commands.append(argv)
    flag = lambda name: argv[argv.index(name) + 1]
    out = Path(flag("--out"))
    out.mkdir()
    (out / "resources.csv").write_text("phase,seconds\ntotal,1\n")
    if argv[2] == "maxrl.evaluate":
        for name in ("run.json", "metrics.json", "responses.jsonl"):
            (out / name).write_text("{}\n")
        return
    assert argv[2] == "maxrl.train"
    cfg = load_config(repo / flag("--config"), method=flag("--method"),
                      lr=float(flag("--lr")), seed=int(flag("--seed")))
    run = {"run_id": f"{cfg.method}-{cfg.seed}", "config": asdict(cfg),
           "manifest_sha256": digest(Path(flag("--manifest"))), "prompt": prompt_metadata(),
           "runtime": asdict(Runtime())}
    (out / "run.json").write_text(json.dumps(run))
    (out / "metrics.jsonl").write_text("{}\n")
    for step in set([*cfg.checkpoint_steps, cfg.updates]):
        checkpoint = out / "checkpoints" / f"step-{step:06d}"
        checkpoint.mkdir(parents=True)
        for name in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"):
            (checkpoint / name).write_bytes(name.encode())
        (checkpoint / "maxrl_checkpoint.json").write_text(json.dumps({
            "checkpoint_id": f"{run['run_id']}/{checkpoint.name}", "prompt": run["prompt"]}))


@pytest.fixture
def cluster(env, monkeypatch):
    """Submit each call immediately; run its stubbed workload when the coordinator waits on it."""
    import sys
    from unittest.mock import Mock

    app = env.app
    cli = env.repo / ".venv/bin/python"
    cli.parent.mkdir(parents=True)
    cli.symlink_to(sys.executable)
    monkeypatch.setenv("PYTHONPATH", str(env.repo / "src"))
    monkeypatch.setattr(app, "volume", env.volume, raising=False)
    monkeypatch.setattr(app, "_matrix_workspace", lambda: "stub-workspace")
    monkeypatch.setattr(app, "VOLUME_ROOT", env.root)
    monkeypatch.setattr(app, "git_provenance", lambda repo: PROVENANCE)
    monkeypatch.setattr(app, "_gpu_name", lambda: "NVIDIA A10")
    result = SimpleNamespace(env=env, launched=[], events=[], commands=[], live=set(),
                             coordinator_id=None, calls={}, reattached=[])

    @contextmanager
    def telemetry(path, emit):
        path.write_text("timestamp,gpu\n")
        yield

    monkeypatch.setattr(app, "_sample_gpu", telemetry)
    monkeypatch.setattr(app, "_run", lambda argv, repo, emit: fake_workload(argv, repo, emit, result.commands))
    original_billing = app.subprocess.check_output

    def billing(argv, *args, **kwargs):
        if argv[:3] == ["modal", "billing", "rates"]:
            return json.dumps(RATES)
        return original_billing(argv, *args, **kwargs)

    monkeypatch.setattr(app.subprocess, "check_output", billing)

    def preflight(provenance):
        assert provenance == PROVENANCE
        result.events.append("preflight")
        attempt = f"preflight-{len(result.events)}"
        folder = env.root / attempt
        folder.mkdir()
        record = dict(stage="preflight", status="completed", attempt_id=attempt, sha256={})
        (folder / "attempt.json").write_text(json.dumps(record))
        return record

    def spawn(matrix_stage, command, rates, provenance, freeze_sha256, main_attempt, attempt_id):
        result.events.append("spawn")
        result.launched.append(command)
        call_id = f"fc-{attempt_id}"
        if call_id in result.calls:
            call_id += "-resubmitted"
        call = Mock(object_id=call_id)

        def get():
            result.events.append("wait")
            return app.run_attempt(env.repo, env.root, lambda: None, stage="matrix", matrix_stage=matrix_stage,
                                   planner_command=command, rates=rates, provenance=provenance,
                                   freeze_sha256=freeze_sha256, main_attempt=main_attempt, attempt_id=attempt_id,
                                   function_call_id=call.object_id)

        call.get = get
        result.calls[call.object_id] = call
        return call

    def start_stage(*args):
        result.coordinator_id = f"fc-coordinator-{len(result.events)}"
        return SimpleNamespace(object_id=result.coordinator_id, get=lambda: app.dispatch_matrix(*args))

    def reattach(call_id):
        result.reattached.append(call_id)
        assert call_id != result.coordinator_id, "the coordinator tried to wait on itself"
        return result.calls[call_id]

    monkeypatch.setattr(app, "preflight", SimpleNamespace(remote=preflight), raising=False)
    monkeypatch.setattr(app, "run_matrix_trial", SimpleNamespace(spawn=spawn), raising=False)
    monkeypatch.setattr(app, "run_matrix_stage", SimpleNamespace(spawn=start_stage), raising=False)
    monkeypatch.setattr(app, "_matrix_call_running", result.live.__contains__)
    monkeypatch.setattr(app, "modal", SimpleNamespace(current_function_call_id=lambda: result.coordinator_id,
                                                       FunctionCall=SimpleNamespace(from_id=reattach)))
    result.launch = lambda stage, **kw: app.launch_matrix(env.repo, stage, **kw)
    return result


def expected_argv(line, env, actual, checkpoint=None):
    expected = shlex.split(line)
    out = Path(actual[actual.index("--out") + 1])
    assert out.is_absolute() and not out.is_relative_to(env.repo) and not out.is_relative_to(env.root)
    substitutions = {"--manifest": str(env.root / "data/manifest.json"), "--out": str(out),
                     "--hourly-usd": str(env.app.hourly_rate(RATES))}
    if checkpoint is not None:
        substitutions["--checkpoint"] = str(checkpoint)
    for key, value in substitutions.items():
        if key in expected:
            expected[expected.index(key) + 1] = value
        else:
            assert key == "--hourly-usd"
            expected.extend([key, value])
    return expected


def test_screen_exact_dispatch_and_dry_run(cluster, capsys):
    env = cluster.env
    cluster.launch("screen", dry_run=True)
    preview = capsys.readouterr().out
    expected = cli_lines(env, "screen")
    assert not cluster.events
    assert all(f"screen/fresh-{i}" in preview for i in range(3))
    assert env.app.verify_manifest(env.repo / "data/manifest.json") == env.manifest_hash
    cluster.launch("screen")
    assert cluster.launched == expected
    assert not (env.root / "matrix/methods.json").exists()
    assert cluster.events == ["preflight", "spawn", "spawn", "spawn", "wait", "wait", "wait"]
    for line, argv in zip(expected, cluster.commands, strict=True):
        assert argv == expected_argv(line, env, argv)
    records = [json.loads(path.read_text()) for path in env.root.glob("matrix-*/attempt.json")]
    assert len(records) == 3
    for rec in records:
        line = expected[int(rec["trial"].rsplit("-", 1)[1])]
        assert rec["status"] == "completed"
        assert (rec["matrix_stage"], rec["planner_command"], rec["method"], rec["seed"]) == (
            "screen", line, "fresh", 17)
        assert rec["rate"] == float(shlex.split(line)[shlex.split(line).index("--lr") + 1])
        assert rec["function_call_id"] == f"fc-{rec['attempt_id']}"
        assert rec["verified_checkpoints"][-1].endswith("/step-000032")
        assert "gpu.csv" in rec["sha256"] and "run.log" in rec["sha256"]
    assert not (env.repo / "results").exists()


def test_relaunch_obeys_latest_attempt_and_rerun(cluster):
    env = cluster.env
    files = training_files()
    record_attempt(env, "screen/fresh-0", files=files)
    live = record_attempt(env, "screen/fresh-1", status="running")
    cluster.live.add(live["function_call_id"])
    record_attempt(env, "screen/fresh-2", files=files)
    record_attempt(env, "screen/fresh-2", serial=2, status="failed", files=files)
    cluster.launch("screen")
    expected = cli_lines(env, "screen")
    assert cluster.launched == [expected[2]]
    cluster.launch("screen")
    assert cluster.launched == [expected[2]]
    cluster.launch("screen", rerun="screen/fresh-0")
    assert cluster.launched == [expected[2], expected[0]]


def test_ended_call_with_running_record_restarts(cluster):
    env = cluster.env
    for index in range(2):
        record_attempt(env, f"screen/fresh-{index}", files=training_files())
    stale = record_attempt(env, "screen/fresh-2", status="running")
    cluster.launch("screen")
    assert cluster.launched == [cli_lines(env, "screen")[2]]
    assert (env.root / stale["attempt_id"] / "attempt.json").is_file()


def test_tune_dispatches_planner_commands_for_owned_methods(cluster, tmp_path, capsys):
    env = cluster.env
    planner_fixtures(env, tmp_path, "screen")
    with pytest.raises(ValueError, match="12"):
        cluster.launch("tune", dry_run=True)
    assert capsys.readouterr().out.count("  would start") == 12
    methods = "count,snis"
    cluster.launch("tune", methods=methods, dry_run=True)
    assert not (env.root / "matrix/methods.json").exists()
    assert cluster.coordinator_id is None
    header = capsys.readouterr().out.splitlines()[0]
    assert "stub-workspace" in header and all(method in header for method in methods.split(","))

    cluster.launch("tune", methods=methods)
    expected = [line for line in cli_lines(env, "tune")
                if any(f"--method {method} " in line for method in methods.split(","))]
    assert cluster.launched == expected and len(expected) == 6
    for command, argv in zip(expected, cluster.commands, strict=True):
        assert argv == expected_argv(command, env, argv)

    # The other six planner trials are still missing. An omitted option must
    # recover ownership from the Volume and skip the six completed here.
    cluster.launch("tune")
    cluster.launch("tune", methods="snis,count")
    assert cluster.launched == expected
    assert [force for path, force in env.volume.uploads if path == "matrix/methods.json"] == [False]
    capsys.readouterr()
    cluster.launch("collect")
    header = capsys.readouterr().out.splitlines()[0]
    assert "stub-workspace" in header and all(method in header for method in methods.split(","))
    assert "fresh" not in header and "uncorrected" not in header

    coordinator = cluster.coordinator_id
    for dry_run in (True, False):
        with pytest.raises(ValueError, match="(?i)method|own"):
            cluster.launch("tune", methods="fresh,uncorrected", dry_run=dry_run)
        with pytest.raises(ValueError, match="(?i)own"):
            cluster.launch("tune", rerun="tune/fresh-0", dry_run=dry_run)
    assert cluster.coordinator_id == coordinator and cluster.launched == expected


def test_method_names_and_invalid_options(cluster):
    from maxrl.matrix import METHODS

    assert set(cluster.env.app.MATRIX_METHODS) == set(METHODS)
    with pytest.raises(ValueError, match="unknown-estimator"):
        cluster.launch("screen", methods="fresh,unknown-estimator")
    with pytest.raises(ValueError, match="methods"):
        cluster.launch("collect", methods="fresh")
    assert cluster.coordinator_id is None
    assert not (cluster.env.root / "matrix/methods.json").exists()


@pytest.mark.parametrize("last_trial,allowed", [
    ("tune/uncorrected-3", False),  # Eight running, plus three new.
    ("screen/fresh-0", True),       # One planned trial already runs: eight plus two.
    ("tune/fresh-0", False),        # A still-live superseded attempt also uses a GPU.
])
def test_gpu_limit_counts_running_attempts_before_claim_or_dispatch(cluster, capsys, last_trial, allowed):
    env = cluster.env
    assert env.app.MATRIX_GPU_LIMIT == 10
    trials = [f"tune/{method}-{i}" for method in ("fresh", "uncorrected") for i in range(4)]
    trials[-1] = last_trial
    for serial, trial in enumerate(trials, start=1):
        running = record_attempt(env, trial, serial=serial, status="running")
        cluster.live.add(running["function_call_id"])

    methods = "fresh,uncorrected"
    if allowed:
        cluster.launch("screen", methods=methods)
        assert cluster.launched == cli_lines(env, "screen")[1:]
        assert (env.root / "matrix/methods.json").is_file()
        return

    with pytest.raises(ValueError) as preview:
        cluster.launch("screen", methods=methods, dry_run=True)
    output = capsys.readouterr().out
    for i in range(3):
        assert f"screen/fresh-{i}  would start" in output
    with pytest.raises(ValueError) as launch:
        cluster.launch("screen", methods=methods)
    assert str(preview.value) == str(launch.value)
    assert "8" in str(launch.value) and "3" in str(launch.value)
    assert cluster.coordinator_id is None and not cluster.launched
    assert not (env.root / "matrix/methods.json").exists()


def test_main_creates_freeze_and_test_checks_final_checkpoint(cluster, tmp_path, capsys):
    env = cluster.env
    for stage in ("screen", "tune"):
        planner_fixtures(env, tmp_path, stage)
    # Run just one main trajectory; every other test must be explicitly skipped.
    cluster.launch("main", rerun="main/count-101")
    frozen = env.matrix / "frozen.json"
    remote_freeze = env.root / "matrix/frozen.json"
    assert frozen.read_bytes() == remote_freeze.read_bytes()
    assert cluster.launched == [cli_lines(env, "main")[0]]
    parent, = (json.loads(p.read_text()) for p in env.root.glob("matrix-*/attempt.json")
               if json.loads(p.read_text())["matrix_stage"] == "main")
    assert parent["freeze_sha256"] == digest(frozen)
    capsys.readouterr()
    cluster.launch("test")
    assert cluster.launched[-1:] == [cli_lines(env, "test")[0]]
    assert len(cluster.launched) == 2
    report = capsys.readouterr().out
    assert "main/snis-105/test" in report and "skip" in report.lower()
    checkpoint = env.root / parent["attempt_id"] / "output/checkpoints/step-000128"
    argv = cluster.commands[-1]
    assert argv == expected_argv(cluster.launched[-1], env, argv, checkpoint)
    test_record, = (json.loads(p.read_text()) for p in env.root.glob("matrix-*/attempt.json")
                    if json.loads(p.read_text())["matrix_stage"] == "test")
    assert test_record["freeze_sha256"] == digest(frozen)
    (checkpoint / "config.json").write_bytes(b"tampered checkpoint")
    # The worker must catch the mismatch before invoking the evaluator.
    calls = len(cluster.commands)
    record = env.app.run_attempt(
        env.repo, env.root, lambda: None, stage="matrix", matrix_stage="test",
        planner_command=cluster.launched[-1], function_call_id="fc-check",
        main_attempt=parent["attempt_id"], freeze_sha256=digest(frozen), rates=RATES,
        provenance=PROVENANCE)
    assert record["status"] == "failed" and "SHA-256" in record["error"]
    assert len(cluster.commands) == calls
    remote_freeze.write_text('{"different":"freeze"}\n')
    with pytest.raises((RuntimeError, ValueError), match="(?i)freeze|frozen"):
        cluster.launch("main")
    assert len(cluster.launched) == 2


def test_queued_dispatch_uses_call_liveness_until_worker_publishes(cluster, capsys):
    env = cluster.env
    queued = record_attempt(env, "screen/fresh-0", status="running")
    pending = env.root / queued["attempt_id"]
    (pending / "attempt.json").rename(pending / "dispatch.json")
    cluster.live.add(queued["function_call_id"])
    cluster.launch("screen", dry_run=True)
    assert "screen/fresh-0  skipped because running" in capsys.readouterr().out
    cluster.live.clear()
    cluster.launch("screen", dry_run=True)
    assert "screen/fresh-0  would start" in capsys.readouterr().out
    assert not cluster.events


def test_planner_error_is_visible_and_no_trial_is_started(cluster, capfd):
    with pytest.raises(RuntimeError):
        cluster.launch("tune")
    assert "screen has missing" in capfd.readouterr().err
    assert cluster.events == ["preflight"]


def test_late_invalid_command_prevents_dispatch_of_valid_prefix(cluster, monkeypatch):
    env = cluster.env
    commands = cli_lines(env, "screen")
    commands[-1] = commands[-1].replace("maxrl.train", "maxrl.evaluate")
    monkeypatch.setattr(env.app, "_run_planner", lambda repo, stage: commands)
    with pytest.raises(ValueError, match="maxrl.train"):
        cluster.launch("screen", methods="fresh")
    assert cluster.events == ["preflight"]
    assert not (env.root / "matrix/methods.json").exists()


@pytest.mark.parametrize("interruption", ["before-spawn", "queued-without-receipt", "lost-receipt", "while-waiting"])
def test_coordinator_redelivery_recovers_its_own_forced_trial(cluster, monkeypatch, interruption):
    env = cluster.env
    app = env.app
    cluster.coordinator_id = "fc-original-coordinator"
    cluster.live.add(cluster.coordinator_id)
    line, *_ = cli_lines(env, "screen")
    trial = "screen/fresh-0"

    def dispatch():
        return app.dispatch_matrix("screen", [line], RATES, PROVENANCE, None, trial)

    spawn = app.run_matrix_trial.spawn

    def interrupted_spawn(*args):
        if interruption == "before-spawn":
            raise KeyboardInterrupt
        call = spawn(*args)
        cluster.live.add(call.object_id)
        if interruption == "queued-without-receipt":
            raise KeyboardInterrupt
        if interruption == "lost-receipt":
            # The GPU worker started and claimed this attempt before its parent
            # could persist the returned call ID.
            marker = env.root / args[-1] / "worker.json"
            marker.write_text(json.dumps({"function_call_id": call.object_id}))
            raise KeyboardInterrupt
        real_get = call.get

        def interrupted_wait():
            call.get = real_get
            raise KeyboardInterrupt

        call.get = interrupted_wait
        return call

    with monkeypatch.context() as before_preemption:
        before_preemption.setattr(app.run_matrix_trial, "spawn", interrupted_spawn)
        with pytest.raises(KeyboardInterrupt):
            dispatch()

    attempt, = env.root.glob("matrix-*")
    original = json.loads((attempt / "dispatch.json").read_text())
    assert original["coordinator_call_id"] == cluster.coordinator_id
    if interruption == "queued-without-receipt":
        queued_call, = cluster.calls.values()

        def duplicate_spawn(*args):
            # The original worker wins the claim after the second call was queued.
            assert args[-1] == attempt.name
            call = spawn(*args)
            call.get = lambda: {"existing_worker_call_id": queued_call.object_id}
            return call

        monkeypatch.setattr(app.run_matrix_trial, "spawn", duplicate_spawn)
    assert not dispatch()
    owner = json.loads((attempt / "attempt.json").read_text())
    assert owner["status"] == "completed" and owner["attempt_id"] == attempt.name
    if interruption != "before-spawn":
        assert owner["function_call_id"] in cluster.reattached
    # Modal may redeliver after completion as well. A single --rerun invocation
    # always owns one attempt, including when its returning coordinator dies.
    assert not dispatch()
    assert list(env.root.glob("matrix-*")) == [attempt]
    submitted_calls = 2 if interruption == "queued-without-receipt" else 1
    assert len(cluster.commands) == 1 and cluster.launched == [line] * submitted_calls
