"""Exercise the Countdown launcher's job and storage contracts without paying for a worker."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest


REPO = Path(__file__).resolve().parents[1]
PROVENANCE = {"git_commit": "1234abcd" * 5, "git_dirty": False}


def import_launcher(with_modal=False):
    script = REPO / "scripts/modal_countdown.py"
    spec = importlib.util.spec_from_file_location(script.stem, script)
    module = importlib.util.module_from_spec(spec)
    modules = {spec.name: module} if with_modal else {spec.name: module, "modal": None}
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


@pytest.fixture
def launcher():
    return import_launcher()


@pytest.fixture
def rates():
    # The previous L4 rate is deliberately distinct from the A10 rate.
    return {"gpu_hour_cost_l4": "3.75", "gpu_hour_cost_a10g": "4.50",
            "cpu_hour_cost": "0.11", "mem_gib_hour_cost": "0.02"}


@pytest.mark.parametrize("device", ["NVIDIA A10", "NVIDIA A10G"])
def test_price_includes_assigned_gpu_requested_cpu_and_memory(launcher, rates, device):
    assert launcher.gpu_type(device) == launcher.GPU == "A10"
    expected = 4.5 + launcher.CPU * 0.11 + launcher.MEMORY_MIB / 1024 * 0.02
    assert launcher.hourly_rate(rates) == pytest.approx(expected)


@pytest.mark.parametrize("changes", ["", "? untracked.txt\n", "1 M. staged change\n", "1 .M unstaged change\n"])
def test_git_provenance_includes_all_local_changes(launcher, monkeypatch, changes):
    def git(cmd, text):
        assert cmd == ["git", "-C", str(REPO), "status", "--porcelain=v2", "--branch", "--untracked-files=normal"]
        return f"# branch.oid {PROVENANCE['git_commit']}\n# branch.head main\n# branch.ab +1 -0\n{changes}"

    monkeypatch.setattr(launcher.subprocess, "check_output", git)
    assert launcher.git_provenance(REPO) == {**PROVENANCE, "git_dirty": bool(changes)}


@pytest.mark.parametrize("stage,module", [("initial-eval", "maxrl.evaluate"), ("pilot", "maxrl.train")])
def test_controlled_command_uses_documented_settings(launcher, tmp_path, stage, module):
    manifest, out = tmp_path / "manifest.json", tmp_path / "output"
    commands = launcher.job_commands(stage, REPO, manifest, out, 2.5)
    first, *rest = [[str(item) for item in command] for command in commands]
    assert first[1:3] == ["-m", module]
    assert first[0] == str(REPO / ".venv/bin/python")
    flags = dict(zip(first[3::2], first[4::2], strict=True))
    assert flags.pop("--panel", "validation") == "validation"
    assert (REPO / flags.pop("--config")).resolve() == REPO / "configs/pilot.json"
    assert flags == {"--manifest": str(manifest), "--out": str(out), "--device": "cuda",
                     "--dtype": "float32", "--generation-batch": "8", "--score-batch": "1",
                     "--hourly-usd": "2.5"}
    if stage == "initial-eval":
        assert rest == []


def test_midpoint_uses_completed_pilot_and_documented_order(launcher, tmp_path):
    manifest, out, pilot = (tmp_path / name for name in ("manifest.json", "midpoint", "pilot"))
    commands = [[str(value) for value in command] for command in
                launcher.job_commands("midpoint", REPO, manifest, out, 2.5, pilot_run=pilot)]
    assert [command[1:4] for command in commands] == [
        ["-m", "maxrl.diagnostics", stage] for stage in ("collect", "score", "analyze", "full")
    ]
    collect, score, analyze, full = [dict(zip(cmd[4::2], cmd[5::2], strict=True)) for cmd in commands]
    assert (collect["--run"], collect["--manifest"], collect["--out"]) == (str(pilot), str(manifest), str(out))
    assert (collect["--panel"], collect["--step"], collect["--microbatch"]) == ("midpoint", "16", "8")
    for options in (collect, score, full):
        assert (options["--device"], options["--dtype"], options["--hourly-usd"]) == ("cuda", "float32", "2.5")
    assert score["--pools"] == analyze["--pools"] == full["--pools"] == str(out)
    assert analyze["--group-size"] == full["--group-size"] == "8"


@pytest.mark.parametrize("cards", [{"gpu_type": "L4", "gpu_name": "NVIDIA L4"}, {},
                                   {"gpu_type": "A10", "gpu_name": "NVIDIA L4"}])
def test_midpoint_refuses_unmatched_pilot_before_loading_models(launcher, rates, tmp_path, monkeypatch, cards):
    pilot = tmp_path / "saved-pilot"
    pilot.mkdir()
    (pilot / "attempt.json").write_text(json.dumps({"status": "completed", "stage": "pilot", **cards}))
    monkeypatch.setattr(launcher, "_gpu_name", lambda: "NVIDIA A10")
    monkeypatch.setattr(launcher, "_run", lambda *a, **k: pytest.fail("unexpected subprocess"))
    record = launcher.run_attempt(REPO, tmp_path, lambda: None, stage="midpoint",
                                  rates=rates, pilot_attempt=pilot.name, provenance=PROVENANCE)
    assert record["status"] == "failed" and "must match pilot" in record["error"]
    assert not (tmp_path / record["attempt_id"] / "output").exists()


def test_midpoint_recognizes_pilot_gpu_name_alias(launcher, tmp_path):
    pilot = tmp_path / "saved-pilot"
    pilot.mkdir()
    (pilot / "attempt.json").write_text(json.dumps({"stage": "pilot", "status": "completed",
                                                   "gpu_name": "NVIDIA A10G", "gpu_type": "A10"}))
    assert launcher._pilot_input(tmp_path, pilot.name, "A10") == pilot / "output"


def test_replay_hashes_saved_batches_and_run_inputs_before_scoring(launcher, tmp_path):
    folder = tmp_path / "source-pilot"
    output = folder / "output"
    (output / "responses").mkdir(parents=True)
    (output / "run.json").write_text(json.dumps({"manifest_sha256": launcher.MANIFEST_SHA256}))
    inputs = ("run.json", "metrics.jsonl", "responses/train-000001.jsonl", "responses/train-000002.jsonl")
    for name in inputs[1:]:
        (output / name).write_text(name)
    record = {"attempt_id": folder.name, "stage": "pilot", "status": "completed", "gpu_name": "NVIDIA A10",
              "manifest_sha256": launcher.MANIFEST_SHA256,
              "sha256": {f"output/{name}": launcher._digest(output / name) for name in inputs}}
    (folder / "attempt.json").write_text(json.dumps(record))
    pilot = launcher._pilot_input(tmp_path, folder.name, "A10", "replay")
    out = tmp_path / "scratch"
    assert launcher.job_commands("replay", REPO, tmp_path / "manifest.json", out, 2.5, pilot) == [[
        str(REPO / ".venv/bin/python"), "-m", "maxrl.replay", "--pilot", str(output), "--out", str(out),
        "--device", "cuda", "--hourly-usd", "2.5",
    ]]
    for name in inputs:
        original = (output / name).read_bytes()
        (output / name).write_bytes(original + b" ")
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            launcher._pilot_input(tmp_path, folder.name, "A10", "replay")
        (output / name).write_bytes(original)


def write_data_fixture(folder):
    folder.mkdir(parents=True)
    shard = folder / "validation.jsonl"
    shard.write_text('{"puzzle_id":"1-2-3=6","numbers":[1,2,3],"target":6}\n')
    manifest = folder / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "split_seed": 623, "panels": {
        "validation": {"file": shard.name, "count": 1, "sha256": hashlib.sha256(shard.read_bytes()).hexdigest()}
    }}))
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


@pytest.mark.parametrize("changed", ["manifest.json", "validation.jsonl"])
def test_prepared_data_detects_changed_bytes(launcher, tmp_path, monkeypatch, changed):
    folder = tmp_path / "data"
    digest = write_data_fixture(folder)
    # Seed 623, documented in docs/data.md.
    assert launcher.MANIFEST_SHA256 == "abdb27fbeeefdae2c9f6cce441811af401e07937f9b6bd4693f8a5cabb556186"
    monkeypatch.setattr(launcher, "MANIFEST_SHA256", digest)
    assert launcher.verify_manifest(folder / "manifest.json") == digest
    with (folder / changed).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError):
        launcher.verify_manifest(folder / "manifest.json")


def test_preflight_prepares_once_and_rechecks_cache(launcher, tmp_path, monkeypatch):
    prepared = write_data_fixture(tmp_path / "reference")
    monkeypatch.setattr(launcher, "MANIFEST_SHA256", prepared)
    root = tmp_path / "volume"
    root.mkdir()
    commands = []

    def run(argv, repo, emit, **kwargs):
        cmd = list(map(str, argv))
        commands.append(cmd)
        if "maxrl.data" in cmd:
            assert cmd[cmd.index("--seed") + 1] == "623"
            write_data_fixture(Path(cmd[cmd.index("--out") + 1]))

    monkeypatch.setattr(launcher, "_run", run)
    results = [launcher.run_attempt(REPO, root, lambda: None, stage="preflight", provenance=PROVENANCE)
               for _ in range(2)]
    assert [result["status"] for result in results] == ["completed", "completed"]
    for result in results:
        persisted = json.loads((root / result["attempt_id"] / "attempt.json").read_text())
        assert persisted["git_commit"] == PROVENANCE["git_commit"] and persisted["git_dirty"] is False
    assert sum("maxrl.data" in cmd for cmd in commands) == 1
    assert sum(any("pytest.main" in arg for arg in cmd) for cmd in commands) == 2
    assert not any("maxrl.evaluate" in cmd or "maxrl.train" in cmd for cmd in commands)
    (root / "data/validation.jsonl").write_bytes(b"bad shard")
    assert launcher.run_attempt(REPO, root, lambda: None, stage="preflight", provenance=PROVENANCE)["status"] == "failed"


def test_failed_gpu_attempts_persist_before_retry(launcher, rates, tmp_path, monkeypatch):
    root = tmp_path / "volume"
    monkeypatch.setattr(launcher, "MANIFEST_SHA256", write_data_fixture(root / "data"))
    snapshots = []

    def commit():
        snapshots.append({p.parent.name: json.loads(p.read_text())["status"]
                          for p in root.rglob("attempt.json")})

    commands = []
    devices = iter(["NVIDIA A10", "NVIDIA A10G"])
    monkeypatch.setattr(launcher, "_gpu_name", lambda: next(devices))

    def run(argv, repo, emit, **kwargs):
        cmd = list(map(str, argv))
        commands.append(cmd)
        if len(commands) == 1:
            raise RuntimeError("CUDA unavailable")
        if "maxrl.evaluate" in cmd:
            assert float(cmd[cmd.index("--hourly-usd") + 1]) == launcher.hourly_rate(rates)
            output = Path(cmd[cmd.index("--out") + 1])
            assert not output.is_relative_to(root)
            output.mkdir()  # evaluator requires a previously nonexistent directory
            (output / "run.json").write_text("partial output")
            raise RuntimeError("model failed")

    monkeypatch.setattr(launcher, "_run", run)
    dirty = {**PROVENANCE, "git_dirty": True}
    first = launcher.run_attempt(REPO, root, commit, stage="initial-eval", rates=rates, provenance=dirty)
    assert "CUDA unavailable" in first["error"]
    assert not any("maxrl.evaluate" in cmd for cmd in commands)
    second = launcher.run_attempt(REPO, root, commit, stage="initial-eval", rates=rates, provenance=dirty)
    assert "model failed" in second["error"]
    assert first["attempt_id"] != second["attempt_id"]
    for record, device in ((first, "NVIDIA A10"), (second, "NVIDIA A10G")):
        folder = root / record["attempt_id"]
        assert (folder / "run.log").is_file()
        assert json.loads((folder / "attempt.json").read_text()) == record
        assert snapshots[-1][folder.name] == record["status"] == "failed"
        assert (record["gpu_name"], record["gpu_type"]) == (device, "A10")
        assert record["git_commit"] == dirty["git_commit"] and record["git_dirty"] is True
        hourly_usd = launcher.hourly_rate(rates)
        assert record["hourly_usd"] == pytest.approx(hourly_usd)
        assert record["estimated_cost_usd"] == pytest.approx(record["wall_seconds"] * hourly_usd / 3600)
    assert (root / second["attempt_id"] / "output/run.json").read_text() == "partial output"


@pytest.mark.parametrize("device,omit_price", [("NVIDIA L4", False), ("NVIDIA H100", False), ("NVIDIA A10G", True)])
def test_unknown_device_or_price_is_saved_before_generation(launcher, rates, tmp_path, monkeypatch, device, omit_price):
    root = tmp_path / "volume"
    monkeypatch.setattr(launcher, "MANIFEST_SHA256", write_data_fixture(root / "data"))
    monkeypatch.setattr(launcher, "_gpu_name", lambda: device)
    if omit_price:
        del rates["gpu_hour_cost_a10g"]
    commands = []
    monkeypatch.setattr(launcher, "_run", lambda argv, *args, **kwargs: commands.append(argv))
    commits = []
    record = launcher.run_attempt(REPO, root, lambda: commits.append(True), stage="initial-eval",
                                  rates=rates, provenance=PROVENANCE)
    assert record["status"] == "failed" and record["error"]
    assert record["hourly_usd"] is None
    assert "estimated_cost_usd" not in record
    assert not any("maxrl.evaluate" in cmd for cmd in commands)
    assert json.loads((root / record["attempt_id"] / "attempt.json").read_text()) == record
    assert len(commits) >= 2


class MemoryVolume:
    def __init__(self, content):
        self.content = content

    def read_file(self, path):
        yield self.content[str(path).lstrip("/")]


def test_replay_publishes_only_rates_diagnostics_and_provenance(launcher, tmp_path):
    results = {"rates.csv": b"rate,prediction\n", "heldout.jsonl": b"{}\n", "replay.json": b"{}",
               "resources.csv": b"phase,seconds\n"}
    files = {f"output/{name}": payload for name, payload in results.items()}
    files["run.log"] = b"completed replay"
    inventory = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    inventory["output/checkpoints/model.safetensors"] = "unavailable on laptop"
    attempt = "replay-123"
    record = {"attempt_id": attempt, "stage": "replay", "status": "completed", "sha256": inventory}
    files["attempt.json"] = json.dumps(record).encode()
    volume = MemoryVolume({f"{attempt}/{name}": data for name, data in files.items()})
    downloaded = launcher.download_attempt(record, tmp_path, volume)
    assert not (downloaded / "output/checkpoints").exists()
    published = tmp_path / "results/midway/replay"
    assert {p.name: p.read_bytes() for p in published.iterdir()} == {**results, "attempt.json": files["attempt.json"]}


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_download_small_attempt_files_and_publish_only_summary(launcher, tmp_path, status):
    attempt = "initial-eval-20261118T150000Z-31d4989b"
    files = {"run.log": b"evaluation log", "output/metrics.json": b'{"responses": 1024}\n',
             "output/resources.csv": b"phase,seconds\ntotal,3\n",
             "output/responses.jsonl": b"full response shard\n"}
    record = {"attempt_id": attempt, "stage": "initial-eval", "status": status,
              "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    files["attempt.json"] = (json.dumps(record) + "\n").encode()
    volume = MemoryVolume({f"{attempt}/{name}": data for name, data in files.items()})
    # Leave a previous complete download intact when recovering with the selective downloader.
    volume.content.pop(f"{attempt}/output/responses.jsonl")
    full = launcher.download_attempt(record, tmp_path, volume)
    small = {name: data for name, data in files.items() if name != "output/responses.jsonl"}
    assert {str(p.relative_to(full)): p.read_bytes() for p in full.rglob("*") if p.is_file()} == small
    legacy = full / "output/responses.jsonl"
    legacy.write_bytes(files["output/responses.jsonl"])
    launcher.download_attempt(record, tmp_path, volume)
    assert legacy.read_bytes() == files["output/responses.jsonl"]
    summary = tmp_path / "results/midway/initial-eval"
    if status == "completed":
        assert {p.name: p.read_bytes() for p in summary.iterdir()} == {
            name: files[f"output/{name}"] for name in ("metrics.json", "resources.csv")
        }
    else:
        assert not summary.exists()


def test_bad_download_never_publishes_partial_files(launcher, tmp_path):
    record = {"attempt_id": "a", "stage": "initial-eval", "status": "completed",
              "sha256": {"output/metrics.json": hashlib.sha256(b"on volume").hexdigest()}}
    volume = MemoryVolume({"a/attempt.json": json.dumps(record).encode(), "a/output/metrics.json": b"truncated"})
    with pytest.raises(ValueError, match="SHA-256"):
        launcher.download_attempt(record, tmp_path, volume)
    assert not list((tmp_path / "runs/countdown").iterdir())
    assert not (tmp_path / "results").exists()


def test_postrun_verification_checks_groups_and_price(launcher, tmp_path):
    pytest.importorskip("torch")  # The repository's accuracy module imports torch.
    from maxrl.data import Puzzle, save_manifest
    from maxrl.schema import Response, write_responses

    puzzles = [Puzzle(f"1-2-3={i}", [1, 2, 3], i) for i in range(128)]
    save_manifest(tmp_path, {"validation": puzzles}, 128, 623, "fixture")
    rows = (Response(p.puzzle_id, [1], [2], [-0.1], [1], [1], j == 0, "initial",
                     f"{p.puzzle_id}/{j}", "validation") for p in puzzles for j in range(8))
    path = tmp_path / "responses.jsonl"
    write_responses(path, rows)
    metrics = {"prompts": 128, "responses": 1024, "generated_tokens": 1024, "pass@1": 0.125,
               "pass@8": 1, "zero_success_frequency": 0, "one_success_frequency": 1,
               "success_counts": {"1": 128}}
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    (tmp_path / "resources.csv").write_text("phase,seconds,responses,hourly_usd,machine_usd\ntotal,36,1024,2,0.02\n")
    manifest = tmp_path / "manifest.json"
    launcher.verify_initial_output(tmp_path, manifest, 2)
    with pytest.raises(ValueError, match="resource"):
        launcher.verify_initial_output(tmp_path, manifest, 3)
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:-1]))
    with pytest.raises(ValueError, match="eight responses"):
        launcher.verify_initial_output(tmp_path, manifest, 2)


@pytest.mark.parametrize("cpu_ok", [True, False], ids=["preflight-passes", "preflight-fails"])
def test_entrypoint_gates_single_gpu_job_on_preflight(cpu_ok, rates, monkeypatch, tmp_path):
    modal = pytest.importorskip("modal")
    requested_gpus = []
    register = modal.App.function

    def function(app, **options):
        if "gpu" in options:
            requested_gpus.append(options["gpu"])
        return register(app, **options)

    monkeypatch.setattr(modal.App, "function", function)
    app = import_launcher(with_modal=True)
    assert set(app.app.registered_entrypoints) == {"main"}
    assert requested_gpus == ["A10"]
    events = []

    def provenance(repo):
        assert repo == REPO
        events.append("git")
        return PROVENANCE

    def preflight(source):
        assert source == PROVENANCE
        events.append("cpu")
        return {"stage": "preflight", "status": "completed" if cpu_ok else "failed", "error": "failed CPU checks"}

    def download(record, repo, volume):
        events.append(f"download {record['stage']}")
        return tmp_path

    def billing(cmd, text):
        assert cmd == ["modal", "billing", "rates", "--json"]
        events.append("billing")
        return json.dumps(rates)

    def gpu(stage, price_list, pilot, source):
        assert (stage, price_list, pilot) == ("initial-eval", rates, None)
        assert source == PROVENANCE
        events.append("gpu")
        return {"stage": stage, "status": "completed"}

    monkeypatch.setattr(app.preflight, "remote", preflight)
    monkeypatch.setattr(app.run_gpu, "remote", gpu)
    monkeypatch.setattr(app.subprocess, "check_output", billing)
    monkeypatch.setattr(app, "git_provenance", provenance)
    monkeypatch.setattr(app, "download_attempt", download)
    if cpu_ok:
        app.main(stage="initial-eval")
        assert events == ["git", "cpu", "download preflight", "billing", "gpu", "download initial-eval"]
    else:
        with pytest.raises(RuntimeError, match="failed CPU checks"):
            app.main(stage="initial-eval")
        assert events == ["git", "cpu", "download preflight"]
