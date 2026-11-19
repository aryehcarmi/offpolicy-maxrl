"""Run the controlled Countdown evaluation, pilot, replay or midpoint protocol on Modal."""
from collections import defaultdict
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from uuid import uuid4


MANIFEST_SHA256 = "abdb27fbeeefdae2c9f6cce441811af401e07937f9b6bd4693f8a5cabb556186"
PILOT_DEPENDENT = ("replay", "midpoint", "saved-verify", "saved-train", "saved-validation")
STAGES = ("preflight", "initial-eval", "pilot", *PILOT_DEPENDENT)
TEST_SDK = "/opt/countdown-test-sdk"
PILOT_INPUTS = ("run.json", "metrics.jsonl", "resources.csv")
BUDGET_CONFIGS = ("pilot.json", "screen.json", "tune.json", "main.json")
REPLAY_FILES = ("rates.csv", "heldout.jsonl", "replay.json", "resources.csv")
REPLAY_OUTPUTS = {"original": REPLAY_FILES, "low": (*REPLAY_FILES, "successes.jsonl")}
SELECTED_RATE = 1.5e-7
SAVED_OUTPUTS = {
    "saved-verify": ("verification.json", "resources.csv"),
    "saved-train": ("trajectories.jsonl", "summary.json", "resources.csv", "pilot/run.json",
                    "probes/first-token.jsonl", "probes/parameter-types.json",
                    "probes/first-token-summary.json", "probes/resources.csv"),
    "saved-validation": ("run.json", "metrics.json", "resources.csv"),
}
MIDPOINT_FILES = ("manifest.json", "resources.csv", "scores/provenance.json", "scores/resources.csv",
                  "analysis-N8/summary.json", "analysis-N8/exposure.csv",
                  "full-initial-N8/summary.json", "full-initial-N8/resources.csv")


def _check_grid(stage, grid):
    if grid not in REPLAY_OUTPUTS or (stage != "replay" and grid != "original"):
        raise ValueError("Use --grid original or --grid low with the replay stage")


def _stage_files(stage, grid):
    if stage == "replay":
        return REPLAY_OUTPUTS[grid]
    if stage == "midpoint":
        return MIDPOINT_FILES
    return SAVED_OUTPUTS.get(stage, ())


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_manifest(manifest: Path) -> str:
    """Check the seed-623 manifest pinned in docs/data.md and every shard it identifies."""
    fingerprint = _digest(manifest)
    if fingerprint != MANIFEST_SHA256:
        raise ValueError(f"Unexpected manifest SHA-256 at {manifest}: {fingerprint}")
    for panel in json.loads(manifest.read_text())["panels"].values():
        shard = manifest.parent / panel["file"]
        if _digest(shard) != panel["sha256"]:
            raise ValueError(f"Shard SHA-256 mismatch: {shard}")
    return fingerprint


def verify_initial_output(output: Path, manifest: Path, hourly_usd: float) -> None:
    """Recompute the initial evaluation from the saved, schema-validated responses."""
    from maxrl.data import load_panel
    from maxrl.metrics import accuracy
    from maxrl.schema import read_responses

    expected = {p.puzzle_id for p in load_panel(manifest, "validation")}
    groups = defaultdict(list)
    for row in read_responses(output / "responses.jsonl"):
        groups[row.puzzle_id].append(row)
    if len(expected) != 128 or groups.keys() != expected or any(len(rows) != 8 for rows in groups.values()):
        raise ValueError("Expected eight responses for each of the 128 validation puzzles")
    metrics = json.loads((output / "metrics.json").read_text())
    actual = accuracy(list(groups.values()))
    actual["success_counts"] = {str(k): n for k, n in actual["success_counts"].items()}
    if any(metrics[k] != value for k, value in actual.items()):
        raise ValueError("Saved metrics differ from the response shard")
    with (output / "resources.csv").open(newline="") as stream:
        total, = [r for r in csv.DictReader(stream) if r["phase"] == "total"]
    seconds = float(total["seconds"])
    if (int(total["responses"]) != actual["responses"] or float(total["hourly_usd"]) != hourly_usd
            or not 0 < seconds < math.inf
            or not math.isclose(float(total["machine_usd"]), seconds * hourly_usd / 3600)):
        raise ValueError("Total resource row differs from the response count or requested machine rate")
    print(f"Verified {len(groups)} validation puzzles, {actual['responses']} responses and resource total")


def verify_pilot_checkpoints(output, saved, inventory, *, completed, prefix="output"):
    """Compare the worker's checkpoint bytes with the SHA-256 inventory read on the Volume."""
    checkpoints = output / "checkpoints"
    for path in checkpoints.rglob("*"):
        if path.is_file():
            key = str(Path(prefix) / path.relative_to(output))
            if inventory.get(key) != _digest(path):
                raise ValueError(f"Checkpoint SHA-256 mismatch: {key}")

    if not completed:
        return []
    run = json.loads((saved / "run.json").read_text())
    config = run["config"]
    steps = sorted({*config["checkpoint_steps"], config["updates"]})
    verified = []
    for step in steps:
        folder = saved / "checkpoints" / f"step-{step:06d}"
        marker = json.loads((folder / "maxrl_checkpoint.json").read_text())
        identity = f"{run['run_id']}/{folder.name}"
        if marker["checkpoint_id"] != identity or marker["prompt"] != run["prompt"]:
            raise ValueError(f"Unexpected checkpoint identity at {folder}")
        # save_pretrained writes one weight file for the pinned 0.6B model.
        for name in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"):
            if (folder / name).stat().st_size == 0:
                raise ValueError(f"Empty checkpoint file: {folder / name}")
        verified.append(identity)
    return verified


def job_commands(stage, repo, manifest, output, hourly_usd, pilot_run=None, grid="original", initial_eval=None):
    """Return exactly the documented commands for one GPU job; no shell or stage chaining."""
    if hourly_usd is None or not math.isfinite(hourly_usd) or hourly_usd <= 0:
        raise ValueError("A positive, finite total hourly machine rate is required")
    python = str(repo / ".venv/bin/python")
    device = ["--device", "cuda", "--dtype", "float32"]
    price = ["--hourly-usd", str(hourly_usd)]
    if stage in ("initial-eval", "pilot"):
        module = "maxrl.evaluate" if stage == "initial-eval" else "maxrl.train"
        return [[python, "-m", module, "--config", "configs/pilot.json", "--manifest", str(manifest),
                 "--out", str(output), *device, "--generation-batch", "8", "--score-batch", "1", *price]]
    if stage not in PILOT_DEPENDENT or pilot_run is None:
        raise ValueError(f"Expected a GPU stage with its required pilot input, got {stage!r}")
    if stage == "saved-validation":
        return [[python, "-m", "maxrl.evaluate", "--config", "configs/pilot.json", "--manifest", str(manifest),
                 "--checkpoint", str(pilot_run / "checkpoints/step-000016"), "--seed", "18",
                 "--method", "count", "--lr", str(SELECTED_RATE), "--panel", "validation", "--out", str(output),
                 *device, "--generation-batch", "8", "--score-batch", "1", *price]]
    if stage in ("saved-verify", "saved-train"):
        mode = "verify" if stage == "saved-verify" else "run"
        inputs = ["--pilot", str(pilot_run), "--device", "cuda", *price]
        command = [python, "-m", "maxrl.saved_trajectory", mode, *inputs, "--out", str(output)]
        if mode == "verify":
            return [command]
        if initial_eval is None:
            raise ValueError("Saved-batch training requires the initial validation pool")
        return [[*command, "--initial-eval", str(initial_eval), "--manifest", str(manifest)],
                [python, "-m", "maxrl.trajectory_probes", *inputs, "--out", str(output / "probes")]]
    if stage == "replay":
        grid_args = ["--grid", grid] if grid != "original" else []
        return [[python, "-m", "maxrl.replay", "--pilot", str(pilot_run), "--out", str(output),
                 "--device", "cuda", *price, *grid_args]]
    diag = [python, "-m", "maxrl.diagnostics"]
    pools = ["--pools", str(output)]
    return [
        [*diag, "collect", "--panel", "midpoint", "--step", "16", "--run", str(pilot_run),
         "--manifest", str(manifest), "--out", str(output), *device, "--microbatch", "8", *price],
        [*diag, "score", *pools, "--run", str(pilot_run), *device, *price],
        [*diag, "analyze", *pools, "--group-size", "8"],
        [*diag, "full", *pools, "--run", str(pilot_run), "--group-size", "8", *device, *price],
    ]


def _run(argv, repo, emit, *, offline=False):
    """Stream the locked interpreter's output into the attempt log and Modal's log."""
    argv = list(map(str, argv))
    emit("$ " + shlex.join(argv))
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    # Modal injects a vendored SDK into the worker. Keep it out of the frozen interpreter.
    env.pop("PYTHONPATH", None)
    if offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with subprocess.Popen(argv, cwd=repo, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, errors="replace") as child:
        try:
            with child.stdout as stream:
                for line in stream:
                    emit(line.rstrip("\n"))
            if child.wait():
                raise subprocess.CalledProcessError(child.returncode, argv)
        finally:
            if child.poll() is None:
                child.kill()


@contextmanager
def _sample_gpu(path, emit, interval=30):
    """Sample the single assigned card while the experiment subprocess runs."""
    stop = threading.Event()

    def sample():
        try:
            with path.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["timestamp_utc", "gpu_utilization_percent", "memory_used_mib", "memory_total_mib"])
                while True:
                    output = subprocess.check_output([
                        "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ], text=True, timeout=5, stderr=subprocess.STDOUT)
                    utilization, used, total = map(int, output.strip().split(","))
                    writer.writerow([_utc(), utilization, used, total])
                    stream.flush()
                    if stop.wait(interval):
                        break
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            emit(f"GPU sampling stopped: {error}")

    worker = threading.Thread(target=sample, name="gpu-sampler", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()


def _preflight(repo, volume_root, scratch, emit, commit):
    python = repo / ".venv/bin/python"
    _run([python, "-c", "import torch; print(torch.__version__, torch.version.cuda); "
          "assert torch.version.cuda and not torch.cuda.is_available(), 'expected CUDA wheel on CPU worker'"],
         repo, emit)
    data = volume_root / "data"
    if not data.exists():
        prepared = scratch / "data"
        _run([python, "-m", "maxrl.data", "--out", prepared, "--seed", "623"], repo, emit)
        verify_manifest(prepared / "manifest.json")
        # Publish as a complete directory even if the container is interrupted during the copy.
        with tempfile.TemporaryDirectory(dir=volume_root) as staging:
            staged = Path(staging) / "data"
            shutil.copytree(prepared, staged)
            staged.rename(data)
        commit()
    digest = verify_manifest(data / "manifest.json")
    emit(f"Verified manifest {digest} and all referenced shards")
    # The SDK tests read distribution metadata. Append the installed CLI after the locked packages.
    suite = (f"import site; site.addsitedir({TEST_SDK!r}); "
             "import pytest; raise SystemExit(pytest.main(['-q']))")
    _run([python, "-c", suite], repo, emit, offline=True)
    return digest


def _attempt_input(root, name, allowed, gpu, stage):
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"A {stage} job requires a valid source attempt directory name")
    folder = root / name
    record = json.loads((folder / "attempt.json").read_text())
    if record["stage"] not in allowed or record["status"] != "completed":
        raise ValueError(f"Invalid or incomplete source attempt {name} for {stage}")
    cards = {gpu_type(record[key]) for key in ("gpu_type", "gpu_name") if record.get(key)}
    if cards != {gpu}:
        raise ValueError(f"{stage} GPU {gpu} must match pilot or source {name} (recorded types {sorted(cards)})")
    return folder, record


def _check_source_record(folder, record):
    if record["attempt_id"] != folder.name or record["manifest_sha256"] != MANIFEST_SHA256:
        raise ValueError("Source attempt or data manifest identity mismatch")


def _check_files(folder, record, names):
    for name in names:
        key = f"output/{name}"
        if record["sha256"].get(key) != _digest(folder / key):
            raise ValueError(f"Source SHA-256 mismatch: {key}")


def _pilot_input(volume_root, pilot_attempt, assigned_gpu, stage="midpoint", grid="original"):
    allowed = {"saved-train"} if stage == "saved-validation" else {"pilot"}
    if stage == "midpoint":
        allowed.add("saved-train")
    folder, record = _attempt_input(volume_root, pilot_attempt, allowed, assigned_gpu, stage)
    output = folder / "output"
    if record["stage"] == "saved-train":
        _check_source_record(folder, record)
        pilot = output / "pilot"
        checkpoint = pilot / "checkpoints/step-000016"
        names = [key.removeprefix("output/") for key in record["sha256"]
                 if key.startswith("output/pilot/checkpoints/step-000016/")]
        for required in ("model.safetensors", "config.json", "tokenizer.json",
                         "tokenizer_config.json", "maxrl_checkpoint.json"):
            if f"pilot/checkpoints/step-000016/{required}" not in names:
                raise ValueError(f"Missing saved checkpoint file: {required}")
        _check_files(folder, record, ["pilot/run.json", *names])
        run = json.loads((pilot / "run.json").read_text())
        marker = json.loads((checkpoint / "maxrl_checkpoint.json").read_text())
        identity = f"{run['run_id']}/step-000016"
        config = run["config"]
        if (record.get("verified_checkpoints") != [identity] or marker["checkpoint_id"] != identity
                or marker["prompt"] != run["prompt"] or run["manifest_sha256"] != MANIFEST_SHA256
                or (config["method"], config["lr"], config["updates"], config["seed"]) != ("count", SELECTED_RATE, 16, 17)):
            raise ValueError("Saved checkpoint identity or selected Count settings do not match")
        return pilot
    if stage == "midpoint":
        return output
    _check_source_record(folder, record)
    last_step = 2 if stage == "replay" and grid == "original" else 16
    names = ["run.json", "metrics.jsonl", *(f"responses/train-{i:06d}.jsonl" for i in range(1, last_step + 1))]
    if stage in ("saved-verify", "saved-train"):
        names.extend(f"responses/weights-{i:06d}.csv" for i in range(1, 17))
    if stage == "saved-train":
        names.append("responses/validation-000000.jsonl")
    _check_files(folder, record, names)
    run = json.loads((output / "run.json").read_text())
    if run["manifest_sha256"] != MANIFEST_SHA256:
        raise ValueError("Source run and attempt disagree on data manifest")
    return output


def _training_sources(volume_root, verification_attempt, initial_attempt, pilot_run, gpu):
    folder, record = _attempt_input(volume_root, verification_attempt, {"saved-verify"}, gpu, "saved-train")
    _check_source_record(folder, record)
    _check_files(folder, record, ["verification.json"])
    check = json.loads((folder / "output/verification.json").read_text())
    pilot_record = json.loads((pilot_run.parent / "attempt.json").read_text())
    pilot = json.loads((pilot_run / "run.json").read_text())
    checked = ["run.json", "metrics.jsonl", *(f"responses/train-{i:06d}.jsonl" for i in range(1, 17)),
               *(f"responses/weights-{i:06d}.csv" for i in range(1, 17))]
    if (record["pilot_attempt"] != pilot_run.parent.name or check["passed"] is not True
            or check["pilot_run_id"] != pilot["run_id"] or check["initial_checkpoint_id"] != pilot["initial_checkpoint_id"]
            or any(check["source_sha256"].get(n) != pilot_record["sha256"][f"output/{n}"] for n in checked)):
        raise ValueError("Saved-batch verification must match this exact pilot and all 16 batches")
    initial, initial_record = _attempt_input(volume_root, initial_attempt, {"initial-eval"}, gpu, "saved-train")
    _check_source_record(initial, initial_record)
    _check_files(initial, initial_record, ["run.json", "responses.jsonl"])
    meta = json.loads((initial / "output/run.json").read_text())
    if (meta["panel"] != "validation" or meta["checkpoint_id"] != pilot["initial_checkpoint_id"]
            or meta["prompt"] != pilot["prompt"] or meta["manifest_sha256"] != MANIFEST_SHA256):
        raise ValueError("Initial evaluation pool has different model, prompt or validation panel")
    return initial / "output"


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _record(path, record):
    staged = path.with_suffix(".tmp")
    staged.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    staged.replace(path)


def git_provenance(repo):
    """Read the launch checkout once, including staged and untracked changes."""
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain=v2", "--branch", "--untracked-files=normal"],
        text=True,
    ).splitlines()
    prefix = "# branch.oid "
    commit = next((line[len(prefix):] for line in status if line.startswith(prefix)), None)
    if not commit or commit == "(initial)":
        raise ValueError(f"No Git commit in the launch checkout: {repo}")
    return {"git_commit": commit, "git_dirty": any(not line.startswith("# ") for line in status)}


def run_attempt(repo: Path, volume_root: Path, commit, *, stage: str, provenance: dict,
                rates: dict | None = None, pilot_attempt: str | None = None, grid: str = "original",
                verification_attempt=None, initial_attempt=None) -> dict:
    """Keep writable experiment files on local disk and preserve one Volume folder per invocation."""
    if stage not in STAGES or (stage not in PILOT_DEPENDENT and pilot_attempt is not None):
        raise ValueError(f"Invalid stage or pilot input for {stage!r}")
    _check_grid(stage, grid)
    repo, volume_root = Path(repo).resolve(), Path(volume_root).resolve()
    started = _utc()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt_id = f"{stage}-{stamp}-{uuid4().hex[:12]}"
    attempt = volume_root / attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    record = {"attempt_id": attempt_id, "stage": stage, "pilot_attempt": pilot_attempt,
              "git_commit": provenance["git_commit"], "git_dirty": provenance["git_dirty"],
              "status": "running", "started_utc": started, "gpu_name": None, "gpu_type": None,
              "hourly_usd": None, "sha256": {}}
    if stage == "replay":
        record["grid"] = grid
    if stage == "saved-train":
        record.update(verification_attempt=verification_attempt, initial_attempt=initial_attempt,
                      selected_method="count", learning_rate=SELECTED_RATE)
    if stage == "saved-validation":
        record["learning_rate"] = SELECTED_RATE
    record_path = attempt / "attempt.json"
    log = attempt / "run.log"
    start = time.monotonic()
    _record(record_path, record)
    commit()  # An uncatchable kill still leaves an identifiable attempt.
    with tempfile.TemporaryDirectory() as work, log.open("x", encoding="utf-8", buffering=1) as stream:
        output = Path(work) / "output"

        def emit(line):
            print(line, file=stream, flush=True)
            print(line, flush=True)

        try:
            emit(f"Starting {attempt_id}")
            if stage == "preflight":
                record["manifest_sha256"] = _preflight(repo, volume_root, Path(work), emit, commit)
            else:
                record["gpu_name"] = _gpu_name()
                record["gpu_type"] = gpu_type(record["gpu_name"])
                if record["gpu_type"] != GPU:
                    raise ValueError(f"Requested {GPU}, received {record['gpu_name']}")
                hourly_usd = hourly_rate(rates)
                record["hourly_usd"] = hourly_usd
                emit(f"Assigned {record['gpu_name']}, {CPU} CPU cores and {MEMORY_MIB} MiB "
                     f"at ${hourly_usd:.6f}/hour")
                _record(record_path, record)
                commit()
                pilot = (_pilot_input(volume_root, pilot_attempt, record["gpu_type"], stage, grid)
                         if stage in PILOT_DEPENDENT else None)
                initial_eval = None
                if stage == "saved-train":
                    initial_eval = _training_sources(volume_root, verification_attempt, initial_attempt,
                                                     pilot, record["gpu_type"])
                if stage == "midpoint" and pilot.name == "pilot":
                    record["source_learning_rate"] = SELECTED_RATE
                manifest = volume_root / "data/manifest.json"
                record["manifest_sha256"] = verify_manifest(manifest)
                commands = job_commands(stage, repo, manifest, output, hourly_usd, pilot, grid, initial_eval)
                _run([repo / ".venv/bin/python", "-c", "import torch; "
                      "assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'expected one CUDA GPU'; "
                      "print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"], repo, emit)
                with _sample_gpu(attempt / "gpu.csv", emit):
                    for command in commands:
                        _run(command, repo, emit)
                if stage in ("initial-eval", "saved-validation"):
                    _run([repo / ".venv/bin/python", "-c",
                          "from pathlib import Path; import sys; "
                          "from scripts.modal_countdown import verify_initial_output; "
                          "verify_initial_output(Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]))",
                          output, manifest, str(hourly_usd)], repo, emit)
                if not output.is_dir():
                    raise FileNotFoundError(f"Job exited without creating {output}")
                for name in _stage_files(stage, grid):
                    if not (output / name).stat().st_size:
                        raise ValueError(f"Empty {stage} result: {name}")
                if stage == "pilot":
                    (output / "configs").mkdir()
                    for name in BUDGET_CONFIGS:
                        shutil.copyfile(repo / "configs" / name, output / "configs" / name)
            record["status"] = "completed"
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            emit(record["error"])
            if not isinstance(error, Exception):
                raise
        finally:
            try:
                if output.exists():
                    shutil.copytree(output, attempt / "output")
                    if stage in ("pilot", "saved-train"):
                        commit()  # Verify the copied files on the Volume before releasing local scratch.
            except BaseException as error:
                previous = record.get("error", "")
                record.update(status="failed", error=f"{previous} Artifact copy failed: {error}".strip())
                emit(record["error"])
                if not isinstance(error, Exception):
                    raise
            finally:
                try:
                    for path in sorted(attempt.rglob("*")):
                        if path.is_file() and path not in (record_path, log):
                            record["sha256"][str(path.relative_to(attempt))] = _digest(path)
                    if stage == "pilot":
                        record["verified_checkpoints"] = verify_pilot_checkpoints(
                            output, attempt / "output", record["sha256"], completed=record["status"] == "completed"
                        )
                    if stage == "saved-train":
                        record["verified_checkpoints"] = verify_pilot_checkpoints(
                            output / "pilot", attempt / "output/pilot", record["sha256"],
                            completed=record["status"] == "completed", prefix="output/pilot",
                        )
                except (OSError, ValueError, KeyError) as error:
                    previous = record.get("error", "")
                    record.update(status="failed", error=f"{previous} Artifact verification failed: {error}".strip())
                    emit(record["error"])
                # Verification failures are part of the durable worker log.
                try:
                    stream.flush()
                    record["sha256"]["run.log"] = _digest(log)
                except OSError as error:
                    previous = record.get("error", "")
                    record.update(status="failed", error=f"{previous} Could not read run.log: {error}".strip())
                record["finished_utc"] = _utc()
                record["wall_seconds"] = time.monotonic() - start
                if record["hourly_usd"] is not None:
                    record["estimated_cost_usd"] = record["hourly_usd"] * record["wall_seconds"] / 3600
                _record(record_path, record)
                commit()
    return record


REMOTE_REPO = Path("/opt/offpolicy-maxrl")
VOLUME_ROOT = Path("/countdown")
VOLUME_NAME = "countdown-maxrl"
GPU = "A10"
CPU = 2
MEMORY_MIB = 16384


def build_image(repo):
    """Cache locked dependencies and pinned model files before mounting the current code."""
    repo = Path(repo)
    image = (
        modal.Image.from_registry("python:3.12.3-slim-bookworm")
        .pip_install("uv==0.13.4")
        .workdir(str(REMOTE_REPO))
        .add_local_file(repo / "pyproject.toml", str(REMOTE_REPO / "pyproject.toml"), copy=True)
        .add_local_file(repo / "uv.lock", str(REMOTE_REPO / "uv.lock"), copy=True)
        .run_commands("uv sync --frozen --all-extras --no-install-project --no-cache")
    )
    config = json.loads((repo / "configs/pilot.json").read_text())
    cache_model = (
        "from huggingface_hub import snapshot_download\n"
        f"snapshot_download({config['model']!r}, revision={config['revision']!r}, "
        "allow_patterns=['*.safetensors', '*.json', '*.txt', '*.jinja'])"
    )
    image = (
        image.dockerfile_commands("RUN " + json.dumps([str(REMOTE_REPO / ".venv/bin/python"), "-c", cache_model]))
        .env({"HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
        .add_local_dir(repo / "src/maxrl", str(REMOTE_REPO / "src/maxrl"), copy=True, ignore=["__pycache__"])
        .run_commands("uv sync --frozen --all-extras --no-cache")
        .run_commands(f"uv pip install --target {TEST_SDK} --no-cache modal==1.6.4")
    )
    for folder in ("configs", "scripts", "tests", "results/midway"):
        image = image.add_local_dir(repo / folder, str(REMOTE_REPO / folder), ignore=["__pycache__"])
    return image.add_local_file(repo / "docs/research-log.md", str(REMOTE_REPO / "docs/research-log.md"))


def _gpu_name():
    """Query the assigned GPU before starting Python model setup or generation."""
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip().splitlines()
    if len(names) != 1:
        raise ValueError(f"Expected one assigned GPU, got {names!r}")
    return names[0].strip()


def gpu_type(name):
    model = name.strip().upper().removeprefix("NVIDIA ")
    return "A10" if model == "A10G" else model


def hourly_rate(rates):
    """Use the current function rates for one GPU, CPU cores and GiB of memory."""
    # Modal calls the card A10 when scheduling and A10G in its billing rate card.
    units = {"gpu_hour_cost_a10g": 1, "cpu_hour_cost": CPU,
             "mem_gib_hour_cost": Decimal(MEMORY_MIB) / 1024}
    total = Decimal(0)
    for key, quantity in units.items():
        try:
            price = Decimal(str(rates[key]))
        except (KeyError, TypeError, InvalidOperation) as exc:
            raise ValueError(f"Missing or invalid Modal rate: {key}") from exc
        if not price.is_finite() or price < 0:
            raise ValueError(f"Invalid Modal rate: {key}")
        total += price * quantity
    if total <= 0:
        raise ValueError("Expected a positive total machine rate")
    return float(total)


def _publish_folder(staged, target, *, keep_extra=False):
    if not target.exists():
        staged.rename(target)
        return
    expected = {p.relative_to(staged) for p in staged.rglob("*") if p.is_file()}
    existing = {p.relative_to(target) for p in target.rglob("*") if p.is_file()}
    if not expected <= existing or (not keep_extra and expected != existing):
        raise FileExistsError(f"Existing local files differ: {target}")
    if any(_digest(staged / p) != _digest(target / p) for p in expected):
        raise FileExistsError(f"Existing local files differ: {target}")


def _small_artifact(stage, path, grid="original"):
    if path in (Path("run.log"), Path("gpu.csv")):
        return True
    if len(path.parts) < 2 or path.parts[0] != "output":
        return False
    relative = path.relative_to("output")
    if stage == "initial-eval":
        return str(relative) in ("run.json", "metrics.json", "resources.csv")
    if stage == "pilot":
        return (str(relative) in (*PILOT_INPUTS, "failure.json") or
                (relative.parent == Path("configs") and relative.name in BUDGET_CONFIGS))
    return str(relative) in _stage_files(stage, grid)


def _pilot_summary(repo, attempt, summary):
    """Run the two public report CLIs with exactly the inputs that will be published."""
    for name in PILOT_INPUTS:
        shutil.copyfile(attempt / "output" / name, summary / name)
    shutil.copytree(attempt / "output/configs", summary / "configs")
    shutil.copyfile(attempt / "attempt.json", summary / "attempt.json")
    env = {**os.environ, "PYTHONPATH": str(repo / "src")}
    for module, filename, options in (
        ("maxrl.report", "report.md", []), ("maxrl.budget", "budget.json", ["--configs", str(summary / "configs")])
    ):
        with (summary / filename).open("xb") as out:
            subprocess.run([sys.executable, "-m", module, "--pilot", str(summary), *options],
                           cwd=repo, env=env, stdout=out, check=True)


def download_attempt(record, repo, volume):
    """Retrieve and hash-check the small artifacts, leaving weights and raw shards on the Volume."""
    name = record["attempt_id"]
    stage = record["stage"]
    grid = record.get("grid", "original")
    _check_grid(stage, grid)
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"Invalid attempt ID: {name!r}")
    folder = repo / "runs/countdown" / name
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder.parent) as tmp:
        staged = Path(tmp) / name
        staged.mkdir()
        raw = b"".join(volume.read_file(f"{name}/attempt.json"))
        if json.loads(raw) != record:
            raise ValueError("Volume attempt.json differs from the returned record")
        (staged / "attempt.json").write_bytes(raw)
        for filename, expected in record["sha256"].items():
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid artifact path: {filename}")
            if not _small_artifact(stage, relative, grid):
                continue
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with target.open("xb") as f:
                for chunk in volume.read_file(f"{name}/{filename}"):
                    f.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"SHA-256 mismatch: {filename}")
        _publish_folder(staged, folder, keep_extra=True)
    if stage != "preflight" and record["status"] == "completed":
        midway = repo / "results/midway"
        target = midway / stage
        if stage == "replay":
            target = target / grid
        elif stage == "saved-train":
            target = midway / "saved-trajectories"
        elif stage == "saved-validation":
            target = midway / "pilot-lr-1.5e-7"
        elif stage == "midpoint" and record.get("source_learning_rate") == SELECTED_RATE:
            target = midway / "midpoint-lr-1.5e-7"
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
            summary = Path(tmp) / stage
            summary.mkdir()
            if stage == "pilot":
                _pilot_summary(repo, folder, summary)
            elif _stage_files(stage, grid):
                shutil.copyfile(folder / "attempt.json", summary / "attempt.json")
                for filename in _stage_files(stage, grid):
                    (summary / filename).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(folder / "output" / filename, summary / filename)
            else:
                for filename in ("metrics.json", "resources.csv"):
                    shutil.copyfile(folder / "output" / filename, summary / filename)
            _publish_folder(summary, target)
    return folder


try:
    import modal
except ModuleNotFoundError as exc:
    if exc.name != "modal":
        raise
    modal = None


if modal is not None:
    app = modal.App("controlled-countdown")
    image = build_image(Path(__file__).resolve().parents[1]) if modal.is_local() else None
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    resources = dict(image=image, cpu=CPU, memory=MEMORY_MIB, volumes={str(VOLUME_ROOT): volume},
                     max_containers=1)

    @app.function(**resources, timeout=1800)
    def preflight(provenance):
        volume.reload()
        return run_attempt(REMOTE_REPO, VOLUME_ROOT, volume.commit, stage="preflight", provenance=provenance)

    @app.function(**resources, gpu=GPU, timeout=24 * 3600)
    def run_gpu(stage, rates, pilot_attempt, provenance, grid="original", verification_attempt=None,
                initial_attempt=None):
        volume.reload()
        return run_attempt(REMOTE_REPO, VOLUME_ROOT, volume.commit, stage=stage, rates=rates,
                           pilot_attempt=pilot_attempt, provenance=provenance, grid=grid,
                           verification_attempt=verification_attempt, initial_attempt=initial_attempt)

    @app.local_entrypoint()
    def main(stage: str = "preflight", pilot_attempt: str = "", recover_attempt: str = "", grid: str = "original",
             verification_attempt: str = "", initial_attempt: str = ""):
        """Run one stage or retrieve the small artifacts of an existing attempt."""
        _check_grid(stage, grid)
        if stage not in STAGES or (stage in PILOT_DEPENDENT) != bool(pilot_attempt):
            raise ValueError(f"Choose a stage from {STAGES}. {PILOT_DEPENDENT} require --pilot-attempt.")
        if ((stage == "saved-train") != bool(verification_attempt)
                or (stage == "saved-train") != bool(initial_attempt)):
            raise ValueError("Supply --verification-attempt and --initial-attempt for saved-train")
        repo = Path(__file__).resolve().parents[1]

        def receive(record):
            folder = download_attempt(record, repo, volume)
            print(f"{record['stage']} {record['status']}: {folder}", flush=True)
            if record["status"] != "completed":
                raise RuntimeError(record.get("error", f"See {folder}"))

        if recover_attempt:
            if stage != "preflight" or pilot_attempt:
                raise ValueError("Use --recover-attempt on its own")
            if Path(recover_attempt).name != recover_attempt or recover_attempt in (".", ".."):
                raise ValueError(f"Invalid attempt ID: {recover_attempt!r}")
            record = json.loads(b"".join(volume.read_file(f"{recover_attempt}/attempt.json")))
            if record["attempt_id"] != recover_attempt or record["status"] not in {"completed", "failed"}:
                raise ValueError(f"Expected a finished record for {recover_attempt}")
            receive(record)
            return

        try:
            provenance = git_provenance(repo)
            receive(preflight.remote(provenance))
            if stage != "preflight":
                rates = json.loads(subprocess.check_output(["modal", "billing", "rates", "--json"], text=True))
                print(f"Requesting one {GPU}, {CPU} CPU cores and {MEMORY_MIB} MiB "
                      f"at ${hourly_rate(rates):.6f}/hour", flush=True)
                extra = (verification_attempt, initial_attempt) if stage == "saved-train" else ()
                receive(run_gpu.remote(stage, rates, pilot_attempt or None, provenance, grid, *extra))
        except Exception:
            print(f"Attempt recovery: modal volume ls {VOLUME_NAME} /\n"
                  "modal run scripts/modal_countdown.py --recover-attempt <attempt-id>", file=sys.stderr)
            raise
