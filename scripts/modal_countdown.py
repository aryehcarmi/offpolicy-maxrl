"""Run controlled Countdown experiments and planner stages on Modal."""
from collections import defaultdict
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import io
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
STAGES = ("preflight", "initial-eval", "pilot", *PILOT_DEPENDENT, "matrix")
MATRIX_STAGES = ("screen", "tune", "main", "test")
MATRIX_METHODS = ("count", "snis", "fresh", "uncorrected")
MATRIX_GPU_LIMIT = 10
MATRIX_FREEZE = Path("matrix/frozen.json")
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
    # Planner commands use the literal executable "python". Resolve it inside the locked venv.
    if argv[0] == "python":
        env["PATH"] = os.pathsep.join([str(repo / ".venv/bin"), env.get("PATH", "")])
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


def parse_matrix_command(matrix_stage, command):
    """Identify one direct planner invocation before any job is dispatched."""
    argv = shlex.split(command)
    test = matrix_stage == "test"
    module = "maxrl.evaluate" if test else "maxrl.train"
    if matrix_stage not in MATRIX_STAGES or argv[:3] != ["python", "-m", module]:
        raise ValueError(f"Expected python -m {module} for {matrix_stage}: {command}")
    keys, values = argv[3::2], argv[4::2]
    required = {"--config", "--manifest", "--out", "--method", "--lr", "--seed"}
    if test:
        required.update(("--panel", "--checkpoint"))
    allowed = required | {"--device", "--dtype", "--generation-batch", "--score-batch", "--hourly-usd"}
    if (len(keys) != len(values) or len(keys) != len(set(keys))
            or not required <= set(keys) <= allowed or any(v.startswith("--") for v in values)):
        raise ValueError(f"Invalid {matrix_stage} planner arguments: {command}")
    flags = dict(zip(keys, values))
    out = Path(flags["--out"])
    trial = out.relative_to("runs/matrix")
    stage = "main" if test else matrix_stage
    if (len(trial.parts) != (3 if test else 2) or trial.parts[0] != stage
            or ".." in trial.parts or (test and trial.name != "test")):
        raise ValueError(f"Invalid {matrix_stage} trial output path: {out}")
    if test and (flags["--panel"] != "test" or Path(flags["--checkpoint"]).parent != out.parent / "checkpoints"):
        raise ValueError(f"Expected --panel test and the main trial's checkpoint: {command}")
    rate, seed = float(flags["--lr"]), int(flags["--seed"])
    if not math.isfinite(rate) or rate <= 0 or seed < 0:
        raise ValueError(f"Invalid rate or seed in planner command: {command}")
    return {"matrix_stage": matrix_stage, "trial": str(trial), "method": flags["--method"],
            "rate": rate, "seed": seed, "planner_command": command}


def _matrix_checkpoint(volume_root, job, main_attempt, freeze_sha256):
    folder, source = _attempt_input(volume_root, main_attempt, {"matrix"}, GPU, "test")
    _check_source_record(folder, source)
    if (source["matrix_stage"] != "main" or source["trial"] != str(Path(job["trial"]).parent)
            or source["freeze_sha256"] != freeze_sha256
            or any(source[k] != job[k] for k in ("method", "rate", "seed"))):
        raise ValueError("Main checkpoint source does not match this test trial and rate freeze")
    _check_files(folder, source, ["run.json"])
    run = json.loads((folder / "output/run.json").read_text())
    step = f"step-{run['config']['updates']:06d}"
    argv = shlex.split(job["planner_command"])
    if Path(argv[argv.index("--checkpoint") + 1]).name != step:
        raise ValueError(f"Expected the main trial's final checkpoint {step}")
    identity = f"{run['run_id']}/{step}"
    if identity not in source["verified_checkpoints"]:
        raise ValueError(f"Main attempt did not verify {identity}")
    checkpoint = folder / "output/checkpoints" / step
    prefix = f"output/checkpoints/{step}/"
    inventory = {key for key in source["sha256"] if key.startswith(prefix)}
    files = {str(path.relative_to(folder)) for path in checkpoint.rglob("*") if path.is_file()}
    if not files or files != inventory:
        raise ValueError(f"Checkpoint files differ from the SHA-256 inventory for {identity}")
    _check_files(folder, source, [key.removeprefix("output/") for key in sorted(files)])
    return checkpoint


def _matrix_argv(job, volume_root, output, hourly_usd, freeze_sha256, main_attempt):
    argv = shlex.split(job["planner_command"])
    replacements = {"--out": output, "--manifest": volume_root / "data/manifest.json"}
    if job["matrix_stage"] in ("main", "test") and _digest(volume_root / MATRIX_FREEZE) != freeze_sha256:
        raise ValueError("Volume rate freeze SHA-256 differs from this launch")
    if job["matrix_stage"] == "test":
        replacements["--checkpoint"] = _matrix_checkpoint(volume_root, job, main_attempt, freeze_sha256)
    if "--hourly-usd" in argv:
        replacements["--hourly-usd"] = hourly_usd
    else:
        argv.extend(["--hourly-usd", str(hourly_usd)])
    for flag, value in replacements.items():
        argv[argv.index(flag) + 1] = str(value)
    return argv


def new_attempt_id(stage):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stage}-{stamp}-{uuid4().hex[:12]}"


def _validate_matrix_attempt_id(name):
    if not isinstance(name, str) or not name.startswith("matrix-") or not all(
            c.isascii() and (c.isalnum() or c == "-") for c in name):
        raise ValueError(f"Invalid preallocated matrix attempt ID: {name!r}")


def _claim_matrix_worker(volume, attempt_id, function_call_id):
    """Pin the first GPU call to an attempt, even if a coordinator repeats an ambiguous spawn."""
    _validate_matrix_attempt_id(attempt_id)
    if not function_call_id:
        raise ValueError("A matrix worker requires its Modal function call ID")
    key = f"{attempt_id}/worker.json"
    try:
        with volume.batch_upload() as batch:
            data = json.dumps({"function_call_id": function_call_id}).encode()
            batch.put_file(io.BytesIO(data), key)
    except FileExistsError:
        return json.loads(b"".join(volume.read_file(key)))["function_call_id"]
    return function_call_id


def run_attempt(repo: Path, volume_root: Path, commit, *, stage: str, provenance: dict,
                rates: dict | None = None, pilot_attempt: str | None = None, grid: str = "original",
                verification_attempt=None, initial_attempt=None, matrix_stage=None, planner_command=None,
                function_call_id=None, freeze_sha256=None, main_attempt=None, attempt_id=None) -> dict:
    """Keep writable experiment files on local disk and preserve one Volume folder per invocation."""
    if stage not in STAGES or (stage not in PILOT_DEPENDENT and pilot_attempt is not None):
        raise ValueError(f"Invalid stage or pilot input for {stage!r}")
    _check_grid(stage, grid)
    repo, volume_root = Path(repo).resolve(), Path(volume_root).resolve()
    job = parse_matrix_command(matrix_stage, planner_command) if stage == "matrix" else None
    if job is not None and not function_call_id:
        raise ValueError("A matrix attempt requires its Modal function call ID")
    started = _utc()
    dispatched = attempt_id is not None
    if dispatched:
        if stage != "matrix":
            raise ValueError("Only matrix trials use preallocated attempt IDs")
        _validate_matrix_attempt_id(attempt_id)
    attempt_id = attempt_id or new_attempt_id(stage)
    attempt = volume_root / attempt_id
    attempt.mkdir(parents=True, exist_ok=dispatched)
    record_path = attempt / "attempt.json"
    if dispatched and record_path.exists():
        # A preempted input can be delivered again. Leave its artifacts intact for collection/retry.
        previous = json.loads(record_path.read_text())
        if previous["function_call_id"] == function_call_id and previous["status"] != "running":
            return previous
        raise FileExistsError(f"Matrix worker already started attempt {attempt_id}")
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
    if job is not None:
        record.update(job, function_call_id=function_call_id, freeze_sha256=freeze_sha256, main_attempt=main_attempt)
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
                if job is not None:
                    commands = [_matrix_argv(job, volume_root, output, hourly_usd, freeze_sha256, main_attempt)]
                else:
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
                    if stage in ("pilot", "saved-train") or (job is not None and matrix_stage != "test"):
                        commit()  # Verify the copied files on the Volume before releasing local scratch.
            except BaseException as error:
                previous = record.get("error", "")
                record.update(status="failed", error=f"{previous} Artifact copy failed: {error}".strip())
                emit(record["error"])
                if not isinstance(error, Exception):
                    raise
            finally:
                try:
                    metadata = {record_path, log, attempt / "dispatch.json", attempt / "worker.json"}
                    for path in sorted(attempt.rglob("*")):
                        if path.is_file() and path not in metadata:
                            record["sha256"][str(path.relative_to(attempt))] = _digest(path)
                    if stage == "pilot" or (job is not None and matrix_stage != "test"):
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
    if stage == "matrix":
        store = _matrix_store()
        names = store.TEST_FILES if record["matrix_stage"] == "test" else store.TRAIN_FILES
        matrix_files = {"run.log", "gpu.csv", *(f"output/{item}" for item in names)}
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
            keep = filename in matrix_files if stage == "matrix" else _small_artifact(stage, relative, grid)
            if not keep:
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
    if stage not in ("preflight", "matrix") and record["status"] == "completed":
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


def _matrix_store():
    # Modal imports a file entrypoint by basename. The checkout is also an
    # importable namespace package when called via python -m pytest.
    try:
        from scripts import modal_matrix_store
    except ModuleNotFoundError as exc:
        if exc.name != "scripts":
            raise
        import modal_matrix_store
    return modal_matrix_store


def _matrix_call_running(call_id):
    return _matrix_store().modal_call_running(modal, call_id)


def _fetch_matrix_file(repo, name, expected):
    target = repo / name
    if target.is_file():
        if _digest(target) != expected:
            raise ValueError(f"Local SHA-256 mismatch: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        staged = Path(tmp) / target.name
        with staged.open("xb") as out:
            for chunk in volume.read_file(name):
                out.write(chunk)
        if _digest(staged) != expected:
            raise ValueError(f"Volume SHA-256 mismatch: {name}")
        staged.replace(target)


def ensure_matrix_data(repo):
    _fetch_matrix_file(repo, "data/manifest.json", MANIFEST_SHA256)
    manifest = repo / "data/manifest.json"
    for panel in json.loads(manifest.read_text())["panels"].values():
        _fetch_matrix_file(repo, f"data/{panel['file']}", panel["sha256"])
    print(f"Verified local manifest {verify_manifest(manifest)} and all shards", flush=True)


def _run_planner(repo, stage):
    # The Modal CLI can live in its own environment without the project installed.
    python = repo / ".venv/bin/python"
    proc = subprocess.run([str(python) if python.is_file() else sys.executable, "-m", "maxrl.matrix", stage],
                          cwd=repo, env={**os.environ, "PYTHONPATH": str(repo / "src")},
                          stdout=subprocess.PIPE, text=True)
    # stderr inherits the terminal, including the planner's fallback notices.
    if proc.returncode:
        raise RuntimeError(f"maxrl.matrix {stage} exited with status {proc.returncode}")
    return proc.stdout.splitlines()


def _remote_freeze():
    try:
        return b"".join(volume.read_file(str(MATRIX_FREEZE)))
    except FileNotFoundError:
        return None


def _matrix_freeze(repo, stage, *, dry_run=False):
    path = repo / "runs/matrix/frozen.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.exists()
    if stage == "main" and not existing:
        _run_planner(repo, "freeze")
    remote = _remote_freeze()
    if remote is not None and not path.exists():
        # Recovery on a fresh laptop, including a standalone collect followed by compare.
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            staged = Path(tmp) / path.name
            staged.write_bytes(remote)
            staged.replace(path)
    if remote is None:
        if stage == "collect":
            return None
        if stage == "test":
            raise ValueError("No main-stage rate freeze on the Volume")
        if existing:
            # A supplied or interrupted local freeze must pass the planner before
            # it can become the immutable Volume copy. The planner owns this check.
            _run_planner(repo, "main")
        if dry_run:
            return _digest(path)
        # Create only. A simultaneous main launch may have won this upload.
        try:
            with volume.batch_upload() as batch:
                batch.put_file(path, str(MATRIX_FREEZE))
        except FileExistsError:
            pass
        remote = _remote_freeze()
    if remote != path.read_bytes():
        raise ValueError(f"The Volume holds a different rate freeze from {path}")
    return hashlib.sha256(remote).hexdigest()


def _matrix_plan(repo, stage, dry_run):
    if stage not in ("main", "test"):
        return None, _run_planner(repo, stage)
    if dry_run and not (repo / "runs/matrix/frozen.json").exists():
        # The planner uses relative paths in both its freeze and its printed commands.
        # Preview a first freeze with those same paths without installing it in the checkout.
        with tempfile.TemporaryDirectory() as tmp:
            preview = Path(tmp)
            for name in ("src", "configs", "data", ".venv", "runs/matrix/screen", "runs/matrix/tune"):
                dest = preview / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.symlink_to(repo / name, target_is_directory=True)
            freeze = _matrix_freeze(preview, stage, dry_run=True)
            return freeze, _run_planner(preview, stage)
    freeze = _matrix_freeze(repo, stage, dry_run=dry_run)
    return freeze, _run_planner(repo, stage)


def _matrix_jobs(matrix_stage, commands, rerun):
    jobs = [parse_matrix_command(matrix_stage, line) for line in commands]
    names = {job["trial"] for job in jobs}
    if len(names) != len(jobs):
        raise ValueError("Planner returned duplicate trial paths")
    if rerun and rerun not in names:
        raise ValueError(f"Unknown {matrix_stage} trial for --rerun: {rerun}")
    return [job for job in jobs if not rerun or job["trial"] == rerun]


def _launch_decision(job, attempts, rerun):
    trial = job["trial"]
    state, _ = attempts.get(trial, (None, None))
    if state in ("done", "running") and trial != rerun:
        return f"skipped because {state}", None
    if job["matrix_stage"] == "test":
        parent = str(Path(trial).parent)
        state, record = attempts.get(parent, (None, {}))
        if state != "done" or record["status"] != "completed":
            return f"skipped because {parent} has no completed main attempt", None
        return None, record["attempt_id"]
    return None, None


def _volume_json(path):
    try:
        return json.loads(b"".join(volume.read_file(path)))
    except FileNotFoundError:
        return None


def _json_stream(value):
    return io.BytesIO(json.dumps(value, allow_nan=False).encode())


def _claim_matrix_attempt(job, previous_id, provenance, freeze_sha256, main_attempt, coordinator, submission):
    # One create-only Volume commit both excludes competing launchers and leaves
    # a receipt this coordinator can find if Modal delivers its input again.
    attempt_id = new_attempt_id("matrix")
    record = dict(job, **provenance, attempt_id=attempt_id, stage="matrix", status="running",
                  started_utc=_utc(), freeze_sha256=freeze_sha256, main_attempt=main_attempt,
                  function_call_id=coordinator, coordinator_call_id=coordinator, sha256={})
    claim = f"matrix/claims/{job['trial']}/{previous_id}.json"
    with volume.batch_upload() as batch:
        batch.put_file(_json_stream(attempt_id), claim)
        batch.put_file(_json_stream(attempt_id), submission)
        batch.put_file(_json_stream(record), f"{attempt_id}/dispatch.json")
    return record


def _submit_matrix_job(job, attempts, rates, provenance, freeze_sha256, rerun, coordinator):
    trial = job["trial"]
    submission = f"matrix/submissions/{coordinator}/{trial}.json"
    attempt_id = _volume_json(submission)
    if attempt_id:
        record = _volume_json(f"{attempt_id}/dispatch.json")
        # Preemption may have happened after spawn but before its ID was saved.
        # A worker can already have taken exclusive ownership in that interval.
        owner = _volume_json(f"{attempt_id}/worker.json")
        call_id = owner["function_call_id"] if owner else record["function_call_id"]
        if call_id != coordinator:
            print(f"{trial}  waiting for previously submitted {attempt_id}  {call_id}", flush=True)
            return modal.FunctionCall.from_id(call_id)
    else:
        while True:
            reason, main_attempt = _launch_decision(job, attempts, rerun)
            if reason:
                print(f"{trial}  {reason}", flush=True)
                return None
            prior = attempts.get(trial, (None, {}))[1].get("attempt_id", "first")
            try:
                record = _claim_matrix_attempt(job, prior, provenance, freeze_sha256, main_attempt,
                                               coordinator, submission)
                break
            except FileExistsError:
                # Another coordinator observed the same predecessor. Apply the
                # requested skip/rerun rule to the winning, durably visible attempt.
                attempts = _matrix_store().matrix_attempts(volume, _matrix_call_running)
                if attempts.get(trial, (None, {}))[1].get("attempt_id", "first") == prior:
                    raise
    attempt_id = record["attempt_id"]
    call = run_matrix_trial.spawn(job["matrix_stage"], job["planner_command"], rates, provenance,
                                  freeze_sha256, record["main_attempt"], attempt_id)
    record["function_call_id"] = call.object_id
    try:
        with volume.batch_upload(force=True) as batch:
            batch.put_file(_json_stream(record), f"{attempt_id}/dispatch.json")
    except Exception as exc:
        # The spawn succeeded. Keep its handle and wait even if this receipt
        # update failed. The worker commits its own call ID and attempt record.
        print(f"{trial}  could not update dispatch receipt for {call.object_id}: {exc}", file=sys.stderr)
    print(f"{trial}  started  {attempt_id}  {call.object_id}", flush=True)
    return call


def dispatch_matrix(matrix_stage, commands, rates, provenance, freeze_sha256, rerun):
    """Run on Modal so every spawn and wait survives a disconnected local entrypoint."""
    jobs = _matrix_jobs(matrix_stage, commands, rerun)
    attempts = _matrix_store().matrix_attempts(volume, _matrix_call_running)
    coordinator = modal.current_function_call_id()
    calls, errors = [], []
    for job in jobs:
        trial = job["trial"]
        try:
            call = _submit_matrix_job(job, attempts, rates, provenance, freeze_sha256, rerun, coordinator)
            if call is not None:
                calls.append((trial, call))
        except Exception as exc:
            errors.append(f"{trial}: {exc}")
            print(f"{trial}  submission failed  {exc}", flush=True)
    for trial, call in calls:
        try:
            result = call.get()
            if result and "existing_worker_call_id" in result:
                modal.FunctionCall.from_id(result["existing_worker_call_id"]).get()
        except Exception as exc:
            errors.append(f"{trial}: {exc}")
    return errors


def _matrix_workspace():
    return modal.Workspace.from_context().hydrate().name


def _account_methods(store, stage, methods):
    workspace = _matrix_workspace()
    recorded = store.matrix_methods(volume, MATRIX_METHODS)
    selected = recorded or MATRIX_METHODS
    try:
        if methods is not None:
            if stage == "collect":
                raise ValueError("--methods applies to a matrix launch, not collect")
            requested = {name.strip() for name in methods.split(",")}
            unknown = requested.difference(MATRIX_METHODS)
            if unknown:
                raise ValueError(f"Unknown --methods names: {sorted(unknown)}. Choose from {MATRIX_METHODS}")
            if recorded and requested != set(recorded):
                raise ValueError(f"This workspace owns {','.join(recorded)} but --methods requested {methods}")
            selected = tuple(name for name in MATRIX_METHODS if name in requested)
    finally:
        print(f"Modal workspace: {workspace}  Methods: {','.join(selected)}", flush=True)
    return selected, recorded


def launch_matrix(repo, matrix_stage, *, rerun="", dry_run=False, methods=None):
    store = _matrix_store()
    selected, recorded = _account_methods(store, matrix_stage, methods)

    def collect():
        return store.collect_matrix(repo, volume, _matrix_call_running, download_attempt)

    if matrix_stage == "collect":
        if rerun or dry_run:
            raise ValueError("--rerun and --dry-run apply to a matrix launch")
        if _matrix_freeze(repo, "collect"):
            ensure_matrix_data(repo)
        collect()
        return
    if matrix_stage not in MATRIX_STAGES:
        raise ValueError(f"Choose --matrix-stage from {(*MATRIX_STAGES, 'collect')}")
    if rerun:
        # Planner trial paths use stage/method-index or main/method-seed[/test].
        parts = Path(rerun).parts
        if len(parts) >= 2:
            method = parts[1].partition("-")[0]
            if method in MATRIX_METHODS and method not in selected:
                raise ValueError(f"Cannot --rerun {rerun}: this workspace does not own method {method}")
    if not dry_run:
        provenance = git_provenance(repo)
        record = preflight.remote(provenance)
        folder = download_attempt(record, repo, volume)
        print(f"preflight {record['status']}: {folder}", flush=True)
        if record["status"] != "completed":
            raise RuntimeError(record.get("error", f"See {folder}"))
    ensure_matrix_data(repo)
    collect()
    freeze, commands = _matrix_plan(repo, matrix_stage, dry_run)
    jobs = _matrix_jobs(matrix_stage, commands, rerun)
    if rerun and jobs[0]["method"] not in selected:
        raise ValueError(f"Cannot --rerun {rerun}: this workspace does not own method {jobs[0]['method']}")
    jobs = [job for job in jobs if job["method"] in selected]
    attempts, running = store.matrix_snapshot(volume, _matrix_call_running)
    would_start = 0
    for job in jobs:
        reason, _ = _launch_decision(job, attempts, rerun)
        if not reason:
            would_start += 1
        if dry_run:
            action = reason or f"would start  {job['planner_command']}"
            print(f"{job['trial']}  {action}", flush=True)
    if running + would_start > MATRIX_GPU_LIMIT:
        raise ValueError(f"{running} running + {would_start} would start exceeds "
                         f"the matrix GPU limit of {MATRIX_GPU_LIMIT}")
    if dry_run:
        return

    rates = json.loads(subprocess.check_output(["modal", "billing", "rates", "--json"], text=True))
    print(f"Matrix {matrix_stage}: up to {MATRIX_GPU_LIMIT} {GPU} workers at ${hourly_rate(rates):.6f}/hour each",
          flush=True)
    commands = [job["planner_command"] for job in jobs]
    if recorded is None:
        if methods is not None:
            store.claim_matrix_methods(volume, selected, MATRIX_METHODS)
        else:
            # A first explicit launch may have claimed this workspace during preflight.
            recorded = store.matrix_methods(volume, MATRIX_METHODS)
            if recorded is not None and recorded != selected:
                raise ValueError(f"Workspace methods changed to {','.join(recorded)} during planning. Retry the launch.")
    try:
        call = run_matrix_stage.spawn(matrix_stage, commands, rates, provenance, freeze, rerun)
        print(f"Stage handed to Modal coordinator {call.object_id}", flush=True)
        errors = call.get()
    finally:
        latest = collect()
    failed = [job["trial"] for job in jobs if latest.get(job["trial"], (None,))[0] == "failed"]
    if errors or failed:
        raise RuntimeError("Matrix attempts failed: " + ", ".join(errors + failed))


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

    @app.cls(**{**resources, "max_containers": MATRIX_GPU_LIMIT}, gpu=GPU, timeout=24 * 3600)
    class MatrixTrials:
        @modal.method()
        def run(self, matrix_stage, command, rates, provenance, freeze_sha256=None, main_attempt=None,
                attempt_id=None):
            if attempt_id is None:
                attempt_id = new_attempt_id("matrix")
            call_id = modal.current_function_call_id()
            owner = _claim_matrix_worker(volume, attempt_id, call_id)
            if owner != call_id:
                return {"existing_worker_call_id": owner}
            volume.reload()
            return run_attempt(REMOTE_REPO, VOLUME_ROOT, volume.commit, stage="matrix", rates=rates,
                               provenance=provenance, matrix_stage=matrix_stage, planner_command=command,
                               function_call_id=call_id, freeze_sha256=freeze_sha256,
                               main_attempt=main_attempt, attempt_id=attempt_id)

    run_matrix_trial = MatrixTrials().run

    @app.function(image=image, max_containers=1, timeout=24 * 3600)
    def run_matrix_stage(matrix_stage, commands, rates, provenance, freeze_sha256, rerun):
        return dispatch_matrix(matrix_stage, commands, rates, provenance, freeze_sha256, rerun)

    @app.local_entrypoint()
    def main(stage: str = "preflight", pilot_attempt: str = "", recover_attempt: str = "", grid: str = "original",
             verification_attempt: str = "", initial_attempt: str = "", matrix_stage: str = "",
             rerun: str = "", dry_run: bool = False, methods: str | None = None):
        """Run one stage or retrieve the small artifacts of an existing attempt."""
        _check_grid(stage, grid)
        if stage == "matrix":
            if any((pilot_attempt, recover_attempt, verification_attempt, initial_attempt)):
                raise ValueError("Choose a matrix stage without legacy attempt options")
            return launch_matrix(Path(__file__).resolve().parents[1], matrix_stage,
                                 rerun=rerun, dry_run=dry_run, methods=methods)
        if matrix_stage or rerun or dry_run or methods is not None:
            raise ValueError("--matrix-stage, --methods, --rerun and --dry-run require --stage matrix")
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
