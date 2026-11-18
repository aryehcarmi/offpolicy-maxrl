"""Run the controlled Countdown evaluation, pilot or midpoint protocol on Modal."""
from collections import defaultdict
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
import time
from uuid import uuid4


MANIFEST_SHA256 = "abdb27fbeeefdae2c9f6cce441811af401e07937f9b6bd4693f8a5cabb556186"
STAGES = ("preflight", "initial-eval", "pilot", "midpoint")
TEST_SDK = "/opt/countdown-test-sdk"


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


def job_commands(stage, repo, manifest, output, hourly_usd, pilot_run=None):
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
    if stage != "midpoint" or pilot_run is None:
        raise ValueError(f"Expected a GPU stage and a pilot input for midpoint, got {stage!r}")
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


def _pilot_input(volume_root, pilot_attempt):
    if not pilot_attempt or Path(pilot_attempt).name != pilot_attempt or pilot_attempt in (".", ".."):
        raise ValueError("A midpoint job requires a pilot attempt directory name")
    folder = volume_root / pilot_attempt
    record = json.loads((folder / "attempt.json").read_text())
    if record["stage"] != "pilot" or record["status"] != "completed":
        raise ValueError(f"Pilot attempt {pilot_attempt} did not complete")
    return folder / "output"


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _record(path, record):
    staged = path.with_suffix(".tmp")
    staged.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    staged.replace(path)


def run_attempt(repo: Path, volume_root: Path, commit, *, stage: str,
                rates: dict | None = None, pilot_attempt: str | None = None) -> dict:
    """Keep writable experiment files on local disk and preserve one Volume folder per invocation."""
    if stage not in STAGES or (stage != "midpoint" and pilot_attempt is not None):
        raise ValueError(f"Invalid stage or pilot input for {stage!r}")
    repo, volume_root = Path(repo).resolve(), Path(volume_root).resolve()
    started = _utc()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt_id = f"{stage}-{stamp}-{uuid4().hex[:12]}"
    attempt = volume_root / attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    record = {"attempt_id": attempt_id, "stage": stage, "pilot_attempt": pilot_attempt,
              "status": "running", "started_utc": started, "gpu_name": None, "gpu_type": None,
              "hourly_usd": None, "sha256": {}}
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
                hourly_usd = hourly_rate(rates, record["gpu_type"])
                record["hourly_usd"] = hourly_usd
                emit(f"Assigned {record['gpu_name']}, {CPU} CPU cores and {MEMORY_MIB} MiB "
                     f"at ${hourly_usd:.6f}/hour")
                _record(record_path, record)
                commit()
                manifest = volume_root / "data/manifest.json"
                record["manifest_sha256"] = verify_manifest(manifest)
                pilot = _pilot_input(volume_root, pilot_attempt) if stage == "midpoint" else None
                commands = job_commands(stage, repo, manifest, output, hourly_usd, pilot)
                _run([repo / ".venv/bin/python", "-c", "import torch; "
                      "assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'expected one CUDA GPU'; "
                      "print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"], repo, emit)
                for command in commands:
                    _run(command, repo, emit)
                if stage == "initial-eval":
                    _run([repo / ".venv/bin/python", "-c",
                          "from pathlib import Path; import sys; "
                          "from scripts.modal_countdown import verify_initial_output; "
                          "verify_initial_output(Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]))",
                          output, manifest, str(hourly_usd)], repo, emit)
                if not output.is_dir():
                    raise FileNotFoundError(f"Job exited without creating {output}")
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
            except BaseException as error:
                previous = record.get("error", "")
                record.update(status="failed", error=f"{previous} Artifact copy failed: {error}".strip())
                emit(record["error"])
                if not isinstance(error, Exception):
                    raise
            finally:
                try:
                    for path in sorted(attempt.rglob("*")):
                        if path.is_file() and path != record_path:
                            record["sha256"][str(path.relative_to(attempt))] = _digest(path)
                except OSError as error:
                    previous = record.get("error", "")
                    record.update(status="failed", error=f"{previous} Artifact read failed: {error}".strip())
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
GPU_OPTIONS = ("L4", "A10")
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
    for folder in ("configs", "scripts", "tests"):
        image = image.add_local_dir(repo / folder, str(REMOTE_REPO / folder), ignore=["__pycache__"])
    return image


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
    if model == "A10G":
        model = "A10"
    if model not in GPU_OPTIONS:
        raise ValueError(f"Unexpected GPU {name!r}; requested {GPU_OPTIONS}")
    return model


def hourly_rate(rates, gpu):
    """Use the current function rates for one GPU, CPU cores and GiB of memory."""
    # Modal calls the fallback A10 when scheduling and A10G in its billing rate card.
    gpu_rate_key = {"L4": "gpu_hour_cost_l4", "A10": "gpu_hour_cost_a10g"}[gpu_type(gpu)]
    units = {gpu_rate_key: 1, "cpu_hour_cost": CPU,
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


def _publish_folder(staged, target):
    if not target.exists():
        staged.rename(target)
        return
    expected = {p.relative_to(staged) for p in staged.rglob("*") if p.is_file()}
    existing = {p.relative_to(target) for p in target.rglob("*") if p.is_file()}
    if expected != existing or any(_digest(staged / p) != _digest(target / p) for p in expected):
        raise FileExistsError(f"Existing local files differ: {target}")


def download_attempt(record, repo, volume):
    """Verify the complete Volume copy, then publish the two initial-evaluation summary files."""
    name = record["attempt_id"]
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
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with target.open("xb") as f:
                for chunk in volume.read_file(f"{name}/{filename}"):
                    f.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"SHA-256 mismatch: {filename}")
        _publish_folder(staged, folder)
    if record["stage"] == "initial-eval" and record["status"] == "completed":
        midway = repo / "results/midway"
        midway.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=midway) as tmp:
            summary = Path(tmp) / "initial-eval"
            summary.mkdir()
            for filename in ("metrics.json", "resources.csv"):
                shutil.copyfile(folder / "output" / filename, summary / filename)
            _publish_folder(summary, midway / "initial-eval")
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
    def preflight():
        volume.reload()
        return run_attempt(REMOTE_REPO, VOLUME_ROOT, volume.commit, stage="preflight")

    @app.function(**resources, gpu=list(GPU_OPTIONS), timeout=24 * 3600)
    def run_gpu(stage, rates, pilot_attempt):
        volume.reload()
        return run_attempt(REMOTE_REPO, VOLUME_ROOT, volume.commit, stage=stage, rates=rates,
                           pilot_attempt=pilot_attempt)

    @app.local_entrypoint()
    def main(stage: str = "preflight", pilot_attempt: str = ""):
        """Check the CPU environment, then execute only the named stage."""
        if stage not in STAGES or (stage == "midpoint") != bool(pilot_attempt):
            raise ValueError(f"Choose a stage from {STAGES}. Only midpoint requires --pilot-attempt.")
        repo = Path(__file__).resolve().parents[1]

        def receive(record):
            folder = download_attempt(record, repo, volume)
            print(f"{record['stage']} {record['status']}: {folder}", flush=True)
            if record["status"] != "completed":
                raise RuntimeError(record.get("error", f"See {folder}"))

        try:
            receive(preflight.remote())
            if stage != "preflight":
                rates = json.loads(subprocess.check_output(["modal", "billing", "rates", "--json"], text=True))
                options = ", ".join(f"{gpu} ${hourly_rate(rates, gpu):.6f}/hour" for gpu in GPU_OPTIONS)
                print(f"Requesting one GPU in preference order ({options}), "
                      f"{CPU} CPU cores and {MEMORY_MIB} MiB", flush=True)
                receive(run_gpu.remote(stage, rates, pilot_attempt or None))
        except Exception:
            print(f"Attempt recovery: modal volume ls {VOLUME_NAME} /\n"
                  f"modal volume get {VOLUME_NAME} <attempt-id> runs/countdown/", file=sys.stderr)
            raise
