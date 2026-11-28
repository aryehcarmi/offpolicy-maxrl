"""Read matrix attempts from a Modal Volume and publish verified planner inputs locally."""
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


TRAIN_FILES = ("run.json", "metrics.jsonl", "resources.csv", "failure.json")
TEST_FILES = ("run.json", "metrics.json", "resources.csv", "responses.jsonl")
METHODS_FILE = "matrix/methods.json"


def modal_call_running(modal, call_id):
    """A zero-timeout get polls one spawned input. A transport failure must abort the launch."""
    errors = modal.exception
    call = modal.FunctionCall.from_id(call_id)
    try:
        call.get(timeout=0)
    except TimeoutError:  # Built-in TimeoutError is the pinned SDK's unfinished-input signal.
        return True
    except (errors.FunctionTimeoutError, errors.OutputExpiredError, errors.NotFoundError,
            errors.RemoteError, errors.InternalFailure, errors.ExecutionError):
        return False
    except Exception:
        # get() can raise the remote Python exception or a local transport error.
        # The call graph is best-effort; only explicit terminal evidence settles this case.
        pending = list(call.get_call_graph())
        while pending:
            item = pending.pop()
            if item.function_call_id == call_id and item.status.name != "PENDING":
                return False
            pending.extend(item.children)
        raise
    return False


def _read(volume, path):
    try:
        return b"".join(volume.read_file(path))
    except FileNotFoundError:
        return None


def matrix_methods(volume, allowed):
    raw = _read(volume, METHODS_FILE)
    if raw is None:
        return None
    record = json.loads(raw)
    methods = record.get("methods") if isinstance(record, dict) else None
    if not isinstance(methods, list) or not methods or not all(name in allowed for name in methods):
        raise ValueError(f"Invalid method ownership in {METHODS_FILE}")
    return tuple(name for name in allowed if name in methods)


def claim_matrix_methods(volume, methods, allowed):
    """Commit the selection once, accepting a concurrent claim only for the same methods."""
    try:
        with volume.batch_upload() as batch:
            batch.put_file(io.BytesIO(json.dumps({"methods": methods}).encode()), METHODS_FILE)
    except FileExistsError:
        if matrix_methods(volume, allowed) != methods:
            raise ValueError(f"The Volume already owns a different set of methods in {METHODS_FILE}")


def _record(volume, name):
    raw = _read(volume, f"{name}/attempt.json")
    from_dispatch = raw is None
    if from_dispatch:
        raw = _read(volume, f"{name}/dispatch.json")
    if raw is None:
        return None
    record = json.loads(raw)
    if from_dispatch and record["status"] == "running":
        # A duplicate delivery can end while the worker that won this attempt is still starting.
        owner = _read(volume, f"{name}/worker.json")
        if owner is not None:
            record["function_call_id"] = json.loads(owner)["function_call_id"]
    stage, trial = record["matrix_stage"], Path(record["trial"])
    parts = trial.parts
    if (record["attempt_id"] != name or record["stage"] != "matrix" or record["trial"] != str(trial)
            or trial.is_absolute() or ".." in parts or
            not ((stage in ("screen", "tune", "main") and len(parts) == 2 and parts[0] == stage) or
                 (stage == "test" and len(parts) == 3 and parts[0] == "main" and parts[2] == "test"))):
        raise ValueError(f"Invalid matrix attempt identity in {name}: {record.get('trial')!r}")
    return record


def _state(volume, record, call_running):
    name = record["attempt_id"]
    while record["status"] == "running":
        call_id = record.get("function_call_id")
        if call_id and call_running(call_id):
            return "running", record
        # The coordinator may have handed off to the worker, or the worker may now be finished.
        record = _record(volume, name) or record
        if record["status"] == "running" and record.get("function_call_id") == call_id:
            return "failed", record
    if record["status"] in ("completed", "failed") and _read(volume, f"{name}/output/run.json") is not None:
        if record["status"] == "completed":
            return "done", record
        raw = _read(volume, f"{name}/output/failure.json")
        if raw is not None:
            if hashlib.sha256(raw).hexdigest() != record["sha256"].get("output/failure.json"):
                raise ValueError(f"SHA-256 mismatch: {name}/output/failure.json")
            try:
                failure = json.loads(raw)
            except ValueError:
                return "failed", record
            if isinstance(failure, dict) and failure.get("category") == "numerical":
                return "done", record
    return "failed", record


def matrix_snapshot(volume, call_running, *, count_superseded=True):
    """Read newest-first by the dispatch-time ID, independent of GPU assignment or finish order."""
    names = {entry.path.strip("/") for entry in volume.listdir("/")}
    latest = {}
    running = 0
    for name in sorted(names, reverse=True):
        if not name.startswith("matrix-") or "/" in name:
            continue
        record = _record(volume, name)
        if record is None:
            continue
        trial = record["trial"]
        if trial not in latest:
            latest[trial] = _state(volume, record, call_running)
            state, _ = latest[trial]
        elif count_superseded and record["status"] == "running":
            # A forced rerun can leave the previous worker using a GPU.
            state, _ = _state(volume, record, call_running)
        else:
            continue
        running += state == "running"
    return latest, running


def matrix_attempts(volume, call_running):
    latest, _ = matrix_snapshot(volume, call_running, count_superseded=False)
    return latest


def _exchange_directories(staged, target):
    """Atomically exchange two nonempty directories on the launch host (macOS or Linux)."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        result = libc.renamex_np(os.fsencode(staged), os.fsencode(target), 2)  # RENAME_SWAP
    else:
        result = libc.renameat2(-100, os.fsencode(staged), -100, os.fsencode(target), 2)  # RENAME_EXCHANGE
    if result:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), str(target))


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _publish_trial(target, record, archive):
    """Install the trial's small files together, preserving nested tests and unrelated local files."""
    names = TEST_FILES if record["matrix_stage"] == "test" else TRAIN_FILES
    if archive is None and not (target / "attempt.json").exists():
        return  # Nothing from an older collection to retire.
    outputs = []
    if archive is not None:
        for name in names:
            if f"output/{name}" in record["sha256"]:
                outputs.append(name)
            elif name != "failure.json" and not (name == "metrics.jsonl" and record["status"] == "failed"):
                raise ValueError(f"Missing SHA-256 inventory entry: {record['attempt_id']}/output/{name}")
        marker = target / "attempt.json"
        if marker.is_file() and marker.read_bytes() == (archive / "attempt.json").read_bytes():
            present = {name for name in names if (target / name).is_file()}
            if present == set(outputs) and all(
                _digest(target / name) == record["sha256"][f"output/{name}"] for name in outputs
            ):
                return

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        staged = Path(temporary) / "trial"
        if target.exists():
            # Ignore only prior planner inputs at the trial root. In particular, main/.../test
            # is its own trial and must stay visible when the parent is republished or retired.
            owned = {*names, "attempt.json"}
            shutil.copytree(target, staged, symlinks=True,
                            ignore=lambda directory, entries: owned if Path(directory) == target else ())
        else:
            staged.mkdir()
        if archive is not None:
            shutil.copyfile(archive / "attempt.json", staged / "attempt.json")
            for name in outputs:
                shutil.copyfile(archive / "output" / name, staged / name)
        if not target.exists():
            staged.rename(target)
        elif not any(staged.iterdir()):
            target.rename(Path(temporary) / "retired")
        else:
            _exchange_directories(staged, target)


def _print_trials(states):
    print("Trial\tState\tAttempt\tStarted (UTC)\tFinished (UTC)\tEstimated USD")
    if not states:
        print("No matrix attempts.")
    now = datetime.now(timezone.utc)
    for trial, (state, record) in sorted(states.items()):
        started = record["started_utc"]
        cost = record.get("estimated_cost_usd")
        if cost is None and state == "running" and record.get("hourly_usd") is not None:
            elapsed = (now - datetime.fromisoformat(started)).total_seconds()
            cost = max(elapsed, 0) * record["hourly_usd"] / 3600
        formatted = "-" if cost is None else f"{cost:.4f}"
        print(f"{trial}\t{state}\t{record['attempt_id']}\t{started}\t{record.get('finished_utc', '-')}\t{formatted}")


def collect_matrix(repo, volume, call_running, download_attempt):
    """Recover immutable per-attempt downloads and atomically refresh the planner's trial tree."""
    root = repo / "runs/matrix"
    root.mkdir(parents=True, exist_ok=True)
    # Concurrent invocations from this checkout must publish their snapshots in observation order.
    lock = os.open(root, os.O_RDONLY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        states = matrix_attempts(volume, call_running)
        _print_trials(states)
        for trial, (state, record) in sorted(states.items()):
            archive = None
            if state == "done":
                # An optional output may be absent, but a present one must have a recorded hash.
                allowed = TEST_FILES if record["matrix_stage"] == "test" else TRAIN_FILES
                output = f"{record['attempt_id']}/output"
                for entry in volume.listdir(output):
                    name = Path(entry.path).name
                    if name in allowed and f"output/{name}" not in record["sha256"]:
                        raise ValueError(f"Missing SHA-256 inventory entry: {output}/{name}")
                archive = download_attempt(record, repo, volume)
            _publish_trial(root / trial, record, archive)
        return states
    finally:
        os.close(lock)
