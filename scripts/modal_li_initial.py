"""Run the pinned Li initial-model evaluation or check its environment on a CPU."""
import csv
from collections import deque
from datetime import datetime, timezone
import filecmp
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
from uuid import uuid4

VOLUME_NAME = "li-replication"


def validate_hourly_usd(rate):
    if rate is not None and (isinstance(rate, bool) or not math.isfinite(rate) or rate < 0):
        raise ValueError("hourly_usd must be a finite, nonnegative total machine rate")
    return rate


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _write_json(path, record):
    data = json.dumps(record, indent=2, allow_nan=False) + "\n"
    with tempfile.TemporaryDirectory(dir=path.parent) as staging:
        temp = Path(staging) / path.name
        temp.write_text(data, encoding="utf-8")
        temp.replace(path)


def _stopping_frame(tail):
    """Extract the innermost frame of the last actual Python traceback."""
    frame = None
    in_trace = False
    exception = ""
    for line in tail:
        if "Traceback (most recent call last):" in line:
            in_trace, frame, exception = True, None, ""
            continue
        if in_trace:
            match = re.search(r'File "([^"]+)", line (\d+), in (\S+)\s*$', line)
            if match:
                frame = match.groups()
            elif frame and re.match(r"^[\w.]+(?:Error|Exception)\b", line.strip()):
                exception = line.strip()
                in_trace = False
    if frame is None:
        return False, next((s.strip() for s in reversed(tail) if s.strip()), "no subprocess output")
    path, number, function = frame
    device_file = Path(path).as_posix().split("/")[-3:] == ["vllm", "config", "device.py"]
    # This pin infers its device in DeviceConfig.__post_init__. An import or
    # code defect there is still a failed preflight, even with the correct path.
    message = exception.removeprefix("RuntimeError:").lower()
    device_diagnostic = (re.search(r"\b(device|gpu|accelerator|platform|cuda|backend)s?\b", message)
                         and re.search(r"\b(no|not|unavailable|unable|cannot|couldn't|failed|"
                                       r"missing|absent|unknown|undetected)\b", message))
    expected = (device_file and function == "__post_init__" and
                exception.startswith("RuntimeError:") and bool(device_diagnostic))
    location = f'{path}:{number} in {function}'
    return expected, f"{location} — {exception}".rstrip(" —")


def _logged_command(argv, cwd, env, emit, *, capture=False):
    argv = list(map(str, argv))
    emit("$ " + shlex.join(argv))
    lines = [] if capture else deque(maxlen=2048)
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
    try:
        with process.stdout as pipe:
            for line in pipe:
                line = line.rstrip("\n")
                emit(line)
                lines.append(line)
        return process.wait(), lines
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def _read_evaluation(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    for key in ("mean_accuracy", "published_fraction"):
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid {key} in {path}")
    json.dumps(result, allow_nan=False)
    return result


def run_attempt(upstream: Path, wrapper: Path, volume_root: Path, commit, *,
                preflight: bool = False, hourly_usd: float | None = None) -> dict:
    """Commit the attempt and return its record, including ordinary evaluator failures."""
    validate_hourly_usd(hourly_usd)
    upstream, wrapper, volume_root = (Path(p).resolve() for p in (upstream, wrapper, volume_root))
    started = _utc()
    stamp = started.replace("-", "").replace(":", "").replace(".", "")
    run_id = ("preflight-" if preflight else "") + stamp + "-" + uuid4().hex[:12]
    attempt = volume_root / run_id
    attempt.mkdir(parents=True, exist_ok=False)
    started_clock = time.monotonic()
    log = attempt / "run.log"
    record_path = attempt / "run.json"
    record = {
        "run_id": run_id, "kind": "preflight" if preflight else "evaluation", "status": "running",
        "started_utc": started, "finished_utc": None, "wall_seconds": None,
        "gpu_name": None, "gpu_memory_mib": None, "driver_version": None,
        "upstream_commit": None, "upstream_diff": None, "hourly_usd": hourly_usd,
        "estimated_cost_usd": None, "log_path": f"{run_id}/run.log", "record_path": f"{run_id}/run.json",
    }
    python = upstream / ".venv-tim/bin/python"
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "VIRTUAL_ENV": str(python.parent.parent),
           "PATH": str(python.parent) + os.pathsep + os.environ.get("PATH", ""),
           "PYTHONPATH": str(upstream)}

    with log.open("x", encoding="utf-8", buffering=1) as stream:
        last_commit = started_clock

        def emit(line):
            nonlocal last_commit
            print(line, file=stream, flush=True)
            print(line, flush=True)
            if time.monotonic() - last_commit >= 60:
                last_commit = time.monotonic()
                try:
                    commit()
                except Exception as error:
                    # A transient snapshot failure must not kill a paid run.
                    # The required terminal commit still propagates errors.
                    notice = f"Periodic Volume snapshot failed: {error}"
                    print(notice, file=stream, flush=True)
                    print(notice, flush=True)

        def probe(command):
            code, lines = _logged_command(command, upstream, env, emit, capture=True)
            if code:
                _, stop = _stopping_frame(lines)
                record["stopping_point"] = stop
                raise RuntimeError(f"probe exited {code}: {stop}")
            return "\n".join(lines)

        try:
            emit(f"Starting {record['kind']} attempt {run_id}")
            _write_json(record_path, record)
            commit()  # an abruptly interrupted run remains identifiable
            config_path = wrapper.parents[1] / "configs/replication/li_frozen.json"
            spec = json.loads(config_path.read_text(encoding="utf-8"))
            head = probe(["git", "-C", upstream, "rev-parse", "HEAD"]).strip()
            record["upstream_commit"] = head
            diff = probe(["git", "-C", upstream, "diff", "HEAD"])
            record["upstream_diff"] = diff + "\n" if diff else ""
            if head != spec["upstream_commit"]:
                raise ValueError(f"upstream HEAD {head} differs from pinned {spec['upstream_commit']}")

            if preflight:
                check_cpu = (
                    "import sys, torch, vllm\n"
                    "print(f'python={sys.executable}')\n"
                    "print(f'torch={torch.__version__} from {torch.__file__}')\n"
                    "print(f'vllm={vllm.__version__} from {vllm.__file__}')\n"
                    "if torch.cuda.is_available():\n"
                    "    raise RuntimeError('preflight requires a CPU worker without CUDA devices')"
                )
                probe([python, "-c", check_cpu])
                # Load the wrapper by path just as the real invocation does.
                # Check the discovered CONFIG without executing its main function.
                check_config = (
                    "import json, pathlib, runpy, sys\n"
                    "path = pathlib.Path(runpy.run_path(sys.argv[1])['CONFIG']).resolve()\n"
                    "if path != pathlib.Path(sys.argv[2]) or "
                    "json.loads(path.read_text())['upstream_commit'] != sys.argv[3]:\n"
                    "    raise ValueError(f'wrapper configuration mismatch at {path}')\n"
                    "print(f'verified wrapper configuration {path}')"
                )
                probe([python, "-c", check_config, wrapper, config_path, head])
            else:
                gpu_csv = probe(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                 "--format=csv,noheader,nounits"])
                gpus = list(csv.reader(gpu_csv.splitlines(), skipinitialspace=True))
                if len(gpus) != 1 or len(gpus[0]) != 3:
                    raise ValueError(f"expected one GPU from nvidia-smi, received {gpus}")
                name, memory, driver = gpus[0]
                record.update(gpu_name=name.strip(), gpu_memory_mib=int(memory.strip()),
                              driver_version=driver.strip())

            with tempfile.TemporaryDirectory() as scratch:
                output = Path(scratch) if preflight else attempt
                argv = [python, wrapper, "evaluate", "--method", "initial",
                        "--upstream", upstream, "--out", output]
                code, tail = _logged_command(argv, upstream, env, emit)
                record["wrapper_returncode"] = code
                reached_device, stop = _stopping_frame(tail)
                if preflight:
                    record["stopping_point"] = stop
                    if any(output.rglob("evaluation.json")):
                        raise RuntimeError("CPU preflight unexpectedly produced an evaluation")
                    if code == 0 or not reached_device:
                        raise RuntimeError(f"expected CPU stop in vLLM device selection; observed {stop}")
                    emit("CPU preflight reached device selection: " + stop)
                else:
                    if code:
                        record["stopping_point"] = stop
                        raise RuntimeError(f"evaluator exited {code}: {stop}")
                    record["evaluation"] = _read_evaluation(output / "initial/evaluation.json")
            record["status"] = "completed"
        except BaseException as error:
            # Python interruptions and cancellations are durable failed attempts.
            # An uncatchable worker kill can only leave the last running snapshot.
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"
            record.setdefault("stopping_point", record["error"])
            print(record["error"], file=stream, flush=True)
            print(record["error"], flush=True)
            if not isinstance(error, Exception):
                raise  # The finally block commits the failure before Modal sees the interruption.
        finally:
            elapsed = time.monotonic() - started_clock
            record["finished_utc"] = _utc()
            record["wall_seconds"] = elapsed
            record["estimated_cost_usd"] = hourly_usd * elapsed / 3600 if hourly_usd is not None else None
            stream.flush()
            with log.open("rb") as f:
                record["log_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
            _write_json(record_path, record)
            try:
                commit()
            except Exception as error:
                raise RuntimeError(f"Could not commit final record and log to Volume {VOLUME_NAME} "
                                   f"at {record['log_path']}: {error}") from error
    return record


def download_attempt(record: dict, folder: Path, volume) -> None:
    """Publish a complete verified copy. Preserve the original bytes of each artifact."""
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder.parent) as tmp:
        stage = Path(tmp)
        raw_record = b"".join(volume.read_file(f"{folder.name}/run.json"))
        persisted = json.loads(raw_record)
        if persisted != record:
            raise ValueError("Volume run.json differs from the returned record")
        (stage / "run.json").write_bytes(raw_record)

        digest = hashlib.sha256()
        with (stage / "run.log").open("wb") as log:
            for chunk in volume.read_file(f"{folder.name}/run.log"):
                log.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != persisted["log_sha256"]:
            raise ValueError("run.log SHA-256 differs from run.json")

        if record["kind"] == "evaluation":
            try:
                evaluation = b"".join(volume.read_file(f"{folder.name}/initial/evaluation.json"))
            except FileNotFoundError:
                if record["status"] == "completed":
                    raise
            else:
                (stage / "initial").mkdir()
                (stage / "initial/evaluation.json").write_bytes(evaluation)

        if folder.exists():
            for source in stage.rglob("*"):
                if source.is_file() and not filecmp.cmp(source, folder / source.relative_to(stage), shallow=False):
                    raise FileExistsError(f"Existing local copy differs: {folder / source.relative_to(stage)}")
        else:
            stage.rename(folder)


def _recover_command(run_id):
    return f"mkdir -p runs/replication && modal volume get {VOLUME_NAME} {run_id} runs/replication/"


def finish_local(record: dict, repo_root: Path, volume) -> None:
    """Save a completed measurement first, then retrieve the worker's diagnostic files."""
    run_id = record["run_id"]
    folder = repo_root / "runs/replication" / run_id
    measured = record["kind"] == "evaluation" and record["status"] == "completed"
    save_error = None
    if measured:
        out = repo_root / "results/midway" / f"li-initial-{run_id}.json"
        result = record["evaluation"]
        print(f"Initial accuracy {result['mean_accuracy']:.2%}, published {result['published_fraction']:.1%}. "
              f"GPU {record['gpu_name']}, wall time {record['wall_seconds']:.2f} seconds.")
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists() and json.loads(out.read_text(encoding="utf-8")) != record:
                raise FileExistsError(out)
            _write_json(out, record)
            print(f"Measurement record: {out}")
        except (OSError, ValueError) as error:
            save_error = error  # Still recover the Volume copy.

    try:
        download_attempt(record, folder, volume)
        print(f"Verified attempt folder: {folder}")
    except Exception as error:
        message = f"Could not download and verify {folder}: {error}\n{_recover_command(run_id)}"
        if not measured:
            detail = record.get("error", record.get("stopping_point", ""))
            raise RuntimeError(f"{record['kind']} FAIL. {detail}\n{message}") from error
        print(f"Warning: {message}", file=sys.stderr)
    if save_error is not None:
        raise RuntimeError(f"Could not save the completed measurement to {out}: {save_error}") from save_error
    if record["status"] != "completed":
        raise RuntimeError(f"{record['kind']} FAIL. {record['error']}\nLocal attempt folder: {folder}")
    if record["kind"] == "preflight":
        print(f"CPU preflight PASS — {record['stopping_point']}")


def rename_project(upstream):
    from pathlib import Path

    path = Path(upstream) / "envs/verl-main-cu129/pyproject.toml"
    original = path.read_bytes()
    old = b'[project]\nname = "tim-verl-env"\n'
    if original.count(old) != 1:
        raise ValueError(f'{path}: expected [project] name = "tim-verl-env" in the pinned checkout')
    path.write_bytes(original.replace(old, b'[project]\nname = "verl-main-cu129-env"\n'))


REMOTE_UPSTREAM = Path("/opt/li")
REMOTE_REPO = Path("/opt/offpolicy-maxrl")
VOLUME_MOUNT = Path("/li-runs")
CONFIG_PATH = "configs/replication/li_frozen.json"
WRAPPER_PATH = "scripts/li_replication.py"


def build_image(repo):
    import inspect

    revision = json.loads((repo / CONFIG_PATH).read_text())["upstream_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("upstream_commit must contain a full Git SHA")
    # Embed just the guarded rename, so a launcher or wrapper edit doesn't invalidate the build.
    patch = inspect.getsource(rename_project) + f"\nrename_project({str(REMOTE_UPSTREAM)!r})"
    return (
        modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
        .entrypoint([])
        .apt_install("git", "ca-certificates")
        .pip_install("uv==0.11.32")
        .run_commands(f"git clone -q https://github.com/Skylanding/Training-Inference-Mismatch.git {REMOTE_UPSTREAM}"
                      f" && git -C {REMOTE_UPSTREAM} checkout --detach {revision}")
        .dockerfile_commands("RUN " + json.dumps(["python", "-c", patch]))
        .run_commands(
            f"cd {REMOTE_UPSTREAM} && bash envs/verl-main-cu129/build_env.sh && uv cache clean",
            f"cd {REMOTE_UPSTREAM} && .venv-tim/bin/python scripts/build_countdown_data.py",
        )
        .add_local_file(repo / WRAPPER_PATH, str(REMOTE_REPO / WRAPPER_PATH))
        .add_local_file(repo / CONFIG_PATH, str(REMOTE_REPO / CONFIG_PATH))
    )


try:
    import modal
except ModuleNotFoundError as exc:
    if exc.name != "modal":
        raise
    modal = None


if modal is not None:
    app = modal.App("li-initial")
    # Only the launching process builds the image graph from laptop files.
    image = build_image(Path(__file__).resolve().parents[1]) if modal.is_local() else None
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    resources = dict(image=image, cpu=8, memory=32768, timeout=3600, volumes={str(VOLUME_MOUNT): volume})

    @app.function(**resources, gpu=["H100", "A100-80GB"])
    def evaluate_initial(hourly_usd=None):
        return run_attempt(REMOTE_UPSTREAM, REMOTE_REPO / WRAPPER_PATH, VOLUME_MOUNT,
                           volume.commit, hourly_usd=hourly_usd)

    @app.function(**resources)
    def preflight_initial(hourly_usd=None):
        return run_attempt(REMOTE_UPSTREAM, REMOTE_REPO / WRAPPER_PATH, VOLUME_MOUNT,
                           volume.commit, preflight=True, hourly_usd=hourly_usd)

    @app.local_entrypoint()
    def main(preflight: bool = False, hourly_usd: float | None = None):
        """Run the CPU preflight or measure initial accuracy. hourly_usd is the total machine rate."""
        validate_hourly_usd(hourly_usd)
        worker = preflight_initial if preflight else evaluate_initial
        try:
            record = worker.remote(hourly_usd)
            if record is None:
                raise RuntimeError("Worker returned no attempt record")
        except BaseException:
            print(f"No attempt record received. Find the newest matching UTC attempt on the Volume:\n"
                  f"modal volume ls {VOLUME_NAME}\n"
                  f"Replace RUN_ID in this command with its directory name:\n{_recover_command('RUN_ID')}",
                  file=sys.stderr)
            raise
        finish_local(record, Path(__file__).resolve().parents[1], volume)
        return record


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply the pinned upstream installation fix.")
    parser.add_argument("--rename-project", required=True, type=Path, metavar="CHECKOUT")
    rename_project(parser.parse_args().rename_project)
