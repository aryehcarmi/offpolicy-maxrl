"""Verify pilot checkpoints on the remote filesystem and exercise both report CLIs locally."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from test_modal_countdown import import_launcher, MemoryVolume, REPO


def test_pilot_verifies_checkpoint_content_and_identity_on_volume(tmp_path):
    launcher = import_launcher()
    source = tmp_path / "scratch" / "output"
    source.mkdir(parents=True)
    run = {"run_id": "fixture", "config": {"checkpoint_steps": [8], "updates": 16}, "prompt": {"id": "prompt"}}
    (source / "run.json").write_text(json.dumps(run))
    for step in (8, 16):
        folder = source / "checkpoints" / f"step-{step:06d}"
        folder.mkdir(parents=True)
        marker = {"checkpoint_id": f"fixture/{folder.name}", "prompt": run["prompt"]}
        (folder / "maxrl_checkpoint.json").write_text(json.dumps(marker))
        for name in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"):
            (folder / name).write_bytes(name.encode())
    saved = tmp_path / "volume/attempt/output"
    shutil.copytree(source, saved)

    def verify():
        inventory = {str(path.relative_to(saved.parent)): launcher._digest(path)
                     for path in saved.rglob("*") if path.is_file()}
        return launcher.verify_pilot_checkpoints(source, saved, inventory, completed=True)

    assert verify() == ["fixture/step-000008", "fixture/step-000016"]
    weights = saved / "checkpoints/step-000008/model.safetensors"
    weights.write_bytes(b"short copy")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify()
    weights.write_bytes(b"model.safetensors")
    marker = saved / "checkpoints/step-000016/maxrl_checkpoint.json"
    marker.write_text(json.dumps({"checkpoint_id": "foreign/step-000016", "prompt": run["prompt"]}))
    shutil.copyfile(marker, source / marker.relative_to(saved))
    with pytest.raises(ValueError, match="identity"):
        verify()


def test_pilot_publishes_cli_outputs_and_replay_inputs_without_large_downloads(tmp_path):
    launcher = import_launcher()
    (tmp_path / "src").symlink_to(REPO / "src", target_is_directory=True)
    cfg = json.loads((REPO / "configs/pilot.json").read_text())
    run = {"config": cfg, "initial_checkpoint_id": "fixture", "runtime": {"dtype": "float32", "hourly_usd": 2},
           "hardware": {"device": "cuda", "gpu": "A10", "torch": "fixture"}}
    stats = {"pass@1": .25, "pass@8": .5, "zero_success_frequency": .5, "one_success_frequency": .25,
             "generated_tokens": 10240}
    events = [{"split": "validation", "step": step, "prompts": 128, "responses": 1024, **stats} for step in (0, 16)]
    events += [{"split": "train", "step": step, "prompts": 32, "responses": 256, **stats,
                "all_weights": None, "successful_weights": None} for step in range(1, 17)]
    files = {
        "run.log": b"run log", "gpu.csv": b"telemetry", "output/run.json": json.dumps(run).encode(),
        "output/metrics.jsonl": "\n".join(map(json.dumps, events)).encode(),
        "output/resources.csv": (b"phase,seconds,responses,generated_tokens,machine_usd\n"
                                 b"evaluation,36,2048,4096,.02\ntraining_generation,72,4096,8192,.04\n"
                                 b"training_update,36,0,0,.02\nsetup_io_other,36,0,0,.02\ntotal,180,6144,12288,.1\n"),
    }
    for name in launcher.BUDGET_CONFIGS:
        files[f"output/configs/{name}"] = (REPO / "configs" / name).read_bytes()
    attempt = "pilot-123"
    inventory = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    # These files exist only in the inventory. Requesting their bytes fails the test.
    inventory.update({"output/checkpoints/step-000016/model.safetensors": "weights",
                      "output/checkpoints/step-000008/config.json": "config", "output/responses/train-000001.jsonl": "shard"})
    record = {"attempt_id": attempt, "stage": "pilot", "status": "completed", "sha256": inventory}
    files["attempt.json"] = json.dumps(record).encode()
    volume = MemoryVolume({f"{attempt}/{name}": data for name, data in files.items()})
    folder = launcher.download_attempt(record, tmp_path, volume)
    assert not (folder / "output/checkpoints").exists()
    assert not (folder / "output/responses").exists()
    assert (folder / "gpu.csv").read_bytes() == files["gpu.csv"]
    summary = tmp_path / "results/midway/pilot"
    assert "Completed 16 updates" in (summary / "report.md").read_text()
    budget = json.loads((summary / "budget.json").read_text())
    assert budget["projection"]["controlled_usd"] > 0
    assert budget["projection"]["projected_core_usd"] is None
    assert all((summary / name).read_bytes() == files[f"output/{name}"] for name in launcher.PILOT_INPUTS)
    assert all((summary / "configs" / name).read_bytes() == files[f"output/configs/{name}"]
               for name in launcher.BUDGET_CONFIGS)
    assert json.loads((summary / "attempt.json").read_text()) == record
    assert launcher.download_attempt(record, tmp_path, volume) == folder


def test_recovery_uses_selective_download_without_dispatch(tmp_path, monkeypatch):
    pytest.importorskip("modal")
    launcher = import_launcher(with_modal=True)
    record = {"stage": "pilot", "status": "completed", "attempt_id": "pilot-id", "sha256": {}}
    volume = MemoryVolume({"pilot-id/attempt.json": json.dumps(record).encode()})
    monkeypatch.setattr(launcher, "volume", volume)
    calls = []

    def download(received, repo, source):
        assert received == record and source is volume
        calls.append(received["attempt_id"])
        return tmp_path

    def forbidden(*args):
        pytest.fail("Recovery tried to start a worker")

    monkeypatch.setattr(launcher, "download_attempt", download)
    monkeypatch.setattr(launcher.preflight, "remote", forbidden)
    monkeypatch.setattr(launcher.run_gpu, "remote", forbidden)
    launcher.main(recover_attempt="pilot-id")
    assert calls == ["pilot-id"]
    with pytest.raises(ValueError, match="Invalid attempt"):
        launcher.main(recover_attempt="../other")
    volume.content["pilot-id/attempt.json"] = json.dumps({**record, "status": "running"}).encode()
    with pytest.raises(ValueError, match="finished record"):
        launcher.main(recover_attempt="pilot-id")


def test_checkpoint_verification_failure_can_still_be_downloaded(tmp_path, monkeypatch):
    launcher = import_launcher()
    volume_root = tmp_path / "volume"
    monkeypatch.setattr(launcher, "_gpu_name", lambda: "A10")
    monkeypatch.setattr(launcher, "verify_manifest", lambda path: "fixture")
    monkeypatch.setattr(launcher, "_sample_gpu", lambda *args: nullcontext())

    def run(argv, repo, emit):
        if "maxrl.train" in argv:
            output = Path(argv[argv.index("--out") + 1])
            output.mkdir()
            (output / "run.json").write_text(json.dumps({
                "run_id": "missing-weights", "config": {"checkpoint_steps": [8], "updates": 16}, "prompt": {},
            }))

    monkeypatch.setattr(launcher, "_run", run)
    record = launcher.run_attempt(
        REPO, volume_root, lambda: None, stage="pilot",
        provenance={"git_commit": "a" * 40, "git_dirty": False},
        rates={"gpu_hour_cost_a10g": "1", "cpu_hour_cost": "0", "mem_gib_hour_cost": "0"},
    )
    assert record["status"] == "failed" and "verification failed" in record["error"]
    volume = MemoryVolume({str(p.relative_to(volume_root)): p.read_bytes()
                           for p in volume_root.rglob("*") if p.is_file()})
    local = launcher.download_attempt(record, tmp_path / "laptop", volume)
    assert record["error"] in (local / "run.log").read_text()
    assert not (tmp_path / "laptop/results").exists()
