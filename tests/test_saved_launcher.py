"""Storage boundaries and command-line contracts for the frozen-batch experiment."""
import hashlib
import json
from pathlib import Path

import pytest

from test_modal_countdown import import_launcher, MemoryVolume, REPO


def test_saved_comparison_and_fresh_eval_use_separate_commands(tmp_path):
    launcher = import_launcher()
    manifest, pilot, initial, output = (tmp_path / n for n in ("manifest.json", "pilot", "initial", "result"))
    verify, = launcher.job_commands("saved-verify", REPO, manifest, output, 1.5, pilot_run=pilot)
    assert verify[1:4] == ["-m", "maxrl.saved_trajectory", "verify"]
    commands = launcher.job_commands("saved-train", REPO, manifest, output, 1.5,
                                     pilot_run=pilot, initial_eval=initial)
    run, = [c for c in commands if c[2] == "maxrl.saved_trajectory"]
    assert run[3] == "run" and run[run.index("--initial-eval") + 1] == str(initial)
    assert run[run.index("--pilot") + 1] == str(pilot)
    assert all("maxrl.train" not in c and "maxrl.evaluate" not in c for c in commands)
    assert any("maxrl.trajectory_probes" in c for c in commands)
    evaluate, = launcher.job_commands("saved-validation", REPO, manifest, output, 1.5, pilot_run=pilot)
    assert evaluate[2] == "maxrl.evaluate"
    assert Path(evaluate[evaluate.index("--checkpoint") + 1]) == pilot / "checkpoints/step-000016"
    for name, expected in {"--seed": "18", "--dtype": "float32", "--device": "cuda",
                           "--generation-batch": "8", "--score-batch": "1"}.items():
        assert evaluate[evaluate.index(name) + 1] == expected


@pytest.mark.parametrize("stage,target,required", [
    ("saved-verify", "saved-verify", {"verification.json", "resources.csv"}),
    ("saved-train", "saved-trajectories", {"trajectories.jsonl", "pilot/run.json"}),
    ("saved-validation", "pilot-lr-1.5e-7", {"metrics.json", "resources.csv"}),
    ("midpoint", "midpoint-lr-1.5e-7", {"manifest.json", "analysis-N8/exposure.csv",
                                         "analysis-N8/summary.json", "full-initial-N8/summary.json"}),
])
def test_saved_stages_publish_verified_small_files(tmp_path, stage, target, required):
    launcher = import_launcher()
    names = set(launcher.MIDPOINT_FILES if stage == "midpoint" else launcher.SAVED_OUTPUTS[stage])
    assert required <= names
    files = {f"output/{n}": (n + "\n").encode() for n in names}
    files["run.log"] = b"worker log"
    inventory = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    # The remote stub has no bytes for weights, raw sampled sequences or full gradients.
    inventory.update({"output/pilot/checkpoints/step-000016/model.safetensors": "remote",
                      "output/responses.jsonl": "remote", "output/reference.jsonl": "remote",
                      "output/scores/initial.npz": "remote"})
    attempt = f"{stage}-123"
    record = {"attempt_id": attempt, "status": "completed", "stage": stage,
              "source_learning_rate": 1.5e-7, "learning_rate": 1.5e-7,
              "pilot_attempt": "saved-train-123", "sha256": inventory}
    files["attempt.json"] = json.dumps(record).encode()
    volume = MemoryVolume({f"{attempt}/{n}": b for n, b in files.items()})
    local = launcher.download_attempt(record, tmp_path, volume)
    published = tmp_path / "results/midway" / target
    for n in required:
        assert (published / n).read_bytes() == (local / "output" / n).read_bytes()
    assert json.loads((published / "attempt.json").read_text()) == record
    assert launcher.download_attempt(record, tmp_path, volume) == local


def test_verification_is_bound_to_all_original_response_and_weight_files(tmp_path):
    launcher = import_launcher()
    pilot = tmp_path / "pilot-original/output"
    pilot.mkdir(parents=True)
    run = {"run_id": "original", "initial_checkpoint_id": "model@pinned", "prompt": {"id": "prompt"}}
    (pilot / "run.json").write_text(json.dumps(run))
    checked = {"run.json": "r", "metrics.jsonl": "m"}
    for step in range(1, 17):
        checked[f"responses/train-{step:06d}.jsonl"] = f"responses-{step}"
        checked[f"responses/weights-{step:06d}.csv"] = f"weights-{step}"
    (pilot.parent / "attempt.json").write_text(json.dumps({"sha256": {f"output/{k}": v for k, v in checked.items()}}))

    def save_source(name, stage, files, **metadata):
        folder = tmp_path / name
        output = folder / "output"
        output.mkdir(parents=True, exist_ok=True)
        for k, doc in files.items():
            (output / k).write_text(json.dumps(doc))
        record = {"attempt_id": name, "stage": stage, "status": "completed", "gpu_name": "NVIDIA A10",
                  "manifest_sha256": launcher.MANIFEST_SHA256,
                  "sha256": {f"output/{k}": launcher._digest(output / k) for k in files}, **metadata}
        (folder / "attempt.json").write_text(json.dumps(record))
        return folder.name

    initial = save_source("initial", "initial-eval", {"responses.jsonl": {}, "run.json": {
        "panel": "validation", "checkpoint_id": run["initial_checkpoint_id"], "prompt": run["prompt"],
        "manifest_sha256": launcher.MANIFEST_SHA256}})
    verified = {"passed": True, "pilot_run_id": run["run_id"], "initial_checkpoint_id": run["initial_checkpoint_id"],
                "source_sha256": checked.copy()}
    def check():
        name = save_source("verified", "saved-verify", {"verification.json": verified}, pilot_attempt=pilot.parent.name)
        return launcher._training_sources(tmp_path, name, initial, pilot, "A10")

    assert check() == tmp_path / initial / "output"
    verified["source_sha256"]["responses/weights-000016.csv"] = "wrong last step"
    with pytest.raises(ValueError, match="verification"):
        check()
    verified.update(source_sha256=checked, passed=False)
    with pytest.raises(ValueError, match="verification"):
        check()
