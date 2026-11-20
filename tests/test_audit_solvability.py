import json

import pytest

from maxrl import audit_solvability as audit
from maxrl.data import Puzzle, digest, save_manifest
from maxrl.schema import Response, write_responses


@pytest.fixture
def saved_inputs(tmp_path, monkeypatch):
    targets = {"validation": 6, "discovery": 5, "confirmation": 7, "test": 99}
    panels = {
        name: [Puzzle(f"1-2-3={target}", [1, 2, 3], target)]
        for name, target in targets.items()
    }
    monkeypatch.setattr(audit, "RESERVE", {name: 1 for name in panels})
    first = Puzzle("2-3-4=9", [2, 3, 4], 9)
    repeated = Puzzle("3-3-3=99", [3, 3, 3], 99)
    last = Puzzle("2-3-5=10", [2, 3, 5], 10)
    panels["train"] = [first, repeated, last]
    save_manifest(tmp_path, panels, 7, 623, "fixture")
    manifest = tmp_path / "manifest.json"
    records = tmp_path / "pilot"
    records.mkdir()
    responses = records / "responses"
    config = {"updates": 2, "prompts_per_step": 2, "responses_per_prompt": 2}
    run = {"run_id": "pilot", "config": config, "manifest_sha256": digest(manifest)}
    run_path = records / "run.json"
    run_path.write_text(json.dumps(run))
    hashes = {"output/run.json": digest(run_path)}
    for step, puzzles in enumerate(((first, repeated), (repeated, last)), start=1):
        path = responses / f"train-{step:06d}.jsonl"
        pool = f"pilot/train/{step}"
        write_responses(path, [
            Response(p.puzzle_id, [1], [2], [-1.], [1], [0], 0,
                     "base", f"{pool}/{p.puzzle_id}/{i}", pool)
            for p in puzzles for i in range(config["responses_per_prompt"])
        ])
        hashes[f"output/responses/{path.name}"] = digest(path)
    attempt = {"attempt_id": "fixture", "manifest_sha256": digest(manifest), "sha256": hashes}
    (records / "attempt.json").write_text(json.dumps(attempt))
    return manifest, responses, records


def test_audit_counts_saved_uses_even_when_all_responses_failed(saved_inputs):
    groups = audit.audit(*saved_inputs)["groups"]
    assert groups["test"] == {
        "checked": 1, "solvable": 0, "verifier_accepted": 0,
        "unsolvable": [{"puzzle_id": "1-2-3=99", "numbers": [1, 2, 3], "target": 99}],
    }
    assert groups["pilot_training"] == {
        "checked": 4, "solvable": 2, "verifier_accepted": 2, "distinct_puzzles": 3,
        "prompts_per_batch": 2, "solvable_by_step": {1: 1, 2: 1},
        "repeated_puzzles": {"3-3-3=99": [1, 2]},
        "unsolvable": [
            {"step": step, "puzzle_id": "3-3-3=99", "numbers": [3, 3, 3], "target": 99}
            for step in (1, 2)
        ],
    }


def test_audit_rejects_modified_response_shard(saved_inputs):
    _, responses, _ = saved_inputs
    with (responses / "train-000002.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        audit.audit(*saved_inputs)


def test_audit_rejects_unverified_solver_witness(monkeypatch):
    monkeypatch.setattr(audit, "find_solution", lambda numbers, target: str(target))
    with pytest.raises(ValueError, match="Verifier rejected witness for 2-3-4=9"):
        audit.check_puzzles([Puzzle("2-3-4=9", [2, 3, 4], 9)])
