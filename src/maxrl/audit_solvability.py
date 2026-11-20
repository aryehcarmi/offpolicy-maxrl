"""Audit the four frozen Countdown panels and the original pilot's saved prompts."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path

from .data import RESERVE, Puzzle, digest, load_panel
from .schema import read_responses
from .solvability import find_solution
from .verifier import verify


def check_puzzles(puzzles: list[Puzzle]) -> dict:
    """Require a verifier-accepted witness for every positive result."""
    verified = 0
    unsolvable = []
    for puzzle in puzzles:
        expression = find_solution(puzzle.numbers, puzzle.target)
        if expression is None:
            unsolvable.append(asdict(puzzle))
        elif verify(f"<answer>{expression}</answer>", puzzle.numbers, puzzle.target):
            verified += 1
        else:
            raise ValueError(f"Verifier rejected witness for {puzzle.puzzle_id}: {expression}")
    return {"checked": len(puzzles), "solvable": verified, "verifier_accepted": verified,
            "unsolvable": unsolvable}


def saved_prompt_ids(responses: Path, run: dict, hashes: dict) -> dict[int, list[str]]:
    """Recover each batch's prompts from the saved responses, including zero-reward groups."""
    config = run["config"]
    batches = {}
    for step in range(1, config["updates"] + 1):
        filename = f"train-{step:06d}.jsonl"
        path = responses / filename
        if digest(path) != hashes[f"output/responses/{filename}"]:
            raise ValueError(f"Pilot response checksum mismatch: {path}")
        counts = Counter()
        for row in read_responses(path):
            if row.pool != f"{run['run_id']}/train/{step}":
                raise ValueError(f"Unexpected training pool in {path}: {row.pool}")
            counts[row.puzzle_id] += 1
        if (len(counts) != config["prompts_per_step"]
                or set(counts.values()) != {config["responses_per_prompt"]}):
            raise ValueError(f"Unexpected prompt or response count in {path}")
        batches[step] = list(counts)
    return batches


def audit(manifest: Path, responses: Path, pilot_records: Path) -> dict:
    attempt = json.loads((pilot_records / "attempt.json").read_text())
    hashes = attempt["sha256"]
    run_path = pilot_records / "run.json"
    if digest(run_path) != hashes["output/run.json"]:
        raise ValueError("Pilot run metadata checksum mismatch")
    run = json.loads(run_path.read_text())
    manifest_hash = digest(manifest)
    if manifest_hash != run["manifest_sha256"] or manifest_hash != attempt["manifest_sha256"]:
        raise ValueError("Split manifest differs from the recorded pilot manifest")

    groups = {}
    for name, size in RESERVE.items():
        puzzles = load_panel(manifest, name)
        if len(puzzles) != size:
            raise ValueError(f"Expected {size} puzzles in {name}")
        groups[name] = check_puzzles(puzzles)

    batches = saved_prompt_ids(responses, run, hashes)
    steps_by_id = defaultdict(list)
    for step, ids in batches.items():
        for puzzle_id in ids:
            steps_by_id[puzzle_id].append(step)
    train = {p.puzzle_id: p for p in load_panel(manifest, "train") if p.puzzle_id in steps_by_id}
    missing = steps_by_id.keys() - train.keys()
    if missing:
        raise ValueError(f"Saved puzzles missing from training split: {sorted(missing)}")

    # Count each saved use, including any puzzle drawn again in a later batch.
    by_step = {}
    unsolvable = []
    for step, ids in batches.items():
        result = check_puzzles([train[puzzle_id] for puzzle_id in ids])
        by_step[step] = result["solvable"]
        unsolvable.extend({"step": step, **puzzle} for puzzle in result["unsolvable"])
    verified = sum(by_step.values())
    groups["pilot_training"] = {
        "checked": sum(map(len, batches.values())), "solvable": verified, "verifier_accepted": verified,
        "unsolvable": unsolvable, "distinct_puzzles": len(train),
        "repeated_puzzles": {pid: steps for pid, steps in steps_by_id.items() if len(steps) > 1},
        "prompts_per_batch": run["config"]["prompts_per_step"], "solvable_by_step": by_step,
    }
    return {"manifest_sha256": manifest_hash, "pilot_attempt": attempt["attempt_id"],
            "pilot_run_id": run["run_id"], "groups": groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="frozen split manifest")
    parser.add_argument("--responses", type=Path, required=True, help="pilot output/responses directory")
    parser.add_argument("--pilot-records", type=Path, default=Path("results/midway/pilot-lr-0.001"),
                        help="directory containing the original pilot's attempt.json and run.json")
    args = parser.parse_args()
    print(json.dumps(audit(args.manifest, args.responses, args.pilot_records), indent=2))


if __name__ == "__main__":
    main()
