"""Estimate the prespecified final test contrast from the five paired seed histories."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from .config import load_config, numerical_runtime
from .data import digest, load_panel
from .matrix import PRIMARY, SEEDS, load_freeze
from .prompts import validate_prompt
from .schema import read_responses


T975_DF4 = 2.7764451051977987


def test_counts(root, frozen, method, seed, puzzle_index, prompts):
    directory = root / "main" / f"{method}-{seed}"
    train = json.loads((directory / "run.json").read_text())
    evaluation = json.loads((directory / "test/run.json").read_text())
    cfg = dict(frozen["main_config"], method=method, seed=seed, lr=frozen["rates"][method])
    identity = f'{train["run_id"]}/step-{cfg["updates"]:06d}'
    if (train["config"] != cfg or evaluation["training_config"] != cfg or evaluation["config"] != cfg
            or train["manifest_sha256"] != frozen["manifest_sha256"]
            or evaluation["manifest_sha256"] != frozen["manifest_sha256"]
            or evaluation["checkpoint_id"] != identity or evaluation["panel"] != "test"):
        raise ValueError(f"{directory}: wrong trajectory, evaluation panel, manifest, or configuration")
    for phase, metadata in (("training", train), ("evaluation", evaluation)):
        try:
            validate_prompt(metadata.get("prompt"))
        except ValueError as error:
            raise ValueError(f"{directory}: {phase} prompt: {error}") from error
        is_evaluation = phase == "evaluation"
        try:
            actual = numerical_runtime(metadata.get("runtime"), evaluation=is_evaluation)
            expected = numerical_runtime(frozen["runtime"], evaluation=is_evaluation)
        except ValueError as error:
            raise ValueError(f"{directory}: {phase} runtime: {error}") from error
        if actual != expected:
            raise ValueError(f"{directory}: {phase} numerical runtime differs from freeze: "
                             f"expected {expected}, recorded {actual}")
    counts = np.zeros(len(puzzle_index), dtype=int)
    draws = np.zeros_like(counts)
    for row in read_responses(directory / "test/responses.jsonl"):
        if (row.puzzle_id not in puzzle_index or row.checkpoint_id != identity
                or row.pool != f"{identity}/test"):
            raise ValueError(f"{directory}: unexpected puzzle, checkpoint or pool in response shard")
        if prompts.setdefault(row.puzzle_id, row.prompt_ids) != row.prompt_ids:
            raise ValueError(f"{directory}: prompt tokens differ for {row.puzzle_id}")
        index = puzzle_index[row.puzzle_id]
        draws[index] += 1
        counts[index] += int(row.reward)
    if np.any(draws != cfg["test_samples"]):
        raise ValueError(f"{directory}: incomplete or unbalanced final test panel")
    return counts, identity, train["initial_checkpoint_id"]


def intervals(differences, sizes, repeats, seed):
    """Rows are paired training seeds, columns are matched test puzzle IDs."""
    seed_effect = differences.mean(axis=1)
    point = float(seed_effect.mean())
    half_width = T975_DF4 * float(seed_effect.std(ddof=1)) / math.sqrt(5)
    # Averaging the seed dimension first holds these five trajectories fixed.
    # Paired resampling uses the same puzzle indices across both methods and all seeds.
    by_puzzle = differences.mean(axis=0)
    bootstrap = np.zeros(repeats)
    rng = np.random.default_rng(seed)
    for size in np.unique(sizes):
        stratum = by_puzzle[sizes == size]
        bootstrap += rng.choice(stratum, size=(repeats, len(stratum))).sum(axis=1) / len(by_puzzle)
    return {"mean_snis_minus_count": point, "paired_training_seed_t_ci95": [point-half_width, point+half_width],
            "paired_puzzle_bootstrap_ci95": np.quantile(bootstrap, [.025, .975]).tolist()}


def compare(root, manifest, configs, repeats, seed):
    if repeats < 1 or seed < 0:
        raise ValueError("Bootstrap repetitions must be positive and seed nonnegative")
    freeze_path = root / "frozen.json"
    frozen = load_freeze(freeze_path, digest(manifest), asdict(load_config(configs / "main.json")))
    puzzles = load_panel(manifest, "test")
    puzzle_index = {p.puzzle_id: i for i, p in enumerate(puzzles)}
    n = frozen["main_config"]["test_samples"]
    if len(puzzle_index) != 256 or len(puzzles) != 256 or n != 16:
        raise ValueError("Expected 256 distinct test puzzles with 16 responses each")
    sizes = np.array([len(p.numbers) for p in puzzles])
    rows, differences, origins, checkpoints = [], [], set(), set()
    prompts = {}
    for run_seed in SEEDS["count"]:
        row = {"seed": run_seed}
        rates = {}
        for method in ("count", "snis"):
            counts, identity, origin = test_counts(root, frozen, method, run_seed, puzzle_index, prompts)
            if identity in checkpoints:
                raise ValueError("Final checkpoint identity reused across trajectories")
            checkpoints.add(identity)
            origins.add(origin)
            rates[method] = counts / n
            row[method] = {"checkpoint_id": identity, "pass@1": float(rates[method].mean()),
                           "pass@8": float(np.mean([1-math.comb(n-int(k), 8)/math.comb(n, 8) for k in counts]))}
        difference = rates["snis"] - rates["count"]
        differences.append(difference)
        row["snis_minus_count_pass@1"] = float(difference.mean())
        rows.append(row)
    if len(origins) != 1:
        raise ValueError("Primary runs must share the same initial checkpoint")
    return {"primary_contrast": PRIMARY, "units": "probability", "per_seed": rows,
            "numerical_runtime": frozen["runtime"],
            "prompt": frozen["prompt"],
            "test_puzzles": len(puzzles), "responses_per_puzzle": n,
            "initial_checkpoint_id": origins.pop(), "manifest_sha256": frozen["manifest_sha256"],
            "freeze_sha256": digest(freeze_path), "bootstrap_repeats": repeats, "bootstrap_seed": seed,
            **intervals(np.stack(differences), sizes, repeats, seed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/matrix"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=10623, help="puzzle bootstrap seed")
    args = parser.parse_args()
    print(json.dumps(compare(args.root, args.manifest, args.configs, args.bootstrap, args.seed),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
