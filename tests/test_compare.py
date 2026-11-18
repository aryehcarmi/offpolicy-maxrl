from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from maxrl import compare
from maxrl.config import load_config
from maxrl.data import Puzzle, digest, save_manifest
from maxrl.matrix import METHODS, PRIMARY, SEEDS
from maxrl.prompts import prompt_metadata


def test_primary_comparison_from_frozen_shards(tmp_path):
    configs = Path(__file__).parents[1] / "configs"
    puzzles = [Puzzle(str(i), [1, 2, 3] if i % 2 else [1, 2, 3, 4], i) for i in range(256)]
    save_manifest(tmp_path, {"test": puzzles}, 256, 623, "fixture")
    manifest = tmp_path / "manifest.json"
    cfg = asdict(load_config(configs / "main.json"))
    runtime = {"dtype": "float32", "generation_batch": 8, "score_batch": 1}
    frozen = {"version": 3, "prompt": prompt_metadata(), "manifest_sha256": digest(manifest), "main_config": cfg, "runtime": runtime,
              "main_seeds": SEEDS, "primary_contrast": PRIMARY, "rates": dict.fromkeys(METHODS, .001)}
    (tmp_path / "frozen.json").write_text(json.dumps(frozen))
    for j, seed in enumerate(SEEDS["count"]):
        for method in ("count", "snis"):
            name = f"{method}-{seed}"
            path = tmp_path / "main" / name
            (path / "test").mkdir(parents=True)
            config = dict(cfg, method=method, seed=seed, lr=.001)
            run = dict(config=config, prompt=prompt_metadata(), manifest_sha256=digest(manifest),
                       runtime=dict(runtime, device=f"cuda:{j % 2}", hourly_usd=2+j))
            identity = f"{name}/step-000128"
            (path / "run.json").write_text(json.dumps(dict(run, run_id=name, initial_checkpoint_id="initial")))
            (path / "test/run.json").write_text(json.dumps(dict(run, training_config=config,
                                                                runtime=dict(runtime, score_batch=32,
                                                                             device="cuda:7", hourly_usd=9),
                                                                checkpoint_id=identity, panel="test")))
            with (path / "test/responses.jsonl").open("w") as stream:
                for puzzle in puzzles:
                    for i in range(16):
                        row = {"puzzle_id": puzzle.puzzle_id, "prompt_ids": [int(puzzle.puzzle_id)],
                               "response_ids": [1], "sampler_logprobs": [-.5], "response_mask": [1],
                               "eos_mask": [1], "reward": int(method == "snis" and i < j),
                               "checkpoint_id": identity, "sample_id": f"{identity}/{puzzle.puzzle_id}/{i}",
                               "pool": f"{identity}/test", "schema_version": 1}
                        stream.write(json.dumps(row) + "\n")
    result = compare.compare(tmp_path, manifest, configs, 64, 1)
    expected = np.arange(5)/16
    np.testing.assert_allclose([r["snis_minus_count_pass@1"] for r in result["per_seed"]], expected)
    assert result["mean_snis_minus_count"] == .125
    width = 2.7764451051977987*expected.std(ddof=1)/np.sqrt(5)
    np.testing.assert_allclose(result["paired_training_seed_t_ci95"], [.125-width, .125+width])
    assert result["paired_puzzle_bootstrap_ci95"] == [.125, .125]
    assert result["numerical_runtime"] == runtime
    # Matched configs are insufficient when a manually edited job changes its runtime.
    for phase, child, changes in (
        ("training", "run.json", [("dtype", "bfloat16"), ("generation_batch", 4), ("score_batch", 2)]),
        ("evaluation", "test/run.json", [("dtype", "bfloat16"), ("generation_batch", 4)]),
    ):
        path = tmp_path / "main/snis-101" / child
        original = path.read_text()
        for key, value in changes:
            wrong = json.loads(original)
            wrong["runtime"][key] = value
            path.write_text(json.dumps(wrong))
            with pytest.raises(ValueError, match=f"{phase} numerical runtime"):
                compare.compare(tmp_path, manifest, configs, 64, 1)
        wrong = json.loads(original)
        del wrong["runtime"]
        path.write_text(json.dumps(wrong))
        with pytest.raises(ValueError, match=f"{phase} runtime: Missing"):
            compare.compare(tmp_path, manifest, configs, 64, 1)
        path.write_text(original)
    # Test prompt tokens can match even when the training prompt was different.
    for phase, child in (("training", "run.json"), ("evaluation", "test/run.json")):
        path = tmp_path / "main/snis-101" / child
        original = path.read_text()
        wrong = json.loads(original)
        wrong["prompt"]["sha256"] = "0" * 64
        path.write_text(json.dumps(wrong))
        with pytest.raises(ValueError, match=f"{phase} prompt"):
            compare.compare(tmp_path, manifest, configs, 64, 1)
        path.write_text(original)
    # A test shard with the correct number of rows but the wrong recorded seed
    # must be rejected before a replacement can silently change the comparison.
    metadata = tmp_path / "main/count-101/test/run.json"
    wrong = json.loads(metadata.read_text())
    wrong["config"]["seed"] = 102
    metadata.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="configuration"):
        compare.compare(tmp_path, manifest, configs, 64, 1)


def test_paired_puzzle_bootstrap_preserves_seed_histories():
    # The puzzle effect varies but is identical on every training seed.
    differences = np.tile([-.25, .25, -.5, .5], (5, 1))
    report = compare.intervals(differences, np.array([3, 3, 4, 4]), 2000, 52)
    assert report["paired_training_seed_t_ci95"] == [0., 0.]
    lower, upper = report["paired_puzzle_bootstrap_ci95"]
    assert lower < 0 < upper
