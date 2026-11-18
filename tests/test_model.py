"""Offline integration with a randomly initialized, one-layer Qwen3 and local tokenizer."""

import csv
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from maxrl.config import Runtime, load_config
from maxrl.data import Puzzle, save_manifest
from maxrl import evaluate, train as train_module
from maxrl.metrics import accuracy
from maxrl.model import checkpoint_id, load_policy, sequence_logprobs
from maxrl.rollout import sample_puzzle
from maxrl.prompts import prompt_metadata
from maxrl.schema import read_responses
from maxrl.train import train, update_policy


def samples(model, tok, n=16):
    return sample_puzzle(model, tok, Puzzle("1-2-3=6", [1, 2, 3], 6), n, checkpoint_id="test-base",
                         pool="toy", seed=2, microbatch=4, max_new_tokens=5)


def test_rollout_matches_teacher_forcing_and_masks(tiny):
    model, tok = load_policy(tiny, device="cpu")
    rows = samples(model, tok)
    assert all(r.prompt_ids[-1] == tok.convert_tokens_to_ids("no_thinking") for r in rows)
    assert any(r.eos_mask[-1] for r in rows) and any(not r.eos_mask[-1] for r in rows)
    assert all(len(r.response_ids) == 5 or r.eos_mask[-1] for r in rows)
    assert rows == samples(model, tok)  # explicit local RNG
    expected = [math.fsum(r.sampler_logprobs) for r in rows]
    padded = [replace(r, response_ids=r.response_ids + [2], eos_mask=r.eos_mask + [0],
                      response_mask=r.response_mask + [0], sampler_logprobs=r.sampler_logprobs + [0])
              for r in rows]
    actual = sequence_logprobs(model, padded)
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=2e-6, rtol=0)
    assert actual.requires_grad


def test_sgd_uses_success_sum_and_zero_prompt_and_detaches_ratios(tiny):
    model, tok = load_policy(tiny, device="cpu")
    rows = samples(model, tok, 3)
    weighted = []
    for row, weight in zip(rows[:2], [.5, 1.5]):
        lp = row.sampler_logprobs.copy()
        lp[0] -= math.log(weight)
        weighted.append(replace(row, sampler_logprobs=lp, reward=1))
    zero = [replace(rows[2], reward=0)]
    params = list(model.parameters())
    initial = [p.detach().clone() for p in params]
    scores = sequence_logprobs(model, weighted)
    # Two successes, two prompts. The second prompt contributes exactly zero.
    manual = (0.5 * scores[0] + 1.5 * scores[1]) / 4
    reference = torch.autograd.grad(manual, params)
    optimizer = torch.optim.SGD(params, lr=.02)
    stats, ratios = update_policy(model, optimizer, [weighted, zero], "count", 1)
    np.testing.assert_allclose(ratios, np.log([.5, 1.5, 1]), atol=2e-6)
    assert stats["gradient_norm"] > 0
    for p, old, gradient in zip(params, initial, reference):
        torch.testing.assert_close(p, old + .02 * gradient, atol=2e-7, rtol=2e-5)
    before = [p.detach().clone() for p in params]
    assert update_policy(model, optimizer, [zero], "count", 1)[0]["gradient_norm"] == 0
    assert all(torch.equal(p, prev) for p, prev in zip(params, before))


def test_pass_at_eight_for_sixteen_samples(tiny):
    model, tok = load_policy(tiny, device="cpu")
    row = samples(model, tok, 1)[0]
    successes = [replace(row, reward=1)] * 2 + [replace(row, reward=0)] * 14
    stats = accuracy([successes, [replace(row, reward=0)] * 16])
    assert stats["pass@1"] == 1/16
    assert stats["pass@8"] == pytest.approx((1 - math.comb(14, 8) / math.comb(16, 8)) / 2)


def test_trajectory_evaluation_and_sampler_refresh(tiny, tmp_path, monkeypatch):
    folder = tmp_path / "data"
    folder.mkdir()
    puzzles = [Puzzle(f"{i}-2-3={i+5}", [i, 2, 3], i + 5) for i in range(1, 6)]
    save_manifest(folder, {"train": puzzles[:3], "validation": puzzles[3:4], "test": puzzles[4:]},
                  5, 623, "tiny")
    manifest = folder / "manifest.json"
    preset = Path(__file__).resolve().parents[1] / "configs" / "pilot.json"
    cfg = replace(load_config(preset), model=tiny, revision=None, updates=3, prompts_per_step=2,
                  responses_per_prompt=2, validation_samples=2, test_samples=2, refresh_every=2,
                  eval_every=3, max_new_tokens=2, checkpoint_steps=[2])
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(asdict(cfg)))
    runtime = Runtime(device="cpu", hourly_usd=.25)
    run = tmp_path / "run"
    sampler_states = []
    generate_groups = train_module.generate_groups

    def training_groups(model, *args, **kwargs):
        is_training = "/train/" in kwargs["pool"]
        if is_training:
            sampler_states.append({name: p.detach().clone() for name, p in model.named_parameters()})
        groups = generate_groups(model, *args, **kwargs)
        if not is_training:
            return groups
        # Force one success per group so the test exercises real weight updates.
        return [[replace(row, reward=int(i == 0)) for i, row in enumerate(group)] for group in groups]

    monkeypatch.setattr(train_module, "generate_groups", training_groups)
    train(cfg, runtime, manifest, run)
    assert json.loads((run / "run.json").read_text())["prompt"] == prompt_metadata()
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [(m["step"], m["responses"]) for m in metrics if m["split"] == "validation"] == [(0, 2), (3, 2)]
    training = [m for m in metrics if m["split"] == "train"]
    assert [m["responses"] for m in training] == [4] * 3
    assert all(m["gradient_norm"] > 0 for m in training)
    assert len(sampler_states) == 3
    initial, stale, refreshed = sampler_states
    for name in initial:
        torch.testing.assert_close(stale[name], initial[name], rtol=0, atol=0)
    current, _ = load_policy(run / "checkpoints" / "step-000002", device="cpu")
    current_weights = dict(current.named_parameters())
    assert any(not torch.equal(current_weights[name], initial[name]) for name in initial)
    for name in initial:
        torch.testing.assert_close(refreshed[name], current_weights[name], rtol=0, atol=0)
    ids = [m["sampler_id"] for m in training]
    assert ids[0] == ids[1] == checkpoint_id(tiny)
    assert ids[2] == checkpoint_id(run / "checkpoints" / "step-000002")
    for step, identity in enumerate(ids, start=1):
        rows = list(read_responses(run / "responses" / f"train-{step:06d}.jsonl"))
        assert len(rows) == 4 and {row.checkpoint_id for row in rows} == {identity}
        weights = (run / "responses" / f"weights-{step:06d}.csv").read_text().splitlines()
        assert [v["sample_id"] for v in csv.DictReader(weights)] == [row.sample_id for row in rows]
    resources = {r["phase"]: r for r in csv.DictReader((run / "resources.csv").read_text().splitlines())}
    assert int(resources["total"]["responses"]) == 16
    assert float(resources["total"]["machine_usd"]) > 0
    test = tmp_path / "final-eval"
    monkeypatch.setattr(sys, "argv", [
        "evaluate", "--config", str(cfg_path), "--manifest", str(manifest), "--checkpoint",
        str(run / "checkpoints" / "step-000003"), "--panel", "test", "--out", str(test), "--device", "cpu",
    ])
    evaluate.main()
    assert json.loads((test / "metrics.json").read_text())["responses"] == 2
    assert len(list(read_responses(test / "responses.jsonl"))) == 2
    assert json.loads((test / "run.json").read_text())["prompt"] == prompt_metadata()
    marker = run / "checkpoints/step-000003/maxrl_checkpoint.json"
    checkpoint = json.loads(marker.read_text())
    assert checkpoint["prompt"] == prompt_metadata()
    checkpoint["prompt"]["sha256"] = "0" * 64
    marker.write_text(json.dumps(checkpoint))
    # Evaluation rejects the checkpoint before creating output or loading weights.
    with pytest.raises(ValueError, match="prompt metadata"):
        evaluate.main()


def test_presets_and_invalid_settings(tmp_path):
    configs = Path(__file__).resolve().parents[1] / "configs"
    for stage, updates in [("pilot", 16), ("screen", 16), ("tune", 96), ("main", 128)]:
        cfg = load_config(configs / f"{stage}.json")
        assert cfg.updates == updates
        assert (cfg.prompts_per_step, cfg.responses_per_prompt, cfg.max_new_tokens) == (32, 8, 377)
        assert cfg.lr is None or stage == "pilot"
    with pytest.raises(ValueError):
        load_config(configs / "pilot.json", lr=float("nan"))
    with pytest.raises(ValueError):
        load_config(configs / "pilot.json", method="exact")
    with pytest.raises(ValueError):
        Runtime(score_batch=0)


def test_initial_evaluation_saves_success_counts_without_training(tiny, tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    puzzles = [Puzzle("1-2-3=6", [1, 2, 3], 6), Puzzle("2-3-4=9", [2, 3, 4], 9)]
    save_manifest(data, {"validation": puzzles}, 2, 623, "fixture")
    preset = Path(__file__).parents[1] / "configs/pilot.json"
    cfg = replace(load_config(preset), model=tiny, revision=None,
                  validation_samples=8, max_new_tokens=2)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(asdict(cfg)))
    before = checkpoint_id(tiny)
    out = tmp_path / "initial-eval"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--config", str(config),
        "--manifest", str(data / "manifest.json"), "--out", str(out), "--device", "cpu"])
    evaluate.main()
    metadata = json.loads((out / "run.json").read_text())
    stats = json.loads((out / "metrics.json").read_text())
    assert metadata["checkpoint_id"] == before == checkpoint_id(tiny)
    assert metadata["training_config"] is None and metadata["prompt"] == prompt_metadata()
    assert stats["prompts"] == 2 and stats["responses"] == 16
    assert sum(stats["success_counts"].values()) == 2
    assert {"pass@1", "pass@8", "zero_success_frequency", "one_success_frequency"} <= stats.keys()
    assert len(list(read_responses(out / "responses.jsonl"))) == 16
    assert not (out / "checkpoints").exists()
