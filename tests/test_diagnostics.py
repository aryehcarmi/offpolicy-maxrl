"""Independent checks of grouping, directional comparisons and full-model replay."""

import numpy as np
import pytest

from maxrl.prompts import prompt_metadata

from maxrl.diagnostics import _plan
from maxrl.gradient_metrics import grouped_updates, summarize


def test_predetermined_panel_counts():
    for panel, step, expected in [("midpoint", 16, 2048), ("discovery", 64, 40960),
                                  ("confirmation", 64, 40960)]:
        p, pools = _plan(panel, step)
        assert p * sum(count for _, count in pools.values()) == expected
    with pytest.raises(ValueError):
        _plan("confirmation", 16)


def test_prompt_balanced_exposure_and_fresh_groups():
    rewards = np.array([[1, 1, 0, 0], [0, 0, 1, 0]])
    weights = np.log([[1, 3, 1, 1], [1, 1, 4, 1]])
    scores = np.array([[[2.], [4.], [9.], [9.]], [[9.], [9.], [6.], [9.]]])
    b, exposure, failures = grouped_updates(rewards, weights, scores, 2, ["a", "b"],
                                  ("count", "snis", "fresh", "loo_cap"))
    assert not failures
    np.testing.assert_allclose(b["fresh"][:, 0], [1.5, 3])
    np.testing.assert_allclose(b["snis"][:, 0], [1.75, 3])
    np.testing.assert_allclose(b["count"][:, 0], [3.5, 12])
    np.testing.assert_allclose([v["exposure_snis"] for v in exposure], [1, 0, 0, 1])
    assert exposure[0]["successful_ess"] == pytest.approx(1.6)
    assert exposure[0]["raw_cap_fraction"] == 0.5
    assert exposure[0]["loo_final_cap_fraction"] == 0.5
    assert exposure[3]["exposure_loo_cap"] == 2


def test_global_norm_matching_and_zero_batches():
    a = np.array([[0., 0.], [2., 0.], [0., 1.], [0., 0.]])
    b = np.array([[0., 0.], [4., 0.], [0., 1.], [1., 0.]])
    ref = np.tile([1., 0.], (6, 1))
    report = summarize({"count": a, "snis": b}, ref, 40)
    count = report["methods"]["count"]
    assert count["zero_batch_fraction"]["estimate"] == 0.5
    assert count["matched_batch_fraction"]["estimate"] == 0.5
    assert count["norm_matched_mse"]["estimate"] == pytest.approx((9 + 2) / 2)
    assert count["raw_batch_mse"]["estimate"] == pytest.approx((1 + 1 + 2 + 1) / 4)
    assert count["cosine_of_mean"]["estimate"] == pytest.approx(2 / np.sqrt(5))
    diff = report["paired_differences"]["snis_minus_count"]
    assert diff["norm_matched_mse"]["estimate"] == pytest.approx(0)
    assert diff["matched_batches"] == 2
    zero = summarize({"snis": 0*b}, 0*ref, 4)["methods"]["snis"]
    assert zero["raw_batch_mse"]["estimate"] == 0
    assert zero["cosine_of_mean"]["estimate"] is None
    assert zero["mean_direction_status"] == "unresolved"


def test_bootstrap_references_vary_independently():
    a = np.tile([1., 0.], (8, 1))
    ref = np.tile([0., 1.], (16, 1))
    ref[::2, :] = [1., 0.]
    report = summarize({"count": a, "snis": a}, ref, 100)
    interval = report["methods"]["snis"]["cosine_of_mean"]["ci95"]
    assert interval[0] < interval[1]
    assert report["paired_differences"]["snis_minus_count"]["cosine_of_mean"]["ci95"] == [0, 0]


def test_rmsnorm_scores_and_accumulated_full_gradient(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from maxrl.diagnostics import score_rows
    from maxrl.full_gradients import aggregate, gram_coordinates, group_terms, store_gradient
    from maxrl.model import sequence_logprobs
    from maxrl.schema import Response

    torch.manual_seed(4)
    config = transformers.Qwen3Config(vocab_size=13, hidden_size=8, intermediate_size=12,
                                     num_hidden_layers=1, num_attention_heads=2,
                                     num_key_value_heads=1, head_dim=4)
    model = transformers.Qwen3ForCausalLM(config).eval()
    rows = [Response(puzzle_id=str(i // 2), prompt_ids=[1, 2], response_ids=[i + 1],
                     sampler_logprobs=[-3.2], response_mask=[1], eos_mask=[0], reward=int(i < 2),
                     checkpoint_id="tiny", sample_id=str(i), pool="tiny") for i in range(4)]
    vals = sequence_logprobs(model, rows)
    # First puzzle has two successes. The second contributes an exact zero.
    objective = (vals[0] + vals[1]) / 4
    truth = torch.autograd.grad(objective, tuple(model.parameters()))
    gold = torch.cat([v.reshape(-1) for v in truth]).detach().numpy()
    norm_param = next(i for i, v in enumerate(model.parameters()) if v is model.model.norm.weight)

    scores, ratios = score_rows(model, rows)
    np.testing.assert_allclose(ratios, vals.detach().numpy() + 3.2, atol=1e-6)
    np.testing.assert_allclose(scores.sum(axis=0) / 4, truth[norm_param].numpy(), atol=1e-6)
    assert not np.any(scores[2:])
    r = np.array([[1, 1], [0, 0]])
    terms = group_terms(rows, ratios.reshape(2, 2), r, 2, 0, "fresh")
    aggregate(model, terms, microbatch=1)
    store_gradient(tmp_path / "value", model)
    np.testing.assert_allclose(np.fromfile(tmp_path / "value", np.float32), gold, atol=1e-6)
    aggregate(model, [], microbatch=1)
    store_gradient(tmp_path / "zero", model)
    xyz = gram_coordinates([tmp_path / "value", tmp_path / "zero"], chunk_size=19)
    np.testing.assert_allclose(xyz @ xyz.T, [[gold @ gold, 0], [0, 0]], rtol=1e-6)


def test_pilot_checkpoint_fallback_and_saved_pipeline(tiny, tmp_path, monkeypatch):
    """Traverse all four stage CLIs on a reduced local trajectory and check provenance."""
    import csv
    import json
    import sys

    from maxrl import diagnostics
    from maxrl.data import Puzzle, digest, save_manifest
    from maxrl.model import checkpoint_id, load_policy, save_policy

    data = tmp_path / "data"
    data.mkdir()
    puzzles = [Puzzle(str(i), [1, 2, i], i + 3) for i in (3, 4)]
    save_manifest(data, {"discovery": puzzles}, 2, 623, "tiny")
    manifest = data / "manifest.json"
    run = tmp_path / "pilot"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({
        "config": {"model": tiny, "revision": None, "seed": 17, "method": "count"},
        "runtime": {"dtype": "float32"}, "run_id": "unit-run",
        "prompt": prompt_metadata(),
        "manifest_sha256": digest(manifest), "initial_checkpoint_id": checkpoint_id(tiny),
    }))
    model, tok = load_policy(tiny, device="cpu")
    save_policy(model, tok, run / "checkpoints/step-000008", "unit-run/step-000008")
    # Production midpoint counts are tested independently. Reduce only the
    # number of groups for a CPU test of real collection, replay and disk I/O.
    def small_plan(panel, step):
        assert (panel, step) == ("midpoint", 8)
        return 2, {"initial": (0, 8), "current": (8, 8), "reference": (8, 16)}

    monkeypatch.setattr(diagnostics, "_plan", small_plan)
    pool = tmp_path / "panel"
    def call(arguments):
        monkeypatch.setattr(sys, "argv", ["diagnostics", *map(str, arguments)])
        diagnostics.main()

    call(["collect", "--panel", "midpoint", "--run", run, "--manifest", manifest, "--out", pool,
          "--device", "cpu", "--hourly-usd", "1"])
    resources = {r["phase"]: r for r in csv.DictReader((pool / "resources.csv").read_text().splitlines())}
    assert int(resources["total"]["responses"]) == 64
    assert int(resources["total"]["generated_tokens"]) >= 64
    assert float(resources["total"]["machine_usd"]) > 0
    record = json.loads((pool / "manifest.json").read_text())
    assert record["step"] == 8
    assert record["prompt"] == prompt_metadata()
    ids = set()
    for name in record["pools"]:
        rows = diagnostics.load_rows(pool, record, name)
        assert not ids.intersection(r.sample_id for r in rows)
        ids.update(r.sample_id for r in rows)
    call(["score", "--pools", pool, "--device", "cpu"])
    call(["analyze", "--pools", pool, "--bootstrap", "5"])
    call(["full", "--pools", pool, "--device", "cpu", "--bootstrap", "5", "--scratch", tmp_path])
    results = json.loads((pool / "analysis-N8/summary.json").read_text())
    assert set(results["results"]) == {"initial", "current"}
    assert np.max(np.abs(results["results"]["current"]["log_weight_quantiles_0_50_95_100"])) < 1e-4
    assert (pool / "full-initial-N8/resources.csv").is_file()
    full = json.loads((pool / "full-initial-N8/summary.json").read_text())
    assert full["batch_indices"] == {"initial": [0], "reference": [0, 1]}
    assert full["metrics"]["reference_batches"] == 2
    run_metadata = json.loads((run / "run.json").read_text())
    run_metadata["prompt"]["sha256"] = "0" * 64
    (run / "run.json").write_text(json.dumps(run_metadata))
    with pytest.raises(ValueError, match="prompt metadata"):
        call(["collect", "--panel", "midpoint", "--run", run, "--manifest", manifest,
              "--out", tmp_path / "wrong-prompt", "--device", "cpu"])
    # A raw-shard edit is detected before attempting any new score replay.
    with (pool / "initial.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="changed"):
        diagnostics.load_rows(pool, record, "initial")


def test_overflow_is_local_to_the_affected_method_or_measurement():
    import json

    rewards = np.ones((1, 4), dtype=int)
    scores = np.ones((1, 4, 3), dtype=np.float32)
    for extreme in (500, 1000):
        lw = np.array([[extreme, -extreme] * 2], dtype=np.float64)
        b, exposure, errors = grouped_updates(rewards, lw, scores, 2, ["x"])
        fresh, _, ref_errors = grouped_updates(rewards, lw, scores, 2, ["x"], ("fresh",))
        assert not ref_errors
        assert "loo" in errors
        assert ("count" in errors) == (extreme == 1000)
        assert exposure[0]["loo_final_cap_fraction"] == 0.5
        assert exposure[0]["raw_cap_fraction"] == 0.5
        assert exposure[0]["log_d_over_k"] == pytest.approx(extreme - np.log(2))
        assert (exposure[0]["d_over_k"] is None) == (extreme == 1000)
        result = summarize(b, fresh["fresh"], 20, failures=errors)
        json.dumps(result, allow_nan=False)
        assert result["methods"]["loo_cap"]["raw_batch_mse"]["estimate"] == pytest.approx(0)
        assert "loo_cap_minus_snis" in result["paired_differences"]
        if extreme == 500:
            count = result["methods"]["count"]
            assert count["raw_batch_mse"]["point_overflow"]
            assert count["raw_batch_mse"]["estimate"] is None
            assert count["cosine_of_mean"]["estimate"] == pytest.approx(1)
            assert count["norm_matched_mse"]["estimate"] == pytest.approx(0)
            assert result["paired_differences"]["snis_minus_count"]["raw_batch_mse"]["point_overflow"]


def test_full_model_overflow_preserves_valid_comparisons(tiny, tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from maxrl import full_gradients as fg
    from maxrl.model import load_policy
    from maxrl.schema import Response

    model, _ = load_policy(tiny, device="cpu")
    rows = [Response("x", [1], [3], [-2.0], [1], [0], 1, "test", str(i), "test")
            for i in range(16)]
    pools = tmp_path / "pools"
    (pools / "scores").mkdir(parents=True)
    (pools / "manifest.json").write_text("{}")
    manifest = {"panel": "midpoint", "pools": {"initial": {}, "reference": {}}}
    monkeypatch.setattr(fg, "target_model", lambda *args: model)
    monkeypatch.setattr(fg, "_scored_manifest", lambda d: manifest)
    monkeypatch.setattr(fg, "load_rows", lambda d, m, n: rows[:8] if n == "initial" else rows)
    for name, length in [("initial", 8), ("reference", 16)]:
        weights = np.zeros((1, length))
        if name == "initial":
            weights[0, 0] = 500  # finite FP64 lambda, unrepresentable FP32 gradient
        np.savez(pools / f"scores/{name}.npz", log_weights=weights,
                 rewards=np.ones((1, length), dtype=int))
    args = SimpleNamespace(pools=pools, group_size=8, source="initial", hourly_usd=None,
                           device="cpu", dtype="float32", scratch=tmp_path, microbatch=4,
                           bootstrap=4, seed=10)
    fg.full(args)
    report = json.loads((pools / "full-initial-N8/summary.json").read_text())["metrics"]
    assert set(report["numerical_failures"]) == {"count"}
    assert report["numerical_failures"]["count"]["group"] == 0
    assert "loo_cap_minus_snis" in report["paired_differences"]
    assert np.isfinite(report["methods"]["snis"]["raw_batch_mse"]["estimate"])
    json.dumps(report, allow_nan=False)
