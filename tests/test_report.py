"""Render measured pilot metrics without loading model dependencies."""

import json

from maxrl.report import pilot_report


def test_partial_count_report_reports_observed_measurements(tmp_path):
    run = {
        "config": {"model": "fixture", "method": "count", "lr": .001, "seed": 17,
                   "prompts_per_step": 32, "responses_per_prompt": 8, "updates": 16},
        "hardware": {"gpu": None, "device": "cpu", "torch": "fixture"},
        "runtime": {"dtype": "float32", "hourly_usd": 2}, "initial_checkpoint_id": "base",
    }
    stats = {
        "step": 0, "split": "validation", "prompts": 128, "responses": 1024, "pass@1": .25,
        "pass@8": .5, "zero_success_frequency": .5, "one_success_frequency": .25,
        "generated_tokens": 10240,
    }
    quantiles = {q: {"log_weight": 0., "weight": 1.} for q in ("min", "p50", "p95", "max")}
    train = {**stats, "step": 1, "split": "train", "prompts": 32, "responses": 256,
             "all_weights": quantiles, "successful_weights": None}
    (tmp_path / "run.json").write_text(json.dumps(run))
    (tmp_path / "metrics.jsonl").write_text("\n".join(map(json.dumps, [stats, train])))
    (tmp_path / "resources.csv").write_text(
        "phase,seconds,responses,generated_tokens,machine_usd\ntotal,36,1280,20480,0.02\n"
    )
    result = pilot_report(tmp_path)
    assert "Partial run with 1 logged updates" in result
    assert "| 0 | 128 | 1024 | 25.00% | 50.00% | 50.00% | 25.00% | 10.0 |" in result
    assert "| 1 | 1 | 1 | 1 | 1 | — | — | — |" in result
    assert "| total | 36.00 | 1280 | 20480 | 0.0200 |" in result
