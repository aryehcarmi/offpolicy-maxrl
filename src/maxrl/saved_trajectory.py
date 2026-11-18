"""Check and train controlled trajectories using a frozen pilot's saved training batches."""

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from .config import Config, Runtime
from .data import digest
from .metrics import ResourceTable, dump_json, hardware
from .model import checkpoint_id, load_policy, save_policy
from .prompts import validate_prompt
from .replay import read_batch
from .train import update_policy


RATES = (1.5e-7, 3e-7)
METHODS = ("count", "snis", "uncorrected")
VALIDATION_STEPS = (4, 8, 16)
# A prior A10 replay of the original step 2 differed by up to 0.01955 nats.
# Apply these bounds to every response and every norm, including near-zero norms.
LOG_WEIGHT_TOL = 0.05
LOG_NORM_TOL = 0.05


def load_training(pilot, device, hourly_usd):
    source = json.loads((pilot / "run.json").read_text())
    validate_prompt(source.get("prompt"))
    config = Config(**source["config"])
    runtime = replace(Runtime(**source["runtime"]), device=device, hourly_usd=hourly_usd)
    identity = checkpoint_id(config.model, config.revision)
    if (source["initial_checkpoint_id"] != identity or config.method != "count" or config.lr != 1e-3
            or config.seed != 17 or config.updates != 16 or config.refresh_every < config.updates
            or runtime.dtype != "float32" or runtime.score_batch != 1):
        raise ValueError("Expected the frozen seed-17, 16-update Count pilot at 1e-3, float32, score batch one")
    batches, seen = [], set()
    hashes = {"run.json": digest(pilot / "run.json"), "metrics.jsonl": digest(pilot / "metrics.jsonl")}
    for step in range(1, config.updates + 1):
        groups = read_batch(pilot, source["run_id"], step, config, identity)
        ids = {row.sample_id for group in groups for row in group}
        if seen.intersection(ids):
            raise ValueError(f"Reused sample IDs in training batch {step}")
        seen.update(ids)
        batches.append(groups)
        name = f"responses/train-{step:06d}.jsonl"
        hashes[name] = digest(pilot / name)
    return source, config, runtime, batches, hashes


def initial_model(config, runtime):
    torch.manual_seed(config.seed)
    return load_policy(config.model, revision=config.revision, device=runtime.device, dtype=runtime.dtype)


def compare_update(weight_file, groups, logs, norm, expected_norm):
    rows = [r for g in groups for r in g]
    with weight_file.open(newline="") as f:
        records = list(csv.DictReader(f))
    saved = {rec["sample_id"]: rec for rec in records}
    if len(records) != len(saved) or saved.keys() != {r.sample_id for r in rows}:
        raise ValueError(f"Response IDs in {weight_file} differ from the training shard")
    diffs = []
    for row, observed in zip(rows, logs, strict=True):
        rec = saved[row.sample_id]
        if rec["puzzle_id"] != row.puzzle_id or float(rec["reward"]) != row.reward:
            raise ValueError(f"Puzzle or reward mismatch for {row.sample_id}")
        diffs.append(abs(observed - float(rec["log_weight"])))
    if not np.all(np.isfinite(diffs)) or any(not math.isfinite(n) or n < 0 for n in (norm, expected_norm)):
        raise ValueError(f"Nonfinite comparison in {weight_file}")
    if norm == expected_norm:
        norm_err = 0.
    elif norm > 0 and expected_norm > 0:
        norm_err = abs(math.log(norm) - math.log(expected_norm))
    else:
        norm_err = None  # One norm is exactly zero.
    worst = int(np.argmax(diffs))
    return {
        "responses": len(rows), "saved_gradient_norm": expected_norm, "replay_gradient_norm": norm,
        "abs_log_gradient_norm_ratio": norm_err,
        "abs_log_weight_error_p50": float(np.median(diffs)),
        "abs_log_weight_error_p95": float(np.quantile(diffs, .95)),
        "abs_log_weight_error_max": float(diffs[worst]),
        "worst_sample_id": rows[worst].sample_id,
        "saved_worst_log_weight": float(saved[rows[worst].sample_id]["log_weight"]),
        "replay_worst_log_weight": float(logs[worst]),
        "passed": bool(norm_err is not None and norm_err <= LOG_NORM_TOL and diffs[worst] <= LOG_WEIGHT_TOL),
    }


def verify(pilot, out, source, config, runtime, batches, hashes, timing):
    events = [json.loads(line) for line in (pilot / "metrics.jsonl").read_text().splitlines()]
    events = [e for e in events if e["split"] == "train"]
    if [e["step"] for e in events] != list(range(1, config.updates + 1)):
        raise ValueError("Source pilot metrics must contain the complete training trajectory in order")
    model, _ = initial_model(config, runtime)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0, weight_decay=0)
    checks = []
    for step, (groups, event) in enumerate(zip(batches, events, strict=True), 1):
        expected_target = source["initial_checkpoint_id"] if step == 1 else f"{source['run_id']}/step-{step-1:06d}"
        if event["sampler_id"] != source["initial_checkpoint_id"] or event["target_id"] != expected_target:
            raise ValueError(f"Wrong model identity in original training event {step}")
        started = timing.clock()
        stats, logs = update_policy(model, optimizer, groups, "count", runtime.score_batch)
        filename = f"responses/weights-{step:06d}.csv"
        hashes[filename] = digest(pilot / filename)
        check = compare_update(pilot / filename, groups, logs, stats["gradient_norm"], event["gradient_norm"])
        timing.add("verify_training_update", started)
        checks.append({"step": step, **check})
        print(f"Verified step {step}: max log-weight error {check['abs_log_weight_error_max']:.6g}, "
              f"norm {stats['gradient_norm']:.6g}, passed={check['passed']}", flush=True)
    passed = all(check["passed"] for check in checks)
    dump_json(out / "verification.json", {
        "passed": passed, "pilot_run_id": source["run_id"], "initial_checkpoint_id": source["initial_checkpoint_id"],
        "source_sha256": hashes, "hardware": hardware(runtime), "max_abs_log_weight_error": LOG_WEIGHT_TOL,
        "max_abs_log_gradient_norm_ratio": LOG_NORM_TOL, "steps": checks,
    })
    if not passed:
        raise ValueError("Reproduction missed the prespecified tolerance; see verification.json")


def success_weight_stats(groups, logs):
    selected = [value for row, value in zip((r for g in groups for r in g), logs, strict=True) if row.reward]
    if not selected:
        return {"count": 0, "mean": None, "ess": 0., "quantiles": None}
    logs = np.asarray(selected)
    maximum = float(logs.max())
    scaled = np.exp(logs - maximum)
    # Scaling prevents overflow in the squared denominator of the Kish ESS.
    # Interpolating linearly after scaling preserves raw-weight percentile definitions.
    def raw(value):
        log_value = math.log(value) + maximum if value > 0 else -math.inf
        return math.exp(log_value) if log_value <= math.log(np.finfo(float).max) else None

    quantiles = np.quantile(scaled, [0., .25, .5, .75, .95, 1.])
    return {"count": len(logs), "mean": raw(scaled.mean()),
            "ess": float(scaled.sum() ** 2 / np.dot(scaled, scaled)),
            "quantiles": {k: raw(q) for k, q in zip(("min", "p25", "p50", "p75", "p95", "max"), quantiles)}}


def run(pilot, initial_eval, manifest, out, source, config, runtime, batches, hashes, timing):
    from .trajectory_probes import load_validation, logit_push, opening_id, opening_logprobs, validation_estimates

    if digest(manifest) != source["manifest_sha256"]:
        raise ValueError("Validation manifest does not match the original pilot")
    baseline_open = None
    for method in METHODS:
        for rate in RATES:
            model, tokenizer = initial_model(config, runtime)
            token = opening_id(tokenizer)
            if baseline_open is None:
                pools, prompts = load_validation(pilot, initial_eval, manifest, tokenizer)
                baseline_open = float(np.exp(opening_logprobs(model, prompts, token)).mean())
            optimizer = torch.optim.SGD(model.parameters(), lr=rate, momentum=0, weight_decay=0)
            start_time = datetime.now(timezone.utc).isoformat()
            for step, groups in enumerate(batches, 1):
                started = timing.clock()
                open_pre = np.exp(opening_logprobs(model, [g[0].prompt_ids for g in groups], token))
                stats, logs = update_policy(model, optimizer, groups, method, runtime.score_batch)
                step_stats = {
                    "method": method, "lr": rate, "step": step, "gradient_norm": stats["gradient_norm"],
                    "success_weights_pre_update": success_weight_stats(groups, logs),
                    "opening_logit_push_pre_update": logit_push(groups, method, logs, open_pre, token),
                    "validation_open_probability_post_update": float(np.exp(opening_logprobs(model, prompts, token)).mean()),
                }
                timing.add(f"{method}_{rate:g}_update_and_opening", started)
                if step in VALIDATION_STEPS:
                    started = timing.clock()
                    step_stats["validation_is_post_update"] = validation_estimates(model, pools, runtime.score_batch)
                    timing.add(f"{method}_{rate:g}_validation_rescoring", started)
                with (out / "trajectories.jsonl").open("a") as stream:
                    stream.write(json.dumps(step_stats, allow_nan=False) + "\n")
                print(f"{method} {rate:g} step {step}: gradient norm {stats['gradient_norm']:.6g}, "
                      f"opening probability {step_stats['validation_open_probability_post_update']:.6f}", flush=True)
            if method == "count" and rate == RATES[0]:
                run_id = uuid4().hex
                folder = out / "pilot"
                checkpoint_dir = f"step-{config.updates:06d}"
                save_policy(model, tokenizer, folder / "checkpoints" / checkpoint_dir, f"{run_id}/{checkpoint_dir}")
                selected_config = replace(config, lr=rate, method=method, checkpoint_steps=[config.updates])
                dump_json(folder / "run.json", {
                    "run_id": run_id, "started_utc": start_time, "config": asdict(selected_config),
                    "runtime": asdict(runtime), "hardware": hardware(runtime), "prompt": source["prompt"],
                    "initial_checkpoint_id": source["initial_checkpoint_id"], "manifest_sha256": source["manifest_sha256"],
                    "training_data": "16 saved frozen-sampler batches", "source_run_id": source["run_id"],
                })
            del optimizer, model, tokenizer

    dump_json(out / "summary.json", {
        "pilot_run_id": source["run_id"], "initial_checkpoint_id": source["initial_checkpoint_id"],
        "source_sha256": hashes,
        "validation_pools_sha256": {"initial_eval": digest(initial_eval / "responses.jsonl"),
                                    "pilot_step0": digest(pilot / "responses/validation-000000.jsonl")},
        "manifest_sha256": source["manifest_sha256"], "rates": RATES, "methods": METHODS,
        "updates": config.updates, "hardware": hardware(runtime), "validation_prompt_count": len(prompts),
        "opening_token_id": token, "initial_validation_open_probability": baseline_open,
        "weights": "Raw pre-update full-response importance weights on successful responses in the training batch. "
                   "ESS = (sum W)^2 / sum(W^2). Quantiles use linear interpolation in raw weight space.",
        "selected": {"method": "count", "lr": RATES[0], "run": "pilot", "checkpoint": "checkpoints/step-000016"},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "run"))
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--initial-eval", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--hourly-usd", type=float)
    args = parser.parse_args()
    if args.mode == "run" and (args.initial_eval is None or args.manifest is None):
        parser.error("run requires --initial-eval and --manifest")
    source, config, runtime, batches, hashes = load_training(args.pilot, args.device, args.hourly_usd)
    args.out.mkdir(parents=True, exist_ok=False)
    timing = ResourceTable(runtime)
    try:
        if args.mode == "verify":
            verify(args.pilot, args.out, source, config, runtime, batches, hashes, timing)
        else:
            run(args.pilot, args.initial_eval, args.manifest, args.out, source, config, runtime, batches, hashes, timing)
    finally:
        timing.save(args.out / "resources.csv")


if __name__ == "__main__":
    main()
