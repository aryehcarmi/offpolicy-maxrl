"""Replay independent one-step Count updates on a pilot's saved training batches."""

import argparse
import csv
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .config import Config, Runtime
from .metrics import ResourceTable, dump_json, hardware
from .model import checkpoint_id, load_policy, sequence_logprobs
from .prompts import validate_prompt
from .schema import read_responses
from .train import accumulate_gradient, chunks, sampler_logprob


RATES = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 1e-3)
LOW_RATES = (5e-8, 1e-7, 1.5e-7, 2e-7, 3e-7, 5e-7, 1e-6)
GRIDS = {"original": RATES, "low": LOW_RATES}
QUANTILES = {
    "min": 0, "p025": .025, "p05": .05, "p25": .25, "p50": .5, "p75": .75,
    "p95": .95, "p975": .975, "max": 1,
}


def select_rate(rows, grid="original"):
    by_rate = {row["rate"]: row for row in rows}
    if len(by_rate) != len(rows) or set(by_rate) != set(GRIDS[grid]):
        raise ValueError("Selection requires exactly one row per specified rate")
    if grid == "original" and not .8 <= by_rate[1e-6]["realized_over_predicted"] <= 1.2:
        return None, "At 1e-6 the realized objective change was outside 80–120% of prediction."
    qualifying = [r["rate"] for r in rows if r["realized_over_predicted"] >= .5
                  and r["heldout_fraction_within_0p1"] >= .95]
    if not qualifying:
        return None, "No rate met both the objective and held-out log-weight thresholds."
    best = max(qualifying)
    if grid == "low" and by_rate[best]["rounding_projection_fraction"] < .99:
        return None, f"The largest eligible rate ({best:g}) retained less than 99% of the intended step."
    return best, None


def read_batch(pilot, run_id, step, config, identity):
    path = pilot / "responses" / f"train-{step:06d}.jsonl"
    groups = {}
    for row in read_responses(path):
        if row.pool != f"{run_id}/train/{step}" or row.checkpoint_id != identity:
            raise ValueError(f"Unexpected pool or sampler identity in {path}")
        group = groups.setdefault(row.puzzle_id, [])
        if group and group[0].prompt_ids != row.prompt_ids:
            raise ValueError(f"Inconsistent prompt tokens for {row.puzzle_id} in {path}")
        group.append(row)
    if (len(groups) != config.prompts_per_step
            or any(len(g) != config.responses_per_prompt for g in groups.values())):
        raise ValueError(f"Incomplete or malformed training batch in {path}")
    return list(groups.values())


@torch.no_grad()
def score_log_weights(model, rows, score_batch):
    values = []
    for part in chunks(rows, score_batch):
        scores = sequence_logprobs(model, part).tolist()
        values.extend(score - sampler_logprob(row) for score, row in zip(scores, part, strict=True))
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite replay log weight")
    return result


@torch.no_grad()
def rounding_stats(parameters, initial, rate):
    """Compare rounded FP32 displacements with -rate times the original gradient in FP64."""
    device = parameters[0].device
    totals = torch.zeros(5, dtype=torch.float64, device=device)
    for param, saved in zip(parameters, initial, strict=True):
        if param.grad is None:
            continue
        flat, grad, base = param.flatten(), param.grad.flatten(), saved.flatten()
        for start in range(0, grad.numel(), 1 << 20):
            block = slice(start, start + (1 << 20))
            actual = flat[block].double() - base[block].double()
            intended = grad[block].double().mul_(-rate)
            totals[0] += torch.dot(intended, intended)
            totals[1] += torch.dot(actual, intended)
            totals[2] += torch.dot(actual, actual)
            totals[3] += torch.count_nonzero(intended)
            totals[4] += torch.count_nonzero(actual)
    ideal_sq, alignment, actual_sq, intended_count, changed_count = totals.tolist()
    if not math.isfinite(ideal_sq) or ideal_sq <= 0:
        raise FloatingPointError("A finite nonzero gradient is required for the replay")
    return {
        "rounding_projection_fraction": alignment / ideal_sq,
        "rounding_norm_fraction": math.sqrt(actual_sq / ideal_sq),
        "rounding_changed_fraction": changed_count / intended_count,
    }


def mean_weight(log_weights):
    if not len(log_weights):
        return None
    return math.exp(float(np.logaddexp.reduce(log_weights)) - math.log(len(log_weights)))


def success_weight_summary(logs, groups, group_count):
    """Estimate E[W | success], grouping the ratio-estimator residuals by prompt draw."""
    weights = np.exp(logs)
    n = len(weights)
    mean = float(np.mean(weights)) if n else None
    se = None
    if n and group_count > 1:
        residuals = np.bincount(groups, weights=weights - mean, minlength=group_count)
        se = math.sqrt(group_count / (group_count - 1) * np.dot(residuals, residuals)) / n
    return {
        "pooled_success_count": n, "pooled_prompt_groups": group_count,
        "pooled_mean_weight_success": mean, "pooled_mean_weight_success_se": se,
        "pooled_mean_weight_success_iid_se": float(np.std(weights, ddof=1) / math.sqrt(n)) if n > 1 else None,
    }


def measure_rates(model, batches, runtime, out, grid):
    first, heldout = batches[:2]
    score_batch = runtime.score_batch
    optimizer = torch.optim.SGD(model.parameters(), lr=GRIDS[grid][0], momentum=0, weight_decay=0)
    optimizer.zero_grad(set_to_none=True)
    stats, first_log_w = accumulate_gradient(model, first, "count", score_batch)
    norm = stats["gradient_norm"]
    if norm <= 0:
        raise FloatingPointError("Cannot test first-order predictions with a zero gradient")
    # The trainer divides by K within a nonempty group and averages over every prompt.
    # Keep the same successful rows and initial detached coefficients for each candidate.
    successful, baseline, coefficients, denominators = [], [], [], []
    weights = iter(first_log_w)
    for group in first:
        k = sum(row.reward for row in group)
        for row in group:
            log_w = next(weights)
            if row.reward:
                successful.append(row)
                baseline.append(log_w)
                denominators.append(k * len(first))
                coefficients.append(math.exp(log_w) / (k * len(first)))

    parameters = list(model.parameters())
    initial = [param.detach().clone() for param in parameters]
    heldout = [row for group in heldout for row in group]
    reward = np.array([row.reward for row in heldout], dtype=bool)
    table = []
    with ExitStack() as stack:
        raw = stack.enter_context((out / "heldout.jsonl").open("x"))
        csvfile = stack.enter_context((out / "rates.csv").open("x", newline=""))
        if grid == "low":
            successes = stack.enter_context((out / "successes.jsonl").open("x"))
            # Batch 2 successes retain their order in logs[reward]; only later successes need scoring.
            success_rows, success_steps, success_groups = [], [], []
            group_count = 0
            for step, batch in enumerate(batches[1:], 2):
                for group in batch:
                    for row in group:
                        if row.reward:
                            success_rows.append(row)
                            success_steps.append(step)
                            success_groups.append(group_count)
                    group_count += 1
            extra_successes = success_rows[int(reward.sum()):]
        for rate in GRIDS[grid]:
            with torch.no_grad():
                for param, base in zip(parameters, initial, strict=True):
                    param.copy_(base)
            optimizer.param_groups[0]["lr"] = rate
            optimizer.step()
            predicted = rate * norm ** 2
            updated = score_log_weights(model, successful, score_batch)
            delta = updated - baseline
            realized = math.fsum(w * d for w, d in zip(coefficients, delta, strict=True))
            # The separate integrated Count objective is sum_i W_i / (K * #prompts).
            try:
                count_change = math.fsum(
                    (math.exp(new) - math.exp(old)) / divisor
                    for new, old, divisor in zip(updated, baseline, denominators, strict=True)
                )
            except OverflowError:
                count_change = None
            logs = score_log_weights(model, heldout, score_batch)
            for row, log_w in zip(heldout, logs, strict=True):
                raw.write(json.dumps(dict(rate=rate, sample_id=row.sample_id, puzzle_id=row.puzzle_id,
                                          reward=row.reward, log_weight=log_w), allow_nan=False) + "\n")
            within = int(np.count_nonzero(np.abs(logs) <= .1))
            entry = {
                "rate": rate, "predicted_change": predicted, "realized_change": realized,
                "realized_over_predicted": realized / predicted,
                "count_weight_objective_change": count_change,
                **rounding_stats(parameters, initial, rate),
                "heldout_count": len(logs), "heldout_within_0p1": within,
                "heldout_fraction_within_0p1": within / len(logs),
                "mean_weight_success": mean_weight(logs[reward]),
                "mean_weight_failure": mean_weight(logs[~reward]),
            }
            if grid == "low":
                later_logs = score_log_weights(model, extra_successes, score_batch)
                success_logs = np.concatenate((logs[reward], later_logs))
                entry.update(success_weight_summary(success_logs, success_groups, group_count))
                for step, row, value in zip(success_steps, success_rows, success_logs, strict=True):
                    successes.write(json.dumps(dict(rate=rate, step=step, sample_id=row.sample_id,
                                                    puzzle_id=row.puzzle_id, reward=row.reward, log_weight=value),
                                               allow_nan=False) + "\n")
                successes.flush()
            quantiles = np.quantile(logs, list(QUANTILES.values()))
            entry.update((f"log_weight_{key}", value) for key, value in zip(QUANTILES, quantiles))
            if not table:
                writer = csv.DictWriter(csvfile, fieldnames=list(entry))
                writer.writeheader()
            writer.writerow(entry)
            csvfile.flush()
            raw.flush()
            table.append(entry)
            print(f"rate {rate:g}  realized/predicted {realized / predicted:.5g}"
                  f"  held out within ±0.1 {within}/{len(logs)}", flush=True)
    return norm, table


def replay(pilot, out, device="cuda", hourly_usd=None, grid="original"):
    pilot, out = Path(pilot), Path(out)
    source = json.loads((pilot / "run.json").read_text())
    validate_prompt(source.get("prompt"))
    config = Config(**source["config"])
    runtime = replace(Runtime(**source["runtime"]), device=device, hourly_usd=hourly_usd)
    if config.method != "count" or runtime.dtype != "float32" or runtime.score_batch != 1:
        raise ValueError("Replay requires a Count pilot with float32 forward and score batch one")
    identity = checkpoint_id(config.model, config.revision)
    if identity != source["initial_checkpoint_id"]:
        raise ValueError("Initial checkpoint identity conflicts with pilot record")
    steps = range(1, 17) if grid == "low" else (1, 2)
    batches, sample_ids = [], set()
    for step in steps:
        batch = read_batch(pilot, source["run_id"], step, config, identity)
        ids = {row.sample_id for group in batch for row in group}
        if sample_ids.intersection(ids):
            raise ValueError("Response sample identities overlap across training and held-out batches")
        sample_ids.update(ids)
        # Preserve even empty groups for the sampling uncertainty across all prompt draws.
        batches.append(batch if step <= 2 else [[r for r in group if r.reward] for group in batch])
    events = [json.loads(line) for line in (pilot / "metrics.jsonl").read_text().splitlines()]
    original, = [e for e in events if e["split"] == "train" and e["step"] == 1]
    out.mkdir(parents=True, exist_ok=False)
    timing = ResourceTable(runtime)
    try:
        torch.manual_seed(config.seed)
        model, _ = load_policy(config.model, device=device, dtype=runtime.dtype, revision=config.revision)
        started = timing.clock()
        norm, rows = measure_rates(model, batches, runtime, out, grid)
        timing.add("gradient_and_replay", started)
        selected, reason = select_rate(rows, grid)
        files = ["run.json", "metrics.jsonl", *(f"responses/train-{step:06d}.jsonl" for step in steps)]
        selection = ("Choose the largest rate with realized/predicted >= 0.5 and at least 95% of individual "
                     "batch-2 log weights inside [-0.1, 0.1].")
        if grid == "original":
            selection = "First require 0.8 <= realized/predicted <= 1.2 at 1e-6. " + selection
        else:
            selection += " Stop if the selected rate retains less than 99% of the intended step by projection."
        dump_json(out / "replay.json", {
            "pilot_run_id": source["run_id"], "initial_checkpoint_id": identity,
            "source_sha256": {name: hashlib.sha256((pilot / name).read_bytes()).hexdigest() for name in files},
            "hardware": hardware(runtime), "grid": grid, "rates": GRIDS[grid], "gradient_norm": norm,
            "pilot_gradient_norm": original["gradient_norm"], "chosen_rate": selected, "stop_reason": reason,
            "objective": "Sum over first-batch successes of initial detached W times current sequence log probability divided by K and by the total number of prompts. K is the number of successes on that prompt.",
            "prediction": "rate times squared L2 norm of the trainer's first-batch gradient",
            "count_weight_objective_change": "Auxiliary change in the sum of updated importance weights over first-batch successes divided by K and the number of prompts. Empty if outside float64 range.",
            "heldout": "Natural log of updated policy probability over recorded sampler probability for each response in the second batch, including sampled EOS. Quantiles use linear interpolation. Raw weights are averaged arithmetically within each reward class.",
            "rounding": "Let a = updated float32 parameters minus their initial values, and d = -rate times the initial loss gradient. Projection is dot(a,d)/dot(d,d), norm fraction is ||a||/||d||, changed fraction is nnz(a)/nnz(d). Differences and sums use float64.",
            "selection": selection,
            **({"pooled_success": "Arithmetic mean of importance weights over successful frozen-sampler responses "
                "in batches 2–16. Estimates the updated-to-original success probability ratio. Primary SE uses "
                "one cluster per (batch, puzzle), including empty-success groups, with G/(G-1) correction. "
                "The iid SE treats individual successes as independent."} if grid == "low" else {}),
        })
    finally:
        timing.save(out / "resources.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True, help="Original pilot output directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--hourly-usd", type=float)
    parser.add_argument("--grid", choices=GRIDS, default="original")
    args = parser.parse_args()
    replay(args.pilot, args.out, args.device, args.hourly_usd, args.grid)


if __name__ == "__main__":
    main()
