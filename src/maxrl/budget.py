"""Count scheduled responses and project machine cost using a completed Count pilot."""
import argparse
import csv
import json
import math
from pathlib import Path

from .config import load_config


def workload(configs: Path, reduced_controls: bool = False):
    rows = []
    for name, runs in (("pilot", 1), ("screen", 3), ("tune", 12), ("main", 13 if reduced_controls else 16)):
        cfg = load_config(configs / f"{name}.json")
        # The trainer evaluates at step 0, every eval_every steps, and at the last step.
        evaluations = 1 + math.ceil(cfg.updates / cfg.eval_every)
        rows.append(dict(phase=name, runs=runs, updates=runs * cfg.updates,
                         training=runs * cfg.updates * cfg.prompts_per_step * cfg.responses_per_prompt,
                         validation=runs * evaluations * 128 * cfg.validation_samples,
                         test=runs * 256 * cfg.test_samples if name == "main" else 0))
    diagnostic = {"midpoint": 8 * (64 + 64 + 128), "discovery": 32 * (3 * 256 + 512),
                  "confirmation": 32 * (3 * 256 + 512)}
    replication = {"initial_evaluation": 64 * 8, "training": 6 * 40 * 64 * 8, "final_evaluations": 6 * 64 * 8}
    total = sum(r[k] for r in rows for k in ("training", "validation", "test"))
    return {"controlled": rows, "diagnostic": diagnostic, "replication": replication,
            "total_responses": total + sum(diagnostic.values()) + sum(replication.values())}


def projection(plan, pilot, hourly_usd, diagnostic_usd, replication_usd):
    run = json.loads((pilot / "run.json").read_text())
    with (pilot / "resources.csv").open() as f:
        phases = {r["phase"]: r for r in csv.DictReader(f)}
    if (pilot / "failure.json").exists() or run["config"]["method"] != "count":
        raise ValueError("projection needs a completed uncapped Count pilot")
    training, evaluation = phases["training_generation"], phases["evaluation"]
    if (int(training["responses"]), int(evaluation["responses"])) != (4096, 2048):
        raise ValueError("pilot must have 4096 training and 2048 validation responses")
    if hourly_usd is None:
        hourly_usd = run["runtime"]["hourly_usd"]
    amounts = (hourly_usd, diagnostic_usd, replication_usd)
    if any(x is not None and (not math.isfinite(x) or x < 0) for x in amounts):
        raise ValueError("prices must be finite and nonnegative")
    rows = plan["controlled"]
    train_n = sum(r["training"] for r in rows)
    eval_n = sum(r["validation"] + r["test"] for r in rows)
    # Include the separate final evaluation processes. Rates assume the pilot's response
    # lengths and hardware; diagnostic backwards and replication are priced separately.
    processes = sum(r["runs"] for r in rows) + rows[-1]["runs"]
    phases_hours = {
        "training_generation": float(training["seconds"]) * train_n / 4096 / 3600,
        "evaluation": float(evaluation["seconds"]) * eval_n / 2048 / 3600,
        "training_update": float(phases["training_update"]["seconds"]) * sum(r["updates"] for r in rows) / 16 / 3600,
        "setup_io_other": float(phases["setup_io_other"]["seconds"]) * processes / 3600,
    }
    controlled_usd = sum(phases_hours.values()) * hourly_usd if hourly_usd is not None else None
    complete = all(x is not None for x in (controlled_usd, diagnostic_usd, replication_usd))
    total = controlled_usd + diagnostic_usd + replication_usd if complete else None
    return {"pilot_gpu": run["hardware"]["gpu"], "machine_hourly_usd": hourly_usd,
            "pilot_mean_training_tokens": int(training["generated_tokens"]) / 4096,
            "pilot_mean_eval_tokens": int(evaluation["generated_tokens"]) / 2048,
            "controlled_phase_hours": phases_hours, "controlled_usd": controlled_usd,
            "diagnostic_usd": diagnostic_usd, "replication_usd": replication_usd,
            "projected_core_usd": total, "ceiling_usd": 1500, "reserved_usd": 300,
            "headroom_before_reserve_usd": 1200 - total if complete else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--reduced-controls", action="store_true", help="budget scenario with two Fresh and one Uncorrected run")
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--hourly-usd", type=float)
    parser.add_argument("--diagnostic-usd", type=float, help="estimated all-in cost of all three diagnostic panels")
    parser.add_argument("--replication-usd", type=float, help="estimated cost of six upstream runs and seven evaluations")
    args = parser.parse_args()
    plan = workload(args.configs, args.reduced_controls)
    if args.pilot:
        plan["projection"] = projection(plan, args.pilot, args.hourly_usd, args.diagnostic_usd, args.replication_usd)
    print(json.dumps(plan, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
