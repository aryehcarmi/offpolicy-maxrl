"""Render measured Count pilot outcomes and machine costs as a Markdown report."""

import argparse
import csv
import json
from pathlib import Path


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def percent(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def weight_at(row, distribution, quantile):
    item = (row[distribution] or {}).get(quantile)
    if item is None:
        return "—"
    return f"{item['weight']:.3g}" if item["weight"] is not None else f"exp({item['log_weight']:.4g})"


def outcome(row):
    return [row["step"], row["prompts"], row["responses"], percent(row.get("pass@1")), percent(row.get("pass@8")),
            percent(row["zero_success_frequency"]), percent(row["one_success_frequency"]),
            f"{row['generated_tokens'] / row['responses']:.1f}"]


def pilot_report(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "run.json").read_text())
    cfg = metadata["config"]
    if cfg["method"] != "count":
        raise ValueError("This report requires an uncapped Count run")
    metrics = directory / "metrics.jsonl"
    events = [json.loads(line) for line in metrics.read_text().splitlines()] if metrics.exists() else []
    training = sorted((e for e in events if e["split"] == "train"), key=lambda e: e["step"])
    validation = sorted((e for e in events if e["split"] == "validation"), key=lambda e: e["step"])
    hardware = metadata["hardware"]
    runtime = metadata["runtime"]
    failure_path = directory / "failure.json"
    if failure_path.is_file():
        failure = json.loads(failure_path.read_text())
        status = f"Stopped during {failure['phase']} at target step {failure['target_step']} ({failure['type']})."
    elif len(training) == cfg["updates"] and validation and validation[-1]["step"] == cfg["updates"]:
        status = f"Completed {cfg['updates']} updates."
    else:
        status = f"Partial run with {len(training)} logged updates."
    device = hardware["gpu"] or hardware["device"]
    fields = ["Step", "Prompts", "Responses", "Pass@1", "Pass@8", "K = 0", "K = 1", "Tokens/response"]
    report = [
        "# Uncapped Count pilot", f"{status} Seed {cfg['seed']}, SGD rate {cfg['lr']:g}, "
        f"{cfg['prompts_per_step']} prompts × {cfg['responses_per_prompt']} draws per update.",
        f"Model `{cfg['model']}`. Initial checkpoint `{metadata['initial_checkpoint_id']}`. "
        f"Device {device}, forward precision {runtime['dtype']}, PyTorch {hardware['torch']}.",
        "## Validation", table(fields, [outcome(row) for row in validation]),
        "## Training rollouts", table(fields, [outcome(row) for row in training]),
        "Pass rates average equally over prompts. K counts successes among the responses to one puzzle.",
        "## Pre-update importance weights",
        table(["Step", "All min", "All median", "All p95", "All max", "Successful median", "Successful p95",
               "Successful max"], [
            [row["step"], *[weight_at(row, "all_weights", q) for q in ("min", "p50", "p95", "max")],
             *[weight_at(row, "successful_weights", q) for q in ("p50", "p95", "max")]]
            for row in training
        ]),
        "Percentiles interpolate full-response log ratios and then exponentiate. "
        "A dash indicates an unavailable measurement.",
        "## Machine time and cost",
    ]
    path = directory / "resources.csv"
    if path.is_file():
        with path.open(newline="") as stream:
            resource_rows = list(csv.DictReader(stream))
        report.append(table(["Phase", "Seconds", "Responses", "Generated tokens", "USD"], [
            [r["phase"], f"{float(r['seconds']):.2f}", r["responses"], r["generated_tokens"],
             f"{float(r['machine_usd']):.4f}" if r["machine_usd"] else "—"] for r in resource_rows
        ]))
        rate = runtime["hourly_usd"]
        report.append(f"Estimated at USD {rate:g} per machine-hour over elapsed process time."
                      if rate is not None else "Machine price was unspecified.")
    else:
        report.append("The run has no saved resource table yet.")
    return "\n\n".join(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True, type=Path, help="Controlled Count run directory")
    args = parser.parse_args()
    print(pilot_report(args.pilot))


if __name__ == "__main__":
    main()
