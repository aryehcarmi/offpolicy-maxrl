"""Build the midpoint report's numerical tables from the published run files."""

import argparse
import csv
import json
import re
from pathlib import Path

from .midpoint_numbers import midpoint_section
from .report import percent, table
from .report_probes import supporting_section


PILOT_METRICS = "pilot-lr-0.001/metrics.jsonl"
RUN_LABELS = {
    "initial-eval": "Initial model evaluation",
    "pilot-lr-0.001": "Original Count pilot",
    "replay/original": "First-update replay, original rates",
    "replay/low": "First-update replay, low rates",
    "saved-verify": "Reproduction on saved batches",
    "saved-trajectories": "Six trajectories and probes",
    "pilot-lr-1.5e-7": "Fresh Count validation",
    "midpoint-lr-1.5e-7": "Midpoint diagnostics",
}


def _source(path):
    return f"[{Path(path).name}]({path})"


def _validation_section(midway):
    pilot = [json.loads(line) for line in (midway / PILOT_METRICS).read_text().splitlines()]
    by_step = {(row["split"], row["step"]): row for row in pilot}
    initial = "initial-eval/metrics.json"
    fresh = "pilot-lr-1.5e-7/metrics.json"
    samples = [
        ("Initial model, standalone", initial, json.loads((midway / initial).read_text())),
        ("Initial model, pilot step 0", PILOT_METRICS, by_step["validation", 0]),
        ("Count at 0.001, step 16", PILOT_METRICS, by_step["validation", 16]),
        ("Count at 1.5e-7, step 16", fresh, json.loads((midway / fresh).read_text())),
    ]
    rows = []
    for label, path, row in samples:
        counts = row["success_counts"]
        correct = sum(int(k) * puzzles for k, puzzles in counts.items())
        n = row["prompts"]
        zero, one = counts.get("0", 0), counts.get("1", 0)
        rows.append([label, f"{correct}/{row['responses']:,} ({percent(row['pass@1'])})",
                     f"{n - zero}/{n} ({percent(row['pass@8'])})",
                     f"{zero}/{n} ({percent(row['zero_success_frequency'])})",
                     f"{one}/{n} ({percent(row['one_success_frequency'])})", _source(path)])

    weights = []
    for step, label in [(1, "Batch 1, before the first update"), (2, "Batch 2, after the first update")]:
        row = by_step["train", step]
        cells = [label]
        for distribution in ("all_weights", "successful_weights"):
            low, high = (row[distribution][q] for q in ("min", "max"))
            cells.append(f"{low['weight']:.8g} to {high['weight']:.8g}")
        cells.append(f"{row['all_weights']['min']['log_weight']:.6g} to {row['all_weights']['max']['log_weight']:.6g}")
        weights.append(cells)

    return "\n\n".join([
        "## Sampled validation accuracy",
        "Pass@1 counts correct responses. The other columns count puzzles by the number of successes "
        "in eight draws. The first two rows are separate samples from the same initial model.",
        table(["Model and sample", "Pass@1", "Pass@8", "No success", "Exactly one", "Source"], rows),
        "### Importance weights around the first update",
        f"The original Count pilot used a learning rate of 0.001. Each range spans responses in "
        f"one training batch. The pilot logged weights before applying that batch's update. {_source(PILOT_METRICS)}",
        table(["When measured", "All weights", "Weights on successes", "All log weights"], weights),
    ])


def _initial_worker(midway, gpu, timing_path):
    # The first launcher published its process timing CSV and recorded its worker total in the log.
    log_path = "../../docs/research-log.md"
    log = (midway / log_path).read_text()
    initial_entry = next((line for line in log.splitlines() if "| Initial Qwen3-0.6B validation," in line), "")
    match = re.search(r"The worker including verification and copying took ([\d,.]+) s \(\$([\d.]+)\)", initial_entry)
    if match is None:
        raise ValueError(f"Missing initial-evaluation worker time or cost in {log_path}")
    seconds, usd = (float(value.replace(",", "")) for value in match.groups())
    return (RUN_LABELS["initial-eval"], gpu, seconds, usd, f"{_source(log_path)} · {_source(timing_path)}")


def _cost_section(midway):
    timing_path = "initial-eval/resources.csv"
    with (midway / timing_path).open(newline="") as stream:
        process = next(row for row in csv.DictReader(stream) if row["phase"] == "total")
    attempts = [(json.loads(path.read_text()), path.relative_to(midway)) for path in midway.rglob("attempt.json")]
    runs = []
    if not (midway / "initial-eval/attempt.json").exists():
        runs.append(_initial_worker(midway, process["gpu"], timing_path))
    for record, path in sorted(attempts, key=lambda entry: entry[0]["started_utc"]):
        if record["stage"] == "preflight":
            continue
        folder = path.parent.as_posix()
        label = RUN_LABELS.get(folder, folder)
        runs.append((label, record["gpu_name"], record["wall_seconds"], record["estimated_cost_usd"], _source(path)))
    amounts = [
        [label, gpu, f"{seconds:,.2f}", f"${cost:.4f}", source]
        for label, gpu, seconds, cost, source in runs
    ]
    amounts.append(["**Total**", "", f"**{sum(run[2] for run in runs):,.2f}**",
                    f"**${sum(run[3] for run in runs):.4f}**", "Sum of the rows above"])
    budget_path = "pilot-lr-0.001/budget.json"
    projected = json.loads((midway / budget_path).read_text())["projection"]["controlled_usd"]
    return "\n\n".join([
        "## GPU time and estimated cost",
        "Each row covers a complete GPU worker, including setup, verification and copying results. "
        "The estimate includes the requested GPU, CPU and memory. The six trajectories shared one worker.",
        table(["GPU run", "Reported GPU", "Worker seconds", "Estimated USD", "Source"], amounts),
        f"For the initial evaluation, the evaluator process alone took {float(process['seconds']):,.2f} s "
        f"({1000 * float(process['seconds']) / int(process['responses']):,.2f} s per 1,000 responses) "
        f"and cost ${float(process['machine_usd']):.4f} at ${float(process['hourly_usd']):.4f} per hour. "
        f"{_source(timing_path)}",
        f"The **${projected:,.0f} projection** (${projected:,.2f}) was for the original plan's "
        f"pilot, screening, tuning and main training runs. Diagnostics and Li replication were unpriced "
        f"in that projection. {_source(budget_path)}",
    ])


def _trajectories_section(midway):
    path = "saved-trajectories/trajectories.jsonl"
    summary_path = "saved-trajectories/summary.json"
    run_path = "saved-trajectories/pilot/run.json"
    summary = json.loads((midway / summary_path).read_text())
    config = json.loads((midway / run_path).read_text())["config"]
    measured = {}
    for line in (midway / path).read_text().splitlines():
        row = json.loads(line)
        if "validation_is_post_update" in row:
            measured[row["method"], row["lr"], row["step"]] = row
    rows = []
    for method in summary["methods"]:
        label = {"count": "Count", "snis": "SNIS", "uncorrected": "Uncorrected"}[method]
        for rate in summary["rates"]:
            samples = [measured[method, rate, step] for step in (4, 8, 16)]
            pools = [sample["validation_is_post_update"]["pooled"] for sample in samples]
            final = pools[-1]
            rows.append([label, f"{rate:g}", *[percent(pool["pass_at_1"]) for pool in pools],
                         f"{final['ess']:,.1f}", f"{final['success_ess']:.1f}",
                         percent(samples[-1]["validation_open_probability_post_update"])])
    # Both validation pools are fixed across all methods, rates and measurement steps.
    pool = next(iter(measured.values()))["validation_is_post_update"]["pooled"]
    return "\n\n".join([
        "## Six runs on the same saved batches",
        f"The six trajectories reused {config['updates']} batches from seed {config['seed']}. "
        f"Each batch contained {config['prompts_per_step']} puzzles and {config['responses_per_prompt']} "
        f"responses per puzzle. {_source(run_path)}",
        f"Each pass@1 estimate averages reward times the response's importance weight across the two "
        f"initial-model validation pools. Together they contained {pool['successes']}/{pool['n']:,} correct "
        f"responses ({percent(pool['successes'] / pool['n'])}). The last three columns are measured at "
        f"step 16. The final column is the model's probability of opening with the exact token `<`, "
        f"averaged over the {summary['validation_prompt_count']} validation prompts. It started at "
        f"{percent(summary['initial_validation_open_probability'])}. {_source(path)} · {_source(summary_path)}",
        table(["Method", "SGD rate", "Pass@1 at step 4", "Step 8", "Step 16", "ESS, all", "ESS, successes",
               "P(first token `<`)"], rows),
    ])


def render(midway: Path) -> str:
    sections = [
        "# Countdown midpoint report numbers",
        "Rebuild from the repository root with `python -m maxrl.report_numbers`. [Results index](README.md)",
        _validation_section(midway),
        _cost_section(midway),
        _trajectories_section(midway),
        midpoint_section(midway),
        supporting_section(midway),
    ]
    return "\n\n".join(sections) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    midway = Path("results/midway")
    parser.add_argument("--out", type=Path, default=midway / "report-numbers.md",
                        help="Markdown destination (default: results/midway/report-numbers.md)")
    args = parser.parse_args()
    args.out.write_text(render(midway), encoding="utf-8")


if __name__ == "__main__":
    main()
