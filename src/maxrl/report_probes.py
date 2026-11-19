"""Render the saved-batch verification, rate selection and first-update probes."""

import csv
import json
from pathlib import Path


def supporting_section(midway: Path) -> str:
    """Return the supporting measurements with links relative to the midway folder."""
    verification_path = "saved-verify/verification.json"
    original_rates_path = "replay/original/rates.csv"
    rates_path = "replay/low/rates.csv"
    summary_path = "saved-trajectories/summary.json"
    updates_path = "saved-trajectories/trajectories.jsonl"
    cluster_path = "replay/batch-2-opening-cluster.json"
    tokens_path = "saved-trajectories/probes/first-token-summary.json"
    types_path = "saved-trajectories/probes/parameter-types.json"

    steps = json.loads((midway / verification_path).read_text())["steps"]
    summary = json.loads((midway / summary_path).read_text())
    selected = summary["selected"]
    rate = selected["lr"]
    with (midway / original_rates_path).open(newline="") as stream:
        smallest = min(csv.DictReader(stream), key=lambda r: float(r["rate"]))
    with (midway / rates_path).open(newline="") as stream:
        row = next(r for r in csv.DictReader(stream) if float(r["rate"]) == rate)
    max_norm_error = max(abs(s["saved_gradient_norm"] - s["replay_gradient_norm"]) for s in steps)
    max_weight_error = max(s["abs_log_weight_error_max"] for s in steps)
    lines = [
        "## Replay and first-update checks",
        "",
        f"The first-batch gradient norm was {steps[0]['saved_gradient_norm']:.2f}. "
        f"Replaying the collapsed pilot gave a maximum absolute gradient-norm difference of "
        f"{max_norm_error:g} across {len(steps)} updates and a maximum absolute log-weight "
        f"difference of {max_weight_error:g} across {sum(s['responses'] for s in steps):,} responses. "
        f"[Per-update verification]({verification_path})",
        "",
        f"At the original grid's smallest rate ({float(smallest['rate']):g}), the first update realized "
        f"{float(smallest['realized_over_predicted']):.2%} of its predicted objective increase. "
        f"Only {smallest['heldout_within_0p1']}/{smallest['heldout_count']} held-out log-weights "
        f"fell within ±0.1. [Original rate grid]({original_rates_path})",
        "",
        f"At the selected rate of {rate:g}, the first update realized "
        f"{float(row['realized_over_predicted']):.2%} of its predicted objective increase. "
        f"{row['heldout_within_0p1']}/{row['heldout_count']} batch-2 log-weights "
        f"({float(row['heldout_fraction_within_0p1']):.2%}) fell within ±0.1. The float32 step "
        f"retained {float(row['rounding_projection_fraction']):.4%} of the intended displacement "
        "along its direction. The study waived the original 99% rounding threshold. "
        f"Across {row['pooled_success_count']} successes in the other saved batches, the mean "
        f"weight was {float(row['pooled_mean_weight_success']):.5f} "
        f"(standard error {float(row['pooled_mean_weight_success_se']):.5f}, clustered by puzzle within batch). "
        f"[Rate table]({rates_path}) · [Selected checkpoint]({summary_path}) · "
        "[Amendment](../../docs/research-log.md)",
    ]

    updates = [json.loads(line) for line in (midway / updates_path).read_text().splitlines()]
    first = next(u for u in updates
                 if u["method"] == selected["method"] and u["lr"] == rate and u["step"] == 1)
    push = first["opening_logit_push_pre_update"]
    method_names = {"count": "Count", "snis": "SNIS", "uncorrected": "Uncorrected"}
    lines.extend([
        "",
        f"In the first Count training batch, coefficient-weighted successes "
        f"opened with `<` {push['weighted_success_opening_share']:.2%} of the time. The model assigned "
        f"it {push['weighted_model_opening_probability']:.2%} on the same weighted prompts, for a "
        f"difference of {100 * push['normalized_logit_push']:+.2f} percentage points. "
        f"[Per-update opening measurements]({updates_path})",
        "",
        "Number of batches with a negative opening-token difference at each rate. "
        f"[Per-update measurements]({updates_path})",
        "",
        "| SGD rate | " + " | ".join(method_names[m] for m in summary["methods"]) + " |",
        "|---:" + "|---:" * len(summary["methods"]) + "|",
    ])
    for lr in summary["rates"]:
        counts = []
        for method in summary["methods"]:
            trajectory = [u for u in updates if u["method"] == method and u["lr"] == lr]
            negative = sum(u["opening_logit_push_pre_update"]["normalized_logit_push"] < 0
                           for u in trajectory)
            counts.append(f"{negative}/{len(trajectory)}")
        lines.append(f"| {lr:g} | " + " | ".join(counts) + " |")

    cluster = json.loads((midway / cluster_path).read_text())
    prefix = next(p for p in cluster["prefix_counts"]["2"] if p["text"] == "<answer")
    tokens = json.loads((midway / tokens_path).read_text())
    lines.extend([
        "",
        f"At {cluster['rate']:g}, {prefix['cluster']}/{cluster['group_statistics']['cluster']['responses']} "
        f"responses in the lower cluster of batch {cluster['batch']} log-weights opened with "
        f"`<` followed by `answer`. That prefix appeared in "
        f"{prefix['other']}/{cluster['group_statistics']['other']['responses']} other responses. "
        f"The table splits the mean change in log probability after one update into contributions "
        f"from the first and remaining tokens, in nats. [Opening counts]({cluster_path}) · "
        f"[Token-level decomposition]({tokens_path})",
        "",
        "| First token | Responses | First token change | Remaining change | Whole-response change |",
        "|---|---:|---:|---:|---:|",
    ])
    for label, key in (("`<`", "opens_angle"), ("Other", "other")):
        group = tokens["groups"][key]
        lines.append(f"| {label} | {group['n']} | {group['mean_first_change']:+.4f} | "
                     f"{group['mean_remaining_change']:+.4f} | {group['mean_full_change']:+.4f} |")

    parameters = json.loads((midway / types_path).read_text())
    lines.extend([
        "",
        f"The last column counts parameters whose float32 value changed on the first Count step "
        f"at {parameters['rate']:g}. The embedding and output head share weights, counted once. "
        f"Normalization includes query and key gains. [Parameter counts and gradient norms]({types_path})",
        "",
        "| Parameter type | Share of squared gradient norm | Parameters of this type that changed |",
        "|---|---:|---:|",
    ])
    labels = {"embedding": "Tied embedding", "norm": "Normalization", "attention": "Attention", "mlp": "MLP"}
    for key, label in labels.items():
        group = parameters["types"][key]
        lines.append(f"| {label} | {group['grad_squared_norm_share']:.2%} | "
                     f"{group['fraction_changed_within_type']:.2%} |")
    return "\n".join(lines)
