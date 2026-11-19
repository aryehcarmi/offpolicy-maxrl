"""Render the gradient comparisons from the published midpoint summaries."""

import json
from pathlib import Path


_LABELS = {
    "uncorrected": "Uncorrected",
    "count": "Count",
    "snis": "SNIS",
    "count_cap": "Capped Count",
    "snis_cap": "Capped SNIS",
    "loo": "Leave-one-out (LOO)",
    "loo_cap": "Capped LOO",
    "hybrid": "Hybrid",
    "fresh": "Fresh",
}
_METRICS = (
    "cosine_of_mean",
    "raw_batch_mse",
    "norm_matched_mse",
    "mean_batch_norm_over_snis_norm_on_matched",
)


def _estimate(stat: dict) -> str:
    value = stat["estimate"]
    if value is None:
        return "Overflow" if stat["point_overflow"] else "Undefined"
    fmt = ",.0f" if abs(value) >= 1000 else ".3f"
    if stat["ci95"] is None:
        return format(value, fmt)
    low, high = stat["ci95"]
    if low != high and format(low, fmt) == format(high, fmt):
        fmt = ".6f"
    return f"{value:{fmt}} [{low:{fmt}}, {high:{fmt}}]"


def _method_table(panel: dict) -> list[str]:
    rows = [
        "| Correction | Cosine | Raw MSE | Norm-matched MSE | Length / SNIS |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, metrics in panel["methods"].items():
        cells = [_LABELS[method], *(_estimate(metrics[field]) for field in _METRICS)]
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def midpoint_section(midway: Path) -> str:
    """Return a Markdown section with links relative to the midway index."""
    run = Path("midpoint-lr-1.5e-7")
    manifest_path = run / "manifest.json"
    rms_path = run / "analysis-N8/summary.json"
    full_path = run / "full-initial-N8/summary.json"
    manifest = json.loads((midway / manifest_path).read_text())
    rms = json.loads((midway / rms_path).read_text())
    full = json.loads((midway / full_path).read_text())
    puzzles = len(manifest["prompt_ids"])
    pools = manifest["pools"]
    initial = rms["results"]["initial"]
    lines = [
        "## Midpoint gradient comparisons",
        "",
        f"At step {manifest['step']}, the {puzzles} fixed Countdown puzzles supplied "
        f"{puzzles * pools['initial']['draws']:,} initial-model responses, "
        f"{puzzles * pools['current']['draws']:,} current-model responses and "
        f"{puzzles * pools['reference']['draws']:,} independent current-model responses "
        f"for the fresh reference. [Panel manifest]({manifest_path})",
        "",
        "Cosine measures alignment of the mean update with the fresh reference. Raw mean squared error (MSE) "
        "is the average squared distance of a batch update from that reference. Norm-matched "
        "MSE first rescales each update to the SNIS length on its own batch. Length / SNIS "
        "averages those per-batch length ratios. Both length comparisons use batches on which "
        "both lengths are positive.",
        "",
        f"Brackets give 95% intervals from {initial['bootstrap_repeats']} bootstrap draws. "
        "Each draw resamples whole candidate and reference batches independently on the fixed "
        "puzzles and uses the same draws for all methods. "
        f"[Bootstrap settings]({rms_path}) · [Diagnostic definitions](../../docs/diagnostics.md)",
    ]
    for source, panel in rms["results"].items():
        step = pools[source]["step"]
        lines.extend([
            "",
            f"### Final RMSNorm gain on {source}-model responses",
            "",
            f"The sampler was at step {step}. The analysis used {panel['stale_batches']} "
            f"candidate batches and {panel['reference_batches']} reference batches, with "
            f"{rms['group_size']} responses per puzzle per batch. [RMSNorm summary]({rms_path})",
            "",
            *_method_table(panel),
        ])
        if source == "initial":
            pair = panel["paired_differences"]["snis_minus_count"]
            lines.extend([
                "",
                f"The paired SNIS minus Count difference in raw MSE was {_estimate(pair['raw_batch_mse'])}. "
                f"Matching lengths on {pair['matched_batches']} shared nonzero batches gave a difference "
                f"of {_estimate(pair['norm_matched_mse'])}. The fresh reference "
                f"split-half cosine was {_estimate(panel['reference_split_half_cosine'])}. "
                f"The direction comparison for SNIS was {panel['methods']['snis']['mean_direction_status']}. "
                f"[Paired differences and split-half check]({rms_path})",
            ])
    panel = full["metrics"]
    lines.extend([
        "",
        f"### Full model on {full['source']}-model responses",
        "",
        f"The full-model check used {panel['stale_batches']} candidate batch and "
        f"{panel['reference_batches']} reference batches. A single candidate batch gives "
        f"point estimates only. [Full-model summary]({full_path})",
        "",
        *_method_table(panel),
    ])
    return "\n".join(lines)
