"""Figures built from exactly the rows written to the toy moments table."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {"fresh": "Fresh", "uncorrected": "Uncorrected", "count": "Count",
          "snis": "SNIS", "count_cap": "Count, raw cap", "snis_cap": "SNIS, raw cap",
          "loo": "LOO", "loo_cap": "LOO, final cap", "hybrid": "Singleton hybrid",
          "exact": "Exact"}


def _panel(rows, case, n, geometry):
    return {r["method"]: r for r in rows
            if (r["case"], r["n"], r["geometry"]) == (case, n, geometry)}


def plot_moments(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5), sharey=True, layout="constrained")
    panels = [("rate_large__composition_equal", 2, "Success rate 0.05 → 0.20\nN = 2, composition fixed"),
              ("rate_equal__composition_large", 8, "Success composition 0.20 → 0.80\nN = 8, success rate fixed")]
    y = np.arange(len(LABELS))
    for ax, (case, n, title) in zip(axes, panels):
        panel = _panel(rows, case, n, "one_prompt")
        bias = [panel[m]["squared_bias"] for m in LABELS]
        variance = [panel[m]["variance"] for m in LABELS]
        ax.barh(y, bias, color="#bb4f35", label="Squared bias")
        ax.barh(y, variance, left=bias, color="#37789c", label="Variance")
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlabel("Mean squared error")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, LABELS.values())
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Single-prompt error against the expected fresh update", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_batch_direction(rows, path):
    methods = ["count", "snis", "count_cap", "snis_cap", "loo_cap", "hybrid"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.20, top=0.84, wspace=0.06)
    for ax, geometry in zip(axes, ("separate", "shared")):
        for shift, marker, color in [("equal", "o", "#37789c"), ("large", "x", "#bb4f35")]:
            panel = _panel(rows, f"rate_large__composition_{shift}", 8, geometry)
            angles = [np.degrees(np.arccos(panel[m]["cosine"])) for m in methods]
            label = (
                "Prompt 1: rate; prompt 2: composition"
                if shift == "equal"
                else "Both prompts: rate and composition"
            )
            ax.scatter(angles, range(len(methods)), color=color, marker=marker, s=42, label=label)
        ax.set_title(f"{geometry.capitalize()} logits", loc="left", fontsize=12)
        ax.set_xlabel("Angle to fresh target (degrees)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_xlim(left=-1)
    axes[0].set_yticks(range(len(methods)), [LABELS[m] for m in methods])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", frameon=False, ncol=2, fontsize=10)
    fig.suptitle("Direction of the two-prompt mean update, N = 8", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)
