"""Prompt-equal accuracy and single-machine resource accounting."""

from collections import Counter
import csv
from dataclasses import asdict
from importlib.metadata import version
import json
import math
import platform
import time

import torch


def dump_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def accuracy(groups):
    if not groups or any(not group for group in groups):
        raise ValueError("Accuracy needs nonempty groups")
    counts = [sum(int(row.reward) for row in group) for group in groups]
    histogram = Counter(counts)
    result = {
        "prompts": len(groups), "responses": sum(map(len, groups)),
        "generated_tokens": sum(sum(row.response_mask) for group in groups for row in group),
        "zero_success_frequency": histogram[0] / len(groups),
        "one_success_frequency": histogram[1] / len(groups),
        "success_counts": dict(sorted(histogram.items())),
    }
    for k in [1, 8]:
        if min(map(len, groups)) >= k:
            # Probability of at least one success in k draws without replacement.
            result[f"pass@{k}"] = sum(
                1 - math.comb(len(group) - c, k) / math.comb(len(group), k)
                for group, c in zip(groups, counts)
            ) / len(groups)
    return result


def hardware(runtime):
    device = torch.device(runtime.device)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda": torch.version.cuda, "torch": torch.__version__,
        "transformers": version("transformers"), "numpy": version("numpy"),
        "runtime": asdict(runtime),
    }


class ResourceTable:
    """Synchronized phase times, response volume and elapsed machine cost."""
    def __init__(self, runtime):
        self.runtime = runtime
        self.started = self.clock()
        self.totals = {}

    def clock(self):
        if torch.device(self.runtime.device).type == "cuda":
            torch.cuda.synchronize(self.runtime.device)
        return time.perf_counter()

    def add(self, phase, started, counts=None):
        seconds = self.clock() - started
        row = self.totals.setdefault(phase, [0.0, 0, 0])
        row[0] += seconds
        if counts:
            row[1] += counts["responses"]
            row[2] += counts["generated_tokens"]
        return seconds

    def save(self, path):
        elapsed = self.clock() - self.started
        measured = sum(item[0] for item in self.totals.values())
        phases = {**self.totals, "setup_io_other": [max(0, elapsed - measured), 0, 0]}
        phases["total"] = [elapsed, sum(v[1] for v in phases.values()), sum(v[2] for v in phases.values())]
        rate = self.runtime.hourly_usd
        device = torch.device(self.runtime.device)
        gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else ""
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["phase", "seconds", "responses", "generated_tokens", "machine_usd",
                             "device", "gpu", "hourly_usd"])
            for phase, (seconds, count, tokens) in phases.items():
                writer.writerow([phase, seconds, count, tokens, None if rate is None else seconds * rate / 3600,
                                 str(device), gpu, rate])
