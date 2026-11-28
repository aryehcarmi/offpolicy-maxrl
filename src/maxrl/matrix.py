"""Print controlled experiment commands and freeze learning rates from validation."""

import argparse
import json
import math
import shlex
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from .config import Runtime, load_config, numerical_runtime, runtime_arguments
from .data import digest
from .prompts import prompt_metadata, validate_prompt

METHODS = ("count", "snis", "fresh", "uncorrected")
SEEDS = {method: list(range(101, 106 if method in ("count", "snis") else 104)) for method in METHODS}
SCREEN = (3e-7, 1e-6, 3e-6)
REPLAY_RATE = 1.5e-7
GAIN_MARGIN = 0.02
STEPS = {"screen": (16, 32), "tune": (64, 80, 96)}
PRIMARY = "final test pass@1, SNIS minus Count within each matched training seed"


def trials(stage, eta0=None):
    methods = ["fresh"] if stage == "screen" else METHODS
    rates = SCREEN if stage == "screen" else (eta0 / 3, eta0, 3 * eta0)
    if stage == "tune" and eta0 >= 1e-6:
        rates = (REPLAY_RATE, *rates)
    for method in methods:
        for index, rate in enumerate(rates):
            yield method, rate, Path(stage) / f"{method}-{index}"


def read_trial(directory, expected_config, manifest_hash, stage, expected_runtime):
    steps = STEPS[stage]
    required_steps = (0, *steps) if stage == "screen" else steps
    result = {"method": expected_config.method, "lr": expected_config.lr,
              "directory": str(directory), "validation": {}}
    if not (directory / "run.json").is_file():
        return dict(result, status="missing")
    try:
        record = json.loads((directory / "run.json").read_text())
        validate_prompt(record.get("prompt"))
        if record["config"] != asdict(expected_config) or record["manifest_sha256"] != manifest_hash:
            return dict(result, status="invalid", reason="config or manifest does not match trial")
        if numerical_runtime(record.get("runtime")) != numerical_runtime(expected_runtime):
            return dict(result, status="invalid", reason="numerical runtime does not match trial")
        metrics = directory / "metrics.jsonl"
        if metrics.is_file():
            for line in metrics.read_text().splitlines():
                event = json.loads(line)
                if event["split"] == "validation" and event["step"] in required_steps:
                    step, score = str(event["step"]), event["pass@1"]
                    if (step in result["validation"] or type(score) not in (int, float)
                            or not 0 <= score <= 1 or event["prompts"] != 128
                            or event["responses"] != 128 * expected_config.validation_samples):
                        return dict(result, status="invalid", reason=f"invalid validation at step {step}")
                    result["validation"][step] = score
        if (directory / "failure.json").is_file():
            failure = json.loads((directory / "failure.json").read_text())
            status = "numerical_failure" if failure.get("category") == "numerical" else "other_failure"
            if status == "numerical_failure" and stage == "screen" and "0" not in result["validation"]:
                status = "incomplete"
            return dict(result, status=status, failure=failure)
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        return dict(result, status="invalid", reason=str(error))
    validation = result["validation"]
    if set(validation) != set(map(str, required_steps)):
        return dict(result, status="incomplete")
    status = "collapsed" if validation[str(steps[-1])] == 0 else "valid"
    return dict(result, status=status, score=mean(validation[str(step)] for step in steps))


def read_trials(stage, args, manifest_hash, eta0=None):
    records = []
    runtime = numerical_runtime(asdict(Runtime.from_args(args)))
    for method, rate, directory in trials(stage, eta0):
        cfg = load_config(args.configs / f"{stage}.json", method=method, lr=rate)
        records.append(read_trial(args.root / directory, cfg, manifest_hash, stage, runtime))
    if any(row["status"] not in ("valid", "collapsed", "numerical_failure") for row in records):
        raise ValueError(f"{stage} has missing, incomplete or invalid trials, or an operational failure.\n"
                         + json.dumps(records, indent=2))
    return records


def best_trial(records, method):
    valid = (row for row in records if row["method"] == method and row["status"] == "valid")
    return max(valid, key=lambda row: (row["score"], -row["lr"]), default=None)


def select_screen(args, manifest_hash):
    records = read_trials("screen", args, manifest_hash)
    best = best_trial(records, "fresh")
    if best is None:
        eta0, gain = REPLAY_RATE, None
        print(f"Fresh screen has no selectable trial. Using replay fallback eta0={eta0:g}.", file=sys.stderr)
    else:
        eta0 = best["lr"]
        gain = best["score"] - best["validation"]["0"]
        if gain < GAIN_MARGIN:
            print(f"Fresh screen gain {gain:.2%} is below the {GAIN_MARGIN:.0%} margin. "
                  f"Proceeding with eta0={eta0:g}.", file=sys.stderr)
    outcome = {"gain": gain, "margin": GAIN_MARGIN, "met_margin": gain is not None and gain >= GAIN_MARGIN,
               "fallback_used": best is None}
    return records, eta0, outcome


def select_tuning(args, manifest_hash, eta0):
    records = read_trials("tune", args, manifest_hash, eta0)
    selected = {}
    for method in METHODS:
        best = best_trial(records, method)
        if best is None:
            action = ("Inspect the failure and decide on the shared smaller-rate extension before any cap."
                      if method in ("count", "snis") else "Inspect the failures before proceeding.")
            raise ValueError(f"Every {method} trial collapsed or failed numerically. {action}\n"
                             + json.dumps(records, indent=2))
        selected[method] = best["lr"]
    return records, selected


def command(args, stage, method, rate, seed, directory, test=False):
    config = args.configs / f"{stage}.json"
    program = "evaluate" if test else "train"
    parts = ["python", "-m", f"maxrl.{program}", "--config", config, "--manifest", args.manifest,
             "--method", method, "--lr", repr(rate), "--seed", seed]
    if test:
        horizon = load_config(config).updates
        parts += ["--panel", "test", "--checkpoint", directory / "checkpoints" / f"step-{horizon:06d}"]
        directory = directory / "test"
    parts += ["--out", directory, "--device", args.device, "--dtype", args.dtype,
              "--generation-batch", args.generation_batch, "--score-batch", args.score_batch]
    if args.hourly_usd is not None:
        parts += ["--hourly-usd", args.hourly_usd]
    return shlex.join(map(str, parts))


def load_freeze(path, manifest_hash, main_config):
    frozen = json.loads(path.read_text())
    if frozen.get("version") != 3:
        raise ValueError("Freeze lacks current prompt and numerical runtime metadata; use trials from the current protocol")
    validate_prompt(frozen.get("prompt"))
    numerical_runtime(frozen.get("runtime"))
    if (frozen["manifest_sha256"] != manifest_hash
            or frozen["main_config"] != main_config or frozen["main_seeds"] != SEEDS
            or frozen["primary_contrast"] != PRIMARY or set(frozen["rates"]) != set(METHODS)):
        raise ValueError("Current protocol or manifest differs from the rate freeze")
    if any(not (math.isfinite(rate) and rate > 0) for rate in frozen["rates"].values()):
        raise ValueError("Invalid learning rate in freeze")
    return frozen


def run(args):
    stage = args.stage
    if stage in ("screen", "tune"):
        cfg = load_config(args.configs / f"{stage}.json")
        eta0 = None
        if stage == "tune":
            _, eta0, _ = select_screen(args, digest(args.manifest))
        for method, rate, directory in trials(stage, eta0):
            print(command(args, stage, method, rate, cfg.seed, args.root / directory))
        return

    manifest_hash = digest(args.manifest)
    main_config = asdict(load_config(args.configs / "main.json"))
    path = args.root / "frozen.json"
    if stage == "freeze":
        screen, eta0, outcome = select_screen(args, manifest_hash)
        tune, rates = select_tuning(args, manifest_hash, eta0)
        record = {"version": 3, "prompt": prompt_metadata(),
                  "primary_contrast": PRIMARY, "manifest_sha256": manifest_hash,
                  "screen": screen, "eta0": eta0, "screen_outcome": outcome, "tune": tune, "rates": rates,
                  "main_seeds": SEEDS, "main_config": main_config,
                  "runtime": numerical_runtime(asdict(Runtime.from_args(args)))}
        if path.exists():
            if json.loads(path.read_text()) != record:
                raise ValueError(f"A different freeze already exists at {path}")
        else:
            with path.open("x") as stream:
                json.dump(record, stream, indent=2, allow_nan=False)
                stream.write("\n")
        print(path)
        return

    frozen = load_freeze(path, manifest_hash, main_config)
    evaluation = stage == "test"
    requested_runtime = numerical_runtime(asdict(Runtime.from_args(args)), evaluation=evaluation)
    expected_runtime = numerical_runtime(frozen["runtime"], evaluation=evaluation)
    if requested_runtime != expected_runtime:
        raise ValueError(f"Requested numerical runtime differs from freeze: expected {expected_runtime}")
    for method, seeds in SEEDS.items():
        rate = frozen["rates"][method]
        for seed in seeds:
            directory = args.root / "main" / f"{method}-{seed}"
            print(command(args, "main", method, rate, seed, directory, test=stage == "test"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["screen", "tune", "freeze", "main", "test"])
    parser.add_argument("--root", type=Path, default=Path("runs/matrix"))
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    runtime_arguments(parser)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
