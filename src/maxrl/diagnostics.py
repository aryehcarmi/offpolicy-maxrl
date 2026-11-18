"""Collect and compare saved-response gradients for the fixed diagnostic panels."""

import argparse
import csv
import json
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .data import digest
from .estimators import STALE_METHODS
from .gradient_metrics import grouped_updates, summarize
from .prompts import prompt_metadata, validate_prompt


@contextmanager
def staged_directory(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.TemporaryDirectory(dir=path.parent) as temp:
        staging = Path(temp) / "staging"
        staging.mkdir()
        yield staging
        staging.rename(path)


def _dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def _plan(panel, step):
    if step < 1:
        raise ValueError("choose a reached positive checkpoint")
    if panel == "midpoint":
        return 8, {"initial": (0, 64), "current": (step, 64), "reference": (step, 128)}
    if step != 64:
        raise ValueError("discovery and confirmation require update 64")
    return 32, {"initial": (0, 256), "lag16": (48, 256),
                "current": (64, 256), "reference": (64, 512)}


def _checkpoint(run, config, step):
    if step:
        path = run / "checkpoints" / f"step-{step:06d}"
        if not path.is_dir():
            raise FileNotFoundError(f"checkpoint has not been saved: {path}")
        return str(path), None
    return config["model"], config["revision"]


def stage_resources(args):
    from .config import Runtime
    from .metrics import ResourceTable, hardware

    runtime = Runtime(device=args.device, dtype=args.dtype, hourly_usd=args.hourly_usd,
                      generation_batch=getattr(args, "microbatch", 1),
                      score_batch=getattr(args, "microbatch", 1))
    return ResourceTable(runtime), hardware(runtime)


def _selected_step(run, panel, requested):
    if requested is not None:
        return requested
    if panel != "midpoint":
        return 64
    steps = [int(p.name[5:]) for p in (run / "checkpoints").glob("step-*")
             if p.is_dir() and p.name[5:].isdigit()]
    if not steps:
        raise ValueError("the pilot has no saved checkpoint yet")
    return max(steps)


def collect(args):
    from .data import load_panel
    from .model import checkpoint_id, load_policy
    from .rollout import sample_puzzle, stream_seed
    from .schema import write_responses

    run = Path(args.run)
    record = json.loads((run / "run.json").read_text())
    validate_prompt(record.get("prompt"))
    config = record["config"]
    if record["manifest_sha256"] != digest(Path(args.manifest)):
        raise ValueError("diagnostics and training must use the same data split")
    if args.dtype != record["runtime"]["dtype"]:
        raise ValueError("use the training forward dtype for diagnostics")
    expected = {"midpoint": ("count", 17), "discovery": ("fresh", 101),
                "confirmation": ("fresh", 102)}[args.panel]
    if (config["method"], config["seed"]) != expected:
        raise ValueError(f"{args.panel} requires {expected[0]} training seed {expected[1]}")
    step = _selected_step(run, args.panel, args.step)
    count, pools = _plan(args.panel, step)
    puzzles = load_panel(args.manifest, args.panel)
    if len(puzzles) != count:
        raise ValueError(f"panel must contain exactly {count} puzzles")
    resources, hardware = stage_resources(args)
    manifest = {
        "version": 1, "panel": args.panel, "seed": args.seed, "step": step,
        "prompt": prompt_metadata(),
        "hardware": hardware, "training_seed": config["seed"], "run": str(run.resolve()), "dtype": args.dtype,
        "device": args.device, "generation_microbatch": args.microbatch,
        "training_config": config, "split_sha256": record["manifest_sha256"],
        "prompt_ids": [p.puzzle_id for p in puzzles], "pools": {},
    }
    with staged_directory(args.out) as dest:
        if args.panel == "confirmation":
            if not args.prediction or not Path(args.prediction).read_text().strip():
                raise ValueError("confirmation needs the written discovery prediction")
            (dest / "prediction.md").write_bytes(Path(args.prediction).read_bytes())
        checkpoints = {}
        for checkpoint_step in sorted({s for s, _ in pools.values()}):
            path, revision = _checkpoint(run, config, checkpoint_step)
            identity = checkpoint_id(path, revision)
            expected_id = (f"{record['run_id']}/step-{checkpoint_step:06d}" if checkpoint_step
                           else record["initial_checkpoint_id"])
            if expected_id != identity:
                raise ValueError("checkpoint identity differs from the specified trajectory")
            checkpoints[checkpoint_step] = (path, revision, identity)
        for checkpoint_step, (path, revision, identity) in checkpoints.items():
            model, tokenizer = load_policy(path, device=args.device, dtype=args.dtype, revision=revision)
            for name, (s, draws) in pools.items():
                if s != checkpoint_step:
                    continue
                pool_id = f"{record['run_id']}/diagnostics/{args.panel}/{step}/{args.seed}/{name}"
                # Stream seeds are independent of the random run ID so a
                # repeated trajectory can replay exactly the same experiment.
                counts = {"responses": 0, "generated_tokens": 0}

                def responses():
                    for puzzle in puzzles:
                        seed = stream_seed(args.seed, args.panel, config["seed"], step, name, puzzle.puzzle_id)
                        rows = sample_puzzle(model, tokenizer, puzzle, draws,
                                             checkpoint_id=identity, pool=pool_id, seed=seed,
                                             microbatch=args.microbatch, max_new_tokens=377)
                        counts["responses"] += len(rows)
                        counts["generated_tokens"] += sum(sum(r.response_mask) for r in rows)
                        yield from rows
                output = dest / f"{name}.jsonl"
                started = resources.clock()
                write_responses(output, responses())
                resources.add(name, started, counts)
                print(f"Saved {counts['responses']} responses from {name}", flush=True)
                manifest["pools"][name] = {"step": s, "draws": draws, "checkpoint_id": identity,
                                           "pool_id": pool_id, "sha256": digest(output)}
            del model, tokenizer
        resources.save(dest / "resources.csv")
        _dump(dest / "manifest.json", manifest)


def load_rows(directory, manifest, name):
    from .schema import read_responses

    pool = manifest["pools"][name]
    path = directory / f"{name}.jsonl"
    if digest(path) != pool["sha256"]:
        raise ValueError(f"saved responses changed for pool {name}")
    rows = list(read_responses(path))
    draws, prompts = pool["draws"], manifest["prompt_ids"]
    if len(rows) != draws * len(prompts):
        raise ValueError(f"incomplete pool {name}")
    for i, r in enumerate(rows):
        if (r.puzzle_id != prompts[i // draws] or r.pool != pool["pool_id"] or
                r.checkpoint_id != pool["checkpoint_id"]):
            raise ValueError(f"mismatched prompt, pool or generating policy at row {i}")
    return rows


def target_model(args, manifest):
    from .model import checkpoint_id, load_policy

    validate_prompt(manifest.get("prompt"))
    path, rev = _checkpoint(Path(args.run or manifest["run"]), manifest["training_config"],
                            manifest["step"])
    if checkpoint_id(path, rev) != manifest["pools"]["current"]["checkpoint_id"]:
        raise ValueError("target checkpoint identity differs from collection")
    if args.dtype != manifest["dtype"]:
        raise ValueError("use the collection dtype to replay the same target policy")
    model, _ = load_policy(path, device=args.device, dtype=args.dtype, revision=rev)
    return model


def score_rows(model, rows):
    import torch
    from .model import sequence_logprobs

    # Freezing layers below the final norm lets PyTorch discard their graph.
    # Each successful row has exactly one score vector d log(pi(y | x))/d gamma.
    model.eval()
    model.requires_grad_(False)
    gain = model.model.norm.weight
    gain.requires_grad_(True)
    scores = np.zeros((len(rows), gain.numel()), dtype=np.float32)
    log_weights = np.empty(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        with torch.set_grad_enabled(bool(row.reward)):
            value = sequence_logprobs(model, [row])[0]
            if row.reward:
                grad, = torch.autograd.grad(value, gain)
                scores[i] = grad.detach().float().cpu().numpy()
        sampler_lp = np.dot(np.asarray(row.sampler_logprobs, dtype=np.float64), row.response_mask)
        log_weights[i] = value.detach().item() - sampler_lp
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(log_weights)):
        raise FloatingPointError("nonfinite replay score or log weight")
    return scores, log_weights


def score(args):
    directory = Path(args.pools)
    manifest = json.loads((directory / "manifest.json").read_text())
    resources, hardware = stage_resources(args)
    model = target_model(args, manifest)
    with staged_directory(directory / "scores") as dest:
        hashes = {}
        for name, pool in manifest["pools"].items():
            rows = load_rows(directory, manifest, name)
            started = resources.clock()
            scores, lw = score_rows(model, rows)
            resources.add(name, started)
            shape = (len(manifest["prompt_ids"]), pool["draws"])
            score_path = dest / f"{name}.npz"
            np.savez(score_path, scores=scores.reshape(*shape, -1),
                     log_weights=lw.reshape(shape),
                     rewards=np.array([r.reward for r in rows], dtype=np.uint8).reshape(shape))
            hashes[name] = digest(score_path)
        resources.save(dest / "resources.csv")
        _dump(dest / "provenance.json", {"manifest_sha256": digest(directory / "manifest.json"),
                                          "score_sha256": hashes,
                                          "target": manifest["pools"]["current"]["checkpoint_id"],
                                          "device": args.device, "dtype": args.dtype, "hardware": hardware})


def _scored_manifest(directory):
    manifest_path = directory / "manifest.json"
    record = json.loads((directory / "scores" / "provenance.json").read_text())
    if record["manifest_sha256"] != digest(manifest_path):
        raise ValueError("score provenance differs from the saved pools manifest")
    manifest = json.loads(manifest_path.read_text())
    validate_prompt(manifest.get("prompt"))
    for name in manifest["pools"]:
        if digest(directory / f"scores/{name}.npz") != record["score_sha256"][name]:
            raise ValueError(f"saved scores changed for pool {name}")
    return manifest


def analyze(args):
    directory = Path(args.pools)
    manifest = _scored_manifest(directory)
    started = time.perf_counter()
    prompts = manifest["prompt_ids"]
    n = args.group_size
    results = {}
    with staged_directory(directory / f"analysis-N{n}") as dest:
        with np.load(directory / "scores/reference.npz") as ref:
            reference, _, ref_failures = grouped_updates(ref["rewards"], ref["log_weights"], ref["scores"],
                                           n, prompts, methods=("fresh",))
            if ref_failures:
                raise FloatingPointError(f"unrepresentable fresh reference: {ref_failures}")
        exposure = []
        for name in manifest["pools"]:
            if name == "reference":
                continue
            with np.load(directory / f"scores/{name}.npz") as pool:
                log_w, r = pool["log_weights"], pool["rewards"]
                methods = (*STALE_METHODS, "fresh") if name == "current" else STALE_METHODS
                batches, rows, failures = grouped_updates(r, log_w, pool["scores"], n, prompts, methods)
                results[name] = summarize(batches, reference["fresh"], args.bootstrap, args.seed, failures)
                results[name]["success_count_histogram"] = np.bincount(
                    [row["K"] for row in rows], minlength=n + 1).tolist()
                results[name]["log_weight_quantiles_0_50_95_100"] = np.quantile(
                    log_w, [0, 0.5, 0.95, 1]).tolist()
                for row in rows:
                    row["source"] = name
                exposure.extend(rows)
        columns = list(dict.fromkeys(k for row in exposure for k in row))
        with (dest / "exposure.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(exposure)
        _dump(dest / "summary.json", {"group_size": n, "results": results,
                                      "cpu_seconds": time.perf_counter() - started,
                                      "manifest_sha256": digest(directory / "manifest.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="stage", required=True)
    for stage in ("collect", "score", "analyze", "full"):
        p = subs.add_parser(stage)
        if stage == "collect":
            p.add_argument("--run", required=True)
            p.add_argument("--manifest", required=True, help="prepared split manifest")
            p.add_argument("--out", required=True)
            p.add_argument("--panel", choices=["midpoint", "discovery", "confirmation"], required=True)
            p.add_argument("--step", type=int)
            p.add_argument("--prediction", help="written prediction to freeze with confirmation data")
            p.add_argument("--microbatch", type=int, default=4)
            p.add_argument("--seed", type=int, default=3011)
        else:
            p.add_argument("--pools", required=True)
        if stage in ("full", "score"):
            p.add_argument("--run", help="relocated training run with the same checkpoint")
        if stage in ("collect", "score", "full"):
            p.add_argument("--device", default="cuda")
            p.add_argument("--hourly-usd", type=float, help="total hourly machine price")
            p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
        if stage in ("analyze", "full"):
            p.add_argument("--group-size", choices=[8, 32], type=int, default=8)
            p.add_argument("--bootstrap", type=int, default=500)
            p.add_argument("--seed", type=int, default=7001)
        if stage == "full":
            p.add_argument("--source", choices=["initial", "current", "lag16"], default="initial")
            p.add_argument("--scratch", help="directory for temporary full-gradient arrays")
            p.add_argument("--microbatch", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "full":
        from .full_gradients import full
        full(args)
    else:
        {"collect": collect, "score": score, "analyze": analyze}[args.stage](args)


if __name__ == "__main__":
    main()
