"""Run one controlled training trajectory with full-response MaxRL updates."""

import argparse
import copy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from .config import Runtime, load_config, runtime_arguments
from .data import load_panel
from .estimators import coefficients
from .evaluate import generate_groups, record_groups
from .metrics import ResourceTable, accuracy, dump_json, hardware
from .model import checkpoint_id, load_policy, save_policy, sequence_logprobs
from .prompts import prompt_metadata
from .rollout import stream_seed


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def summarize_weights(log_weights):
    """Keep log-ratio quantiles even when a diagnostic raw weight overflows."""
    if not log_weights:
        return None
    values = np.quantile(log_weights, [0, .25, .5, .75, .95, 1])
    return {
        name: {"log_weight": float(value), "weight": math.exp(value) if value < 709.78 else None}
        for name, value in zip(["min", "p25", "p50", "p75", "p95", "max"], values)
    }


def update_policy(model, optimizer, groups, method, score_batch):
    """Accumulate success scores and apply one plain SGD step to the entire batch."""
    if not groups or any(not group for group in groups):
        raise ValueError("Training requires nonempty response groups")
    model.eval()  # Score the same dropout-free policy used for generation.
    optimizer.zero_grad(set_to_none=True)
    all_log_w, success_log_w = [], []
    for group in groups:
        with torch.no_grad():
            current = torch.cat([sequence_logprobs(model, part) for part in chunks(group, score_batch)])
        behavior = [
            math.fsum(lp for lp, keep in zip(row.sampler_logprobs, row.response_mask) if keep)
            for row in group
        ]
        log_w = current.cpu().numpy() - behavior
        if not np.isfinite(log_w).all():
            raise FloatingPointError("Nonfinite full-response log ratio")
        rewards = [r.reward for r in group]
        k = sum(rewards)
        all_log_w.extend(log_w)
        success_log_w.extend(w for w, reward in zip(log_w, rewards) if reward)
        if k == 0:
            continue
        # NumPy produces detached weights. Re-score successes in small batches to
        # keep at most one differentiable microbatch in GPU memory at a time.
        lambdas = coefficients(method, log_w, rewards)
        weighted = [(row, float(weight) / (k * len(groups)))
                    for row, weight in zip(group, lambdas) if row.reward]
        for part in chunks(weighted, score_batch):
            rows, scale = zip(*part)
            score = sequence_logprobs(model, rows)
            weights = score.new_tensor(scale)
            loss = -torch.dot(weights, score)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite controlled loss")
            loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    norm = torch.nn.utils.get_total_norm(gradients).item() if gradients else 0.0
    if not math.isfinite(norm):
        raise FloatingPointError("Nonfinite SGD gradient norm")
    optimizer.step()
    if gradients:
        # A finite gradient can still yield an overflowing SGD update. The
        # infinity norm scans FP32 parameters without allocating a model copy.
        with torch.no_grad():
            largest = torch.nn.utils.get_total_norm(list(model.parameters()), norm_type=math.inf)
            if not torch.isfinite(largest):
                raise FloatingPointError("Nonfinite parameters after SGD")
    optimizer.zero_grad(set_to_none=True)
    summary = {"gradient_norm": norm, "all_weights": summarize_weights(all_log_w),
               "successful_weights": summarize_weights(success_log_w)}
    return summary, all_log_w


def log_event(out, event):
    with (out / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps(event, allow_nan=False) + "\n")
    print(json.dumps({key: event[key] for key in ("split", "step", "pass@1")}), flush=True)


def train(config, runtime, manifest, out):
    if config.lr is None:
        raise ValueError("Supply the trial or frozen learning rate with --lr")
    out = Path(out)
    resources = ResourceTable(runtime)
    train_puzzles = load_panel(manifest, "train")
    validation = load_panel(manifest, "validation")
    if len(train_puzzles) < config.prompts_per_step or not validation:
        raise ValueError("Manifest needs enough training prompts and a validation panel")
    out.mkdir(parents=True, exist_ok=False)
    (out / "responses").mkdir()
    run_id = uuid4().hex
    base_id = checkpoint_id(config.model, config.revision)
    dump_json(out / "run.json", {
        "run_id": run_id, "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config), "runtime": asdict(runtime), "hardware": hardware(runtime),
        "prompt": prompt_metadata(),
        "initial_checkpoint_id": base_id,
        "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
    })
    step = 0
    phase = "load_policy"
    last_validation = last_checkpoint = None
    try:
        torch.manual_seed(config.seed)
        policy, tokenizer = load_policy(config.model, device=runtime.device, dtype=runtime.dtype,
                                       revision=config.revision)
        sampler = policy if config.method == "fresh" else copy.deepcopy(policy).requires_grad_(False)
        optimizer = torch.optim.SGD(policy.parameters(), lr=config.lr, momentum=0, weight_decay=0)
        prompt_rng = np.random.default_rng(stream_seed(config.seed, "training-prompt-order"))
        current_id = sampler_id = base_id
        for step in range(config.updates + 1):
            if step == config.updates or step in config.checkpoint_steps:
                phase = "save_checkpoint"
                save_policy(policy, tokenizer, out / "checkpoints" / f"step-{step:06d}", current_id)
                last_checkpoint = step
            if step % config.eval_every == 0 or step == config.updates:
                phase = "validation"
                started = resources.clock()
                val = generate_groups(
                    policy, tokenizer, validation, config.validation_samples, identity=current_id,
                    pool=f"{run_id}/validation/{step}", seed=stream_seed(config.seed, "validation", step),
                    runtime=runtime, token_cap=config.max_new_tokens,
                )
                stats = accuracy(val)
                seconds = resources.add("evaluation", started, stats)
                record_groups(out / "responses" / f"validation-{step:06d}.jsonl", val)
                log_event(out, dict(split="validation", step=step, **stats, seconds=seconds))
                last_validation = {"step": step, **stats}
            if step == config.updates:
                break
            phase = "sampler_refresh"
            if config.method == "fresh":
                sampler_id = current_id
            elif step and step % config.refresh_every == 0:
                sampler.load_state_dict(policy.state_dict())
                sampler_id = current_id
            phase = "training_generation"
            ids = prompt_rng.choice(len(train_puzzles), config.prompts_per_step, replace=False)
            started = resources.clock()
            groups = generate_groups(
                sampler, tokenizer, [train_puzzles[i] for i in ids], config.responses_per_prompt,
                identity=sampler_id, pool=f"{run_id}/train/{step + 1}",
                seed=stream_seed(config.seed, "training-rollout", step), runtime=runtime,
                token_cap=config.max_new_tokens,
            )
            stats = accuracy(groups)
            generation_time = resources.add("training_generation", started, stats)
            phase = "save_responses"
            record_groups(out / "responses" / f"train-{step + 1:06d}.jsonl", groups)
            started = resources.clock()
            phase = "training_update"
            update, log_w = update_policy(policy, optimizer, groups, config.method, runtime.score_batch)
            update_time = resources.add("training_update", started)
            phase = "save_update_metrics"
            with (out / "responses" / f"weights-{step + 1:06d}.csv").open("x", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["sample_id", "puzzle_id", "reward", "log_weight"])
                rows = (row for group in groups for row in group)
                for row, value in zip(rows, log_w, strict=True):
                    writer.writerow([row.sample_id, row.puzzle_id, row.reward, value])
            log_event(out, dict(split="train", step=step + 1, sampler_id=sampler_id,
                                target_id=current_id, **stats, **update,
                                generation_seconds=generation_time, update_seconds=update_time))
            current_id = f"{run_id}/step-{step + 1:06d}"
    except (Exception, KeyboardInterrupt) as error:
        dump_json(out / "failure.json", {
            "type": type(error).__name__, "message": str(error), "phase": phase,
            "category": "numerical" if isinstance(error, FloatingPointError) else "other",
            "target_step": step, "last_validation": last_validation,
            "last_saved_checkpoint_step": last_checkpoint,
        })
        raise
    finally:
        resources.save(out / "resources.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    runtime_arguments(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, method=args.method, lr=args.lr, seed=args.seed)
    train(cfg, Runtime.from_args(args), args.manifest, args.out)


if __name__ == "__main__":
    main()
