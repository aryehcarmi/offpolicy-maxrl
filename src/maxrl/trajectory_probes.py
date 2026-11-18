"""Token, validation and parameter measurements for fixed-batch training trajectories."""

import argparse
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config, Runtime
from .data import digest, load_panel
from .estimators import coefficients
from .metrics import ResourceTable, dump_json
from .model import autocast_policy, checkpoint_id, load_policy
from .prompts import prompt_ids, validate_prompt
from .replay import read_batch, score_log_weights
from .schema import read_responses
from .train import accumulate_gradient, sampler_logprob


def opening_id(tokenizer):
    """The observed answer-first marker is the exact single generated token '<'."""
    ids = tokenizer.encode("<", add_special_tokens=False)
    if len(ids) != 1 or tokenizer.decode(ids, skip_special_tokens=False) != "<":
        raise ValueError("Tokenizer does not have the expected single '<' token")
    return ids[0]


@torch.no_grad()
def opening_logprobs(model, prompts, token_id):
    """Full-softmax probability at the next position, with one observation per prompt."""
    device = next(model.parameters()).device
    result = []
    with autocast_policy(model):
        for prompt in prompts:
            inputs = torch.tensor([prompt], dtype=torch.long, device=device)
            logits = model(input_ids=inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            result.append(F.log_softmax(logits, dim=-1)[token_id].clone())
    return torch.stack(result).double().cpu().numpy()


def logit_push(groups, method, log_weights, opening_probs, token_id):
    """Loss-ascent derivative for a shared additive '<' logit at the first position."""
    if len(opening_probs) != len(groups) or sum(map(len, groups)) != len(log_weights):
        raise ValueError("Training groups, opening probabilities and log weights differ in length")
    observed, expected, mass = [], [], []
    offset = 0
    for group, probability in zip(groups, opening_probs, strict=True):
        logs = log_weights[offset:offset + len(group)]
        offset += len(group)
        rewards = [row.reward for row in group]
        k = sum(rewards)
        if k:
            weights = coefficients(method, logs, rewards) / (k * len(groups))
            total = math.fsum(weights)
            mass.append(total)
            expected.append(total * probability)
            observed.append(math.fsum(w for w, r in zip(weights, group) if r.response_ids[0] == token_id))
    denom, observed, expected = math.fsum(mass), math.fsum(observed), math.fsum(expected)
    return {
        "coefficient_mass": denom, "direct_logit_push": observed - expected,
        "weighted_success_opening_share": observed / denom if denom else None,
        "weighted_model_opening_probability": expected / denom if denom else None,
        "normalized_logit_push": (observed - expected) / denom if denom else None,
    }


def load_validation(pilot, initial_eval, manifest, tokenizer):
    """Validate the two reserved-panel samples and the exact prompt tokens for all draws."""
    pilot, initial_eval, manifest = Path(pilot), Path(initial_eval), Path(manifest)
    run = json.loads((pilot / "run.json").read_text())
    initial = json.loads((initial_eval / "run.json").read_text())
    validate_prompt(run["prompt"])
    validate_prompt(initial["prompt"])
    base_id = run["initial_checkpoint_id"]
    if (initial["checkpoint_id"] != base_id or initial["panel"] != "validation"
            or initial["manifest_sha256"] != run["manifest_sha256"] or digest(manifest) != run["manifest_sha256"]):
        raise ValueError("The initial evaluation and pilot must use the same validation panel and initial policy")
    puzzles = load_panel(manifest, "validation")
    prompts = {p.puzzle_id: prompt_ids(tokenizer, p) for p in puzzles}
    if len(prompts) != 128:
        raise ValueError("Expected exactly 128 validation puzzles")
    sources = [
        ("initial_eval", initial_eval / "responses.jsonl", f"{base_id}/validation"),
        ("pilot_step0", pilot / "responses/validation-000000.jsonl", f"{run['run_id']}/validation/0"),
    ]
    pools, seen = {}, set()
    for name, path, expected_pool in sources:
        rows = list(read_responses(path))
        draws = Counter()
        for r in rows:
            if (r.pool != expected_pool or r.checkpoint_id != base_id or r.prompt_ids != prompts.get(r.puzzle_id)
                    or r.sample_id in seen):
                raise ValueError(f"Inconsistent validation sample: {r.sample_id}")
            seen.add(r.sample_id)
            draws[r.puzzle_id] += 1
        if draws != {p: 8 for p in prompts}:
            raise ValueError(f"Expected 8 draws per validation puzzle in {name}")
        pools[name] = rows
    return pools, list(prompts.values())


def weight_ess(logs):
    if not len(logs):
        return 0.
    w = np.exp(logs - np.max(logs))
    return float(np.sum(w) ** 2 / np.dot(w, w))


def is_estimate(logs, rewards):
    """Ordinary importance sampling. All prompts supply the same fixed number of draws."""
    successes = logs[rewards]
    log_p = float(np.logaddexp.reduce(successes) - math.log(len(logs))) if len(successes) else None
    try:
        p = math.exp(log_p) if log_p is not None else 0.
    except OverflowError:
        p = None
    return {"n": len(logs), "successes": len(successes), "pass_at_1": p,
            "log_pass_at_1": log_p, "ess": weight_ess(logs), "success_ess": weight_ess(successes)}


def validation_estimates(model, pools, score_batch=1):
    estimates, all_logs, all_rewards = {}, [], []
    for name, rows in pools.items():
        logs = score_log_weights(model, rows, score_batch)
        rewards = np.array([r.reward for r in rows], dtype=bool)
        estimates[name] = is_estimate(logs, rewards)
        all_logs.append(logs)
        all_rewards.append(rewards)
    estimates["pooled"] = is_estimate(np.concatenate(all_logs), np.concatenate(all_rewards))
    return estimates


@torch.no_grad()
def first_and_total(model, rows):
    """Match the trainer's FP32 token cross-entropy and FP64 sum on the same forward pass."""
    device = next(model.parameters()).device
    result = []
    for r in rows:
        ids = torch.tensor([r.prompt_ids + r.response_ids], device=device)
        attention = torch.ones_like(ids[:, :-1])
        with autocast_policy(model):
            logits = model(input_ids=ids[:, :-1], attention_mask=attention, use_cache=False).logits
        start = len(r.prompt_ids) - 1
        stop = start + sum(r.response_mask)
        token_logp = -F.cross_entropy(logits[0, start:stop].float(), ids[0, start+1:stop+1], reduction="none")
        result.append(torch.stack([token_logp[0].double(), token_logp.double().sum()]))
    return torch.stack(result).cpu().numpy()


def attribution_records(rows, before, after, token_id):
    for r, base, updated in zip(rows, before, after, strict=True):
        sampler = np.array([r.sampler_logprobs[0], sampler_logprob(r)])
        first, full = updated - base
        initial_first, initial_full = base - sampler
        updated_first, updated_full = updated - sampler
        yield {
            "sample_id": r.sample_id, "puzzle_id": r.puzzle_id, "reward": r.reward,
            "first_token_id": r.response_ids[0], "opens_angle": r.response_ids[0] == token_id,
            "initial_first_log_weight": initial_first, "initial_log_weight": initial_full,
            "updated_first_log_weight": updated_first, "updated_log_weight": updated_full,
            "first_log_weight_change": first, "remaining_log_weight_change": full - first,
            "full_log_weight_change": full, "first_fraction_of_change": first / full if full else None,
        }


def attribution_summary(rows):
    result = {}
    for key, group in (("all", rows), ("opens_angle", [r for r in rows if r["opens_angle"]]),
                       ("other", [r for r in rows if not r["opens_angle"]])):
        if not group:
            result[key] = {"n": 0}
            continue
        mean_first = math.fsum(r["first_log_weight_change"] for r in group) / len(group)
        mean_total = math.fsum(r["full_log_weight_change"] for r in group) / len(group)
        ratios = [r["first_fraction_of_change"] for r in group if r["first_fraction_of_change"] is not None]
        result[key] = {"n": len(group), "mean_first_change": mean_first, "mean_full_change": mean_total,
                       "mean_remaining_change": mean_total - mean_first,
                       "ratio_of_mean_changes": mean_first / mean_total if mean_total else None,
                       "median_first_fraction": float(np.median(ratios)) if ratios else None}
    return result


def parameter_type(name):
    if "norm" in name:
        return "norm"
    if name.endswith(("embed_tokens.weight", "lm_head.weight")):
        return "embedding"
    if ".self_attn." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    raise ValueError(f"Unrecognized parameter type: {name}")


@torch.no_grad()
def parameter_summary(model, initial):
    totals = {}
    for (name, p), old in zip(model.named_parameters(), initial, strict=True):
        kind = parameter_type(name)
        entry = totals.setdefault(kind, {"parameter_count": 0, "changed_count": 0, "grad_squared_norm": 0.})
        entry["parameter_count"] += p.numel()
        if p.grad is None:
            continue
        entry["changed_count"] += int(torch.count_nonzero(p != old))
        grad = p.grad.flatten()
        # The tied embedding can contain hundreds of millions of parameters.
        for block in grad.split(1 << 20):
            v = block.double()
            entry["grad_squared_norm"] += torch.dot(v, v).item()
    grad_squared = math.fsum(d["grad_squared_norm"] for d in totals.values())
    changed = sum(d["changed_count"] for d in totals.values())
    for d in totals.values():
        d["grad_squared_norm_share"] = d["grad_squared_norm"] / grad_squared if grad_squared else None
        d["fraction_changed_within_type"] = d["changed_count"] / d["parameter_count"]
        d["share_of_all_changed_parameters"] = d["changed_count"] / changed if changed else None
    return totals


def run_checks(pilot, out, device, hourly_usd):
    source = json.loads((pilot / "run.json").read_text())
    cfg = Config(**source["config"])
    identity = checkpoint_id(cfg.model, cfg.revision)
    validate_prompt(source["prompt"])
    if (identity != source["initial_checkpoint_id"] or cfg.method != "count"
            or source["runtime"]["dtype"] != "float32" or source["runtime"]["score_batch"] != 1):
        raise ValueError("These probes require the original float32 Count pilot")
    runtime = replace(Runtime(**source["runtime"]), device=device, hourly_usd=hourly_usd)
    first, second = [read_batch(pilot, source["run_id"], step, cfg, identity) for step in (1, 2)]
    rows = [r for g in second for r in g]
    out.mkdir(parents=True, exist_ok=False)
    resources = ResourceTable(runtime)
    try:
        torch.manual_seed(cfg.seed)
        model, tokenizer = load_policy(cfg.model, device=device, dtype=runtime.dtype, revision=cfg.revision)
        token_id = opening_id(tokenizer)
        started = resources.clock()
        original = first_and_total(model, rows)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.5e-7)
        optimizer.zero_grad(set_to_none=True)
        stats, _ = accumulate_gradient(model, first, "count", runtime.score_batch)
        initial = [p.detach().clone() for p in model.parameters()]
        optimizer.step()
        dump_json(out / "parameter-types.json", {
            "method": "count", "step": 1, "rate": 1.5e-7, "gradient_norm": stats["gradient_norm"],
            "definition": "Unique parameters, with tied token embedding/output head counted once. All RMSNorm gains, including attention Q/K norms, are in norm.",
            "types": parameter_summary(model, initial),
        })
        with torch.no_grad():
            for p, saved in zip(model.parameters(), initial, strict=True):
                p.copy_(saved)
        del initial
        optimizer.param_groups[0]["lr"] = 1e-6
        optimizer.step()
        updated = first_and_total(model, rows)
        records = list(attribution_records(rows, original, updated, token_id))
        with (out / "first-token.jsonl").open("x") as f:
            for r in records:
                f.write(json.dumps(r, allow_nan=False) + "\n")
        dump_json(out / "first-token-summary.json", {
            "rate": 1e-6, "method": "count", "training_step": 1, "heldout_batch": 2,
            "opening_token_id": token_id, "source_run_id": source["run_id"],
            "definition": "Changes subtract initial-policy from updated-policy teacher-forced log probabilities for each response. Saved sampler denominators cancel in the change. Individual fractions are signed and can exceed one.",
            "groups": attribution_summary(records),
        })
        resources.add("first_update_probes", started)
    finally:
        resources.save(out / "resources.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--hourly-usd", type=float)
    args = p.parse_args()
    run_checks(args.pilot, args.out, args.device, args.hourly_usd)


if __name__ == "__main__":
    main()
