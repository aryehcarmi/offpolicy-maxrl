"""Independent evaluation of controlled checkpoints on reserved panels."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from .config import Runtime, load_config, runtime_arguments
from .data import load_panel
from .metrics import ResourceTable, accuracy, dump_json, hardware
from .model import checkpoint_id, load_policy
from .prompts import prompt_metadata, validate_prompt
from .rollout import sample_puzzle, stream_seed
from .schema import write_responses


def generate_groups(model, tokenizer, puzzles, count, *, identity, pool, seed, runtime, token_cap):
    return [
        sample_puzzle(
            model, tokenizer, puzzle, count, checkpoint_id=identity, pool=pool,
            seed=stream_seed(seed, puzzle.puzzle_id), microbatch=runtime.generation_batch,
            max_new_tokens=token_cap,
        )
        for puzzle in puzzles
    ]


def record_groups(path, groups):
    write_responses(path, (row for group in groups for row in group))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", help="Saved project checkpoint; defaults to the initial model")
    parser.add_argument("--panel", choices=["validation", "test"], default="validation")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, help="Evaluation sampling seed")
    parser.add_argument("--method", help="Training method recorded for this checkpoint")
    parser.add_argument("--lr", type=float, help="Training rate recorded for this checkpoint")
    runtime_arguments(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, seed=args.seed, method=args.method, lr=args.lr)
    runtime = Runtime.from_args(args)
    resources = ResourceTable(runtime)
    puzzles = load_panel(args.manifest, args.panel)
    if not puzzles:
        raise ValueError("Selected panel is empty")
    checkpoint = args.checkpoint or cfg.model
    revision = cfg.revision if checkpoint == cfg.model else None
    identity = checkpoint_id(checkpoint, revision)
    training_config = None
    saved = Path(checkpoint).resolve()
    marker = saved / "maxrl_checkpoint.json"
    if marker.is_file():
        validate_prompt(json.loads(marker.read_text()).get("prompt"))
    origin = saved.parent.parent / "run.json"
    if saved.is_dir() and saved.parent.name == "checkpoints" and origin.is_file():
        metadata = json.loads(origin.read_text())
        validate_prompt(metadata.get("prompt"))
        if not identity.startswith(metadata["run_id"] + "/"):
            raise ValueError("Checkpoint identity conflicts with its training directory")
        training_config = metadata["config"]
        for key in ("method", "lr"):
            supplied = getattr(args, key)
            if supplied is not None and supplied != training_config[key]:
                raise ValueError(f"Requested {key} conflicts with the saved trajectory")
        cfg = replace(cfg, method=training_config["method"], lr=training_config["lr"])
    args.out.mkdir(parents=True, exist_ok=False)
    dump_json(args.out / "run.json", {
        "config": asdict(cfg), "runtime": asdict(runtime), "hardware": hardware(runtime),
        "prompt": prompt_metadata(),
        "checkpoint_id": identity, "panel": args.panel, "training_config": training_config,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
    })
    try:
        model, tokenizer = load_policy(checkpoint, device=runtime.device, dtype=runtime.dtype, revision=revision)
        seed = stream_seed(cfg.seed, args.panel)
        sampled_at = resources.clock()
        groups = generate_groups(
            model, tokenizer, puzzles, cfg.test_samples if args.panel == "test" else cfg.validation_samples,
            identity=identity, pool=f"{identity}/{args.panel}", seed=seed, runtime=runtime,
            token_cap=cfg.max_new_tokens,
        )
        stats = accuracy(groups)
        stats["seconds"] = resources.add("evaluation", sampled_at, stats)
        record_groups(args.out / "responses.jsonl", groups)
        dump_json(args.out / "metrics.json", stats)
        print(stats)
    finally:
        resources.save(args.out / "resources.csv")


if __name__ == "__main__":
    main()
