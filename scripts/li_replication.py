"""Run in the pinned upstream Linux environment. The upstream code owns training and evaluation."""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "configs/replication/li_frozen.json"


def run_name(method, seed):
    return f"li_{method}_seed{seed}"


def train(args, spec, upstream, base):
    env = os.environ.copy()
    env.update(
        PY=sys.executable, REPO=str(upstream), MODEL=base, METHOD=args.method, SOURCE=spec["source"],
        SEED=str(args.seed), STEPS=str(spec["updates"]), LR=str(spec["learning_rate"]),
        EXP=run_name(args.method, args.seed), OUT_DIR=str(args.out), SAVE_FREQ="5",
        TRAIN_FILE=str(upstream / "data/countdown/train.parquet"),
        VAL_FILE=str(upstream / "data/countdown/heldout.parquet"),
        DRY="1" if args.dry_run else "0",
    )
    subprocess.run(["bash", "scripts/train.sh"], cwd=upstream, env=env, check=True)


def evaluate(args, spec, upstream, base):
    from tim.measure import config, engines
    from tim.measure.deviation import heldout_accuracy

    is_initial = args.method == "initial"
    name = "initial" if is_initial else run_name(args.method, args.seed)
    dest = args.out / name / "evaluation.json"
    if dest.exists():
        raise FileExistsError(dest)
    checkpoint = base if is_initial else str(
        dest.parent / f"ckpt/global_step_{spec['updates']}/actor/huggingface"
    )
    if not Path(checkpoint).is_dir():
        raise FileNotFoundError(checkpoint)
    settings = config.load(str(upstream / spec["upstream_measurement"]), resolve=False)
    evaluation = settings["deviation"]["heldout"]
    engines.setup_kernel_env(["fa3_liger"])
    start = time.perf_counter()
    sampler = engines.shared(base, settings["topk"]).sampler("bf16", checkpoint)
    result = heldout_accuracy(sampler, base, **evaluation, cfg=settings)
    reference = spec["reported_accuracy"][args.method]
    result.update(
        published_fraction=reference, difference_percentage_points=100 * (result["mean_accuracy"] - reference),
        elapsed_seconds=time.perf_counter() - start, evaluation_settings=evaluation, checkpoint=checkpoint,
        method=args.method, training_seed=None if is_initial else args.seed, upstream_commit=spec["upstream_commit"],
        model_revision=spec["revision"],
    )
    if result["n_completions"] != evaluation["n_prompts"] * evaluation["n_samples"]:
        raise RuntimeError("incomplete upstream evaluation")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


def summarize(root, spec):
    """Three matched seeds. Intervals describe between-training-seed variation."""
    results = {}
    scores = {}
    for method in ("is", "is_sc"):
        scores[method] = []
        for seed in spec["training_seeds"]:
            metrics = json.loads((root / run_name(method, seed) / "evaluation.json").read_text())
            if (metrics["method"], metrics["training_seed"], metrics["upstream_commit"]) != (
                method, seed, spec["upstream_commit"]
            ):
                raise ValueError("replication evaluation provenance differs from the protocol")
            scores[method].append(metrics["mean_accuracy"])
    contrasts = [b - a for a, b in zip(scores["is"], scores["is_sc"], strict=True)]
    for method, values in {**scores, "is_sc_minus_is": contrasts}.items():
        mean = statistics.mean(values)
        # t_(0.975, 2), conditional on the fixed puzzles and approximate normal seed effects.
        half_width = 4.3026527299 * statistics.stdev(values) / len(values) ** 0.5
        results[method] = {"by_seed": dict(zip(spec["training_seeds"], values, strict=True)),
                           "mean": mean, "seed_t_interval_95": [mean - half_width, mean + half_width]}
        if method in spec["reported_accuracy"]:
            results[method]["published_mean"] = spec["reported_accuracy"][method]
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "evaluate", "summary"])
    parser.add_argument("--upstream", type=Path, default=Path("external/li"))
    parser.add_argument("--out", type=Path, default=Path("runs/replication"))
    parser.add_argument("--method", choices=["initial", "is", "is_sc"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="ask the upstream training launcher to print its command")
    args = parser.parse_args()
    spec = json.loads(CONFIG.read_text())
    if args.action == "summary":
        print(json.dumps(summarize(args.out, spec), indent=2, allow_nan=False))
        return
    if args.method is None:
        parser.error("select --method for training or evaluation")
    if args.seed not in spec["training_seeds"]:
        parser.error("training seeds must be 0, 1 or 2")
    if args.method == "initial" and (args.action != "evaluate" or args.seed != 0):
        parser.error("initial evaluation uses the untrained model and seed zero")
    if args.dry_run and args.action != "train":
        parser.error("--dry-run applies to training")
    upstream = args.upstream.resolve()
    args.out = args.out.resolve()
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != spec["upstream_commit"]:
        parser.error(f"upstream checkout must be at {spec['upstream_commit']}")
    # The vLLM worker subprocesses also import the upstream refit hook.
    sys.path.insert(0, str(upstream))
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (str(upstream), os.environ.get("PYTHONPATH"))))
    from huggingface_hub import snapshot_download
    base = snapshot_download(spec["model"], revision=spec["revision"])
    if args.action == "train":
        train(args, spec, upstream, base)
    else:
        evaluate(args, spec, upstream, base)


if __name__ == "__main__":
    main()
