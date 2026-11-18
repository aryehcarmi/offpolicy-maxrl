# Li et al. frozen-sampler replication

The targets are 8.2% initial accuracy, 20.9% after truncated importance sampling and 32.0% after truncated importance sampling with score centering. Each percentage is per-completion accuracy. The initial-model check measures agreement of the evaluation pipeline. Only the trained comparison tests replication of learning. [Li, Zhou & Lan, Tables 2 and 5 and Appendix B](https://arxiv.org/html/2610.40945v1)

## Build the authors' environment

Run these commands from the repository root on a Linux x86-64 NVIDIA GPU machine with uv and a compatible driver. The upstream lock uses Python 3.12 and CUDA 12.9. Keep the full pinned environment: the evaluation sampler uses both vLLM and verl's weight-loading code. FlashAttention-3 is not needed for the initial evaluation. [Environment and GPU notes](https://github.com/Skylanding/Training-Inference-Mismatch/blob/98e8f9fa15200f294c9395626969f745771ee460/envs/README.md) · [Sampler implementation](https://github.com/Skylanding/Training-Inference-Mismatch/blob/98e8f9fa15200f294c9395626969f745771ee460/tim/measure/engines.py)

```bash
mkdir -p external
git clone https://github.com/Skylanding/Training-Inference-Mismatch.git external/li
git -C external/li checkout --detach 98e8f9fa15200f294c9395626969f745771ee460
(
  cd external/li
  bash envs/verl-main-cu129/build_env.sh
  .venv-tim/bin/python scripts/build_countdown_data.py
)
LI_PY="$PWD/external/li/.venv-tim/bin/python"
```

The upstream seed-0 generator produces 20,000 four-number puzzles. Its first 64 form the held-out set. Upstream code supplies the prompt template, reward, trainer and BF16 vLLM evaluation engine. The held-out evaluator generates eight responses for each puzzle at temperature one with seed equal to the prompt index and a total length limit of 512 tokens. The training launcher uses a 377-token response limit. [Evaluator](https://github.com/Skylanding/Training-Inference-Mismatch/blob/98e8f9fa15200f294c9395626969f745771ee460/tim/measure/deviation.py) · [Training launcher](https://github.com/Skylanding/Training-Inference-Mismatch/blob/98e8f9fa15200f294c9395626969f745771ee460/scripts/train.sh)

## Initial-model evaluation

```bash
"$LI_PY" scripts/li_replication.py evaluate --method initial
```

The wrapper writes the measured accuracy, its difference from 8.2% in percentage points, generation statistics and elapsed seconds to `runs/replication/initial/evaluation.json`. This check can run before building the training kernels.

## Training kernels and full comparison

The published training configuration uses `KERNELS=fa3_liger`. Its FlashAttention-3 build targets H100 (`sm_90a`) and requires CUDA build tools. This requirement applies to the default replication trainer, not to this repository's controlled HF/SDPA experiments. [Build script](https://github.com/Skylanding/Training-Inference-Mismatch/blob/98e8f9fa15200f294c9395626969f745771ee460/envs/fa3/build_fa3.sh)

```bash
(
  cd external/li
  export FA3_PATH="$PWD/.fa3"
  PY="$PWD/.venv-tim/bin/python" bash envs/fa3/build_fa3.sh
  .venv-tim/bin/python envs/fa3/check_fa3.py
)
export FA3_PATH="$PWD/external/li/.fa3"
```

Upstream also supports `KERNELS=sdpa` for these methods on compatible GPUs, without the FA3 build. It changes attention kernels, padding and the FSDP configuration; record it as a deviation from the published training setup. Do not assume the supplied H100 build works unchanged on another GPU architecture.

To train and evaluate all six replications, use the same shell and environment:

```bash
for method in is is_sc; do
  for seed in 0 1 2; do
    "$LI_PY" scripts/li_replication.py train --method "$method" --seed "$seed" || exit 1
    "$LI_PY" scripts/li_replication.py evaluate --method "$method" --seed "$seed" || exit 1
  done
done
"$LI_PY" scripts/li_replication.py summary
```

The authors' launcher owns the 64 × 8 batches, fixed initial sampler, 40 updates, SGD at 0.01, ±1 rewards, group-mean centered advantages, token-mean loss and token ratio cap of two. The score-centering arm uses the authors' top-128 head. Training and held-out evaluation use the same immutable initial Qwen revision as the controlled pipeline. Checkpoints are saved every five steps. The accuracy measurement uses step 40. The summary reports all seeds, each method mean and a paired contrast with a small-sample t interval. Outputs use upstream formats, and each run has its own output directory. The upstream evaluator reports aggregate accuracy and length statistics. We keep its sampling call unchanged for replication. Only the controlled runs save per-token probabilities for replay. The declared config is [li_frozen.json](../configs/replication/li_frozen.json).

Set `CUDA_VISIBLE_DEVICES` to select the device. Upstream supports `N_GPUS`, `MICRO_BS`, `TIM_MEASURE_SAMPLER_GMU` and alternate kernels. Record hardware and any changes to published settings beside the results. Use `train --method is --dry-run` to inspect the generated Hydra arguments after setup. Comparing all three trained seeds with the published means requires reporting seed variation and any implementation deviations. For the two CPU-tested upstream components, run `(cd external/li && .venv-tim/bin/python -m pytest -q tests/test_countdown.py tests/test_train_launcher.py)`. The initial CPU verification is recorded in the [research log](research-log.md). Published percentages remain reference values until the GPU runs finish.
