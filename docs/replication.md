# Li et al. frozen-sampler replication

The targets are 8.2% initial accuracy, 20.9% after truncated importance sampling and 32.0% after truncated importance sampling with score centering. Each percentage is per-completion accuracy. The initial-model check measures agreement of the evaluation pipeline. Only the trained comparison tests replication of learning. [Li, Zhou & Lan, Tables 2 and 5 and Appendix B](https://arxiv.org/html/2610.40945v1)

## Initial evaluation on Modal

Run these commands from the project checkout on your laptop. uv installs the pinned Modal client in its own environment. Use `modal setup` for browser login or `modal token set` to enter an existing Modal token ID and secret at the prompts. An authenticated profile can skip login. The pinned public Qwen revision downloads anonymously.

```bash
uv tool install 'modal==1.6.4'
modal setup
modal run scripts/modal_li_initial.py --preflight
```

One local entry point dispatches to two Modal functions with separate hardware requests. The CPU preflight builds the shared image, imports torch and vLLM, and runs the pinned evaluator until vLLM tries to select a device. The laptop verifies the committed log and run record. Preflights write nothing under `results/`. For the 512-completion measurement, the launcher requests one H100 with A100-80GB as the fallback.

```bash
modal run scripts/modal_li_initial.py
```

Pass the total hourly machine price with `--hourly-usd` to estimate cost from the worker's wall time. Use `modal run --detach scripts/modal_li_initial.py` to keep the job running after a disconnect. Each attempt, including a retry after preemption, gets a new UTC-stamped directory on the `li-replication` Volume. Each worker writes `run.log` and `run.json`. A completed measurement also has the wrapper's `initial/evaluation.json`. The attached client saves a copy of the measurement record to `results/midway/li-initial-<run_id>.json`. A disconnect before return leaves that copy on the Volume. [Modal preemption](https://modal.com/docs/guide/preemption)

```bash
modal volume ls li-replication
# Set ATTEMPT to the directory name from the listing.
modal volume ls li-replication "$ATTEMPT"
mkdir -p runs/replication
modal volume get li-replication "$ATTEMPT" runs/replication/
```

Failed workers preserve their records and logs. A hard kill can leave the last `running` snapshot. For a failure during image construction, use the build log linked by the Modal CLI. [Modal run options](https://modal.com/docs/reference/cli/run) · [Volume listing and download](https://modal.com/docs/reference/cli/volume)

## Build the authors' environment on a GPU machine

On a Linux x86-64 NVIDIA GPU machine, install uv, clone the project and enter the repository root. The upstream lock uses Python 3.12 and CUDA 12.9. Use a compatible NVIDIA driver. The evaluation sampler needs both vLLM and verl's weight-loading code. FlashAttention-3 is needed only for the default training kernel setup below. [Environment and GPU notes](https://github.com/Skylanding/Training-Inference-Mismatch/blob/98e8f9fa15200f294c9395626969f745771ee460/envs/README.md) · [Sampler implementation](https://github.com/Skylanding/Training-Inference-Mismatch/blob/98e8f9fa15200f294c9395626969f745771ee460/tim/measure/engines.py)

```bash
set -eu
set -o pipefail
curl -LsSf https://astral.sh/uv/0.11.32/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
git clone https://github.com/aryehcarmi/offpolicy-maxrl.git
cd offpolicy-maxrl
```

A clone contains the published revision. If using unpushed edits, transfer your current launcher, evaluation wrapper and replication config to the GPU checkout before continuing. For example, from the laptop checkout, set `GPU_HOST` to your SSH host and copy the following files.

```bash
rsync -a scripts/modal_li_initial.py scripts/li_replication.py "$GPU_HOST:offpolicy-maxrl/scripts/"
rsync -a configs/replication/li_frozen.json "$GPU_HOST:offpolicy-maxrl/configs/replication/"
```

The sole installation deviation from the published procedure is a guarded rename of the upstream pyproject's `tim-verl-env` to the original lockfile's `verl-main-cu129-env`, which lets the authors' `uv sync --frozen` run. Back in the GPU checkout, run this block in Bash.

```bash
set -eu
mkdir -p external
git clone https://github.com/Skylanding/Training-Inference-Mismatch.git external/li
git -C external/li checkout --detach 98e8f9fa15200f294c9395626969f745771ee460
python3 scripts/modal_li_initial.py --rename-project external/li
(
  cd external/li
  bash envs/verl-main-cu129/build_env.sh
  .venv-tim/bin/python scripts/build_countdown_data.py
)
LI_PY="$PWD/external/li/.venv-tim/bin/python"
```

The upstream seed-0 generator produces 20,000 four-number puzzles. Its first 64 form the held-out set. Upstream code supplies the prompt template, reward, trainer and BF16 vLLM evaluation engine. The held-out evaluator generates eight responses for each puzzle at temperature one with seed equal to the prompt index and a total length limit of 512 tokens. The training launcher uses a 377-token response limit. [Evaluator](https://github.com/Skylanding/Training-Inference-Mismatch/blob/98e8f9fa15200f294c9395626969f745771ee460/tim/measure/deviation.py) · [Training launcher](https://github.com/Skylanding/Training-Inference-Mismatch/blob/98e8f9fa15200f294c9395626969f745771ee460/scripts/train.sh)

## Initial-model evaluation on the GPU machine

```bash
"$LI_PY" scripts/li_replication.py evaluate --method initial
```

The wrapper writes the measured accuracy, its difference from 8.2% in percentage points, generation statistics and elapsed seconds to `runs/replication/initial/evaluation.json`. This check can run before building the training kernels.

On a 24 GB AWS G5/G6 GPU the upstream sampler's 0.15 memory share gives roughly 3.5 GB. The BF16 weights plus the fixed 2,310-block KV cache may need roughly 5 GB. If vLLM reports insufficient memory at startup, increase `TIM_MEASURE_SAMPLER_GMU` and record the value. For example, prefix the evaluation command with `TIM_MEASURE_SAMPLER_GMU=0.30`. The upstream code sets `num_gpu_blocks_override`, so a change to the memory reservation alone is expected to preserve the result. These estimates and the invariance expectation come from reading the upstream engine configuration. The 24 GB setup has not been tested here. [Engine configuration](https://github.com/Skylanding/Training-Inference-Mismatch/blob/98e8f9fa15200f294c9395626969f745771ee460/tim/measure/engines.py)

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
