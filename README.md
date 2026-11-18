# Off-policy MaxRL

Experiments on success normalization for off-policy MaxRL, using exact toy models and Qwen3-0.6B on Countdown.

**Count** divides the importance-weighted score sum by the number of successful responses. **SNIS** divides the same sum by the total importance weight of those successes. For one nonempty group, the two updates have the same direction. Their different scales can change the direction of a batch update across puzzles. The primary training comparison is final test pass@1, SNIS minus Count, over five matched seeds.

The repository contains the toy results and CPU tests, including integration tests with a tiny randomly initialized Qwen3. Experiments with the pretrained model remain to be run.

## Installation

Use Python 3.12.3 and uv from the repository root:

```bash
uv sync --frozen --all-extras
source .venv/bin/activate
python -m pytest -q
```

For the toy experiments and tests without model dependencies:

```bash
uv sync --frozen --extra study --extra test
```

The controlled environment is pinned in [uv.lock](uv.lock). CUDA runs need a driver compatible with the locked PyTorch build. Li et al.'s replication has a [separate environment](docs/replication.md). Tested versions and external commits are listed in [sources](docs/sources.md).

## Toy experiments

```bash
python -m maxrl.toy --checks-only
python -m maxrl.toy --output runs/toy
```

Routine runs write to the ignored `runs/toy/` directory. The committed `results/toy/` files are reference outputs; last-digit floating-point values and figure bytes can differ across platforms.

The enumerator computes exact finite-group means, squared bias and variance for four-outcome softmax policies. Gradients use four free logits. See the [definitions and check cases](docs/toy.md), [results table](results/toy/moments.csv), [bias and variance figure](results/toy/bias-variance.png), and [batch direction figure](results/toy/batch-direction.png).

## Countdown pilot

Prepare the pinned dataset and grouped split:

```bash
python -m maxrl.data --out data --seed 623
```

Before training, evaluate the initial model with the controlled prompt and verifier:

```bash
python -m maxrl.evaluate --config configs/pilot.json --manifest data/manifest.json \
  --out runs/initial-eval --device cuda --dtype float32 --generation-batch 8 --score-batch 1
```

This generates 1,024 responses without updating weights. Inspect `metrics.json` for pass@1, pass@8, the success-count histogram, and zero/one-success frequencies; inspect saved responses for formatting errors. If almost every group has zero successes, investigate before paying for training. This is a separate sample from the pilot's step-zero evaluation.

Then run the 16-update Count pilot and print its report:

```bash
python -m maxrl.train --config configs/pilot.json --manifest data/manifest.json \
  --out runs/pilot --device cuda --dtype float32 --generation-batch 8 --score-batch 1
python -m maxrl.report --pilot runs/pilot
```

The pilot uses seed 17, a frozen initial sampler, SGD at 0.001, and 32 puzzles with eight responses each per update. Validation uses 128 puzzles with eight responses each at steps 0 and 16: 4,096 training and 2,048 evaluation responses in total.

Set `--device`, `--dtype`, `--generation-batch` and `--score-batch` for the available hardware. Use the same precision and batch settings across matched methods. Add `--hourly-usd` to record the machine's actual hourly price. The [diagnostic guide](docs/diagnostics.md) gives the next commands for the 2,048-response midpoint panel and full-model gradient check.

## Experiment conventions

For a prompt with binary rewards $r_i$, let $K=\sum_i r_i$, $W_i=\pi_\theta(y_i\mid x)/\nu(y_i\mid x)$ and $S_i=\nabla_\theta\log\pi_\theta(y_i\mid x)$. The two primary estimators are

$$g_{\mathrm{Count}}=\frac{\sum_i W_i r_i S_i}{K},\qquad
g_{\mathrm{SNIS}}=\frac{\sum_i W_i r_i S_i}{\sum_i W_i r_i}.$$

Groups with no successes contribute zero and remain in the batch average. Ratios use recorded sampler probabilities over the complete response; weights and denominators are detached. The [methods guide](docs/methods.md) defines every estimator, the fresh finite-group target, and the two distinct clipping operations.

## Guides

| Guide | Contents |
|---|---|
| [Data and response format](docs/data.md) | Verifier, splits, reserved panels and saved token records |
| [Methods](docs/methods.md) | Estimators, objective, gradient and optimizer conventions |
| [Toy experiments](docs/toy.md) | Exact calculation, numerical checks and figure captions |
| [Gradient diagnostics](docs/diagnostics.md) | Collection, scoring, uncertainty and full-model checks |
| [Training experiments](docs/matrix.md) | Tuning, matched seeds, final evaluation and budget |
| [Replicating Li et al.](docs/replication.md) | Li et al.'s setup and six training runs |
| [Sources](docs/sources.md) | Dependencies, source revisions, licenses and related work |
| [Research log](docs/research-log.md) | Dated measurements and verification records |

## Code layout

| Location | Purpose |
|---|---|
| `src/maxrl/estimators.py`, `toy.py`, `toy_plots.py` | Shared coefficients, exact enumeration and figures |
| `src/maxrl/data.py`, `verifier.py`, `schema.py` | Puzzles, rewards and response storage |
| `src/maxrl/prompts.py`, `model.py`, `rollout.py`, `train.py`, `evaluate.py` | Prompt, model loading, sampling, training and evaluation |
| `src/maxrl/diagnostics.py`, `gradient_metrics.py`, `full_gradients.py` | Saved-response gradient comparisons |
| `src/maxrl/matrix.py`, `compare.py`, `budget.py`, `report.py` | Experiment commands, analysis and cost reports |
| `configs/`, `tests/` | Experiment settings and CPU tests |
| `scripts/li_replication.py` | Adapter to the pinned upstream implementation |

The controlled backend uses one device. Run outputs and gradient scratch should use a local disk with hard-link support; copy completed artifacts to shared storage afterward. Profile the pilot before scheduling long runs: the trainer has no resume support. See the [resource notes](docs/matrix.md#storage-and-scheduling) for checkpoint and scratch space. Pretrained-model GPU accuracy, throughput and cost remain to be measured.
