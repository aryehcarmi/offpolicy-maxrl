# Saved-response gradient diagnostics

The target is the prompt-balanced expected update from fresh groups of size $N$, including empty-success groups. Each panel uses the same fixed puzzles across samplers. Every candidate uses the same stale responses and the same independent fresh reference. The `current` pool also serves as the ordinary Fresh comparison.

## Midpoint

Run these commands from the repository root after preparing the split and running the Count pilot. The input is the pilot directory containing `run.json` and the saved checkpoint, together with the prepared data. These artifacts are excluded from Git and must be accessible on the machine running diagnostics.

```sh
python -m maxrl.diagnostics collect --panel midpoint --step 16 --run runs/pilot \
    --manifest data/manifest.json --out runs/midpoint --device cuda --dtype float32 --microbatch 8
python -m maxrl.diagnostics score --pools runs/midpoint --device cuda --dtype float32
python -m maxrl.diagnostics analyze --pools runs/midpoint --group-size 8
python -m maxrl.diagnostics full --pools runs/midpoint --group-size 8 --device cuda --dtype float32
```

Use a reached, saved checkpoint for `--step`. Omit it to use the latest saved pilot checkpoint, including step 8 after an interrupted run. Collection checks the training prompt metadata, split, seed, method and forward dtype. Specify `--dtype bfloat16` consistently across training, collection and scoring if using autocast. `--microbatch` controls generation in `collect` and full-model gradient accumulation in `full`. The default midpoint allocation is eight discovery puzzles × (64 initial + 64 current + 128 additional fresh) = 2,048 responses. RNG streams differ across puzzles, trajectories, panels and pool roles. Changing generation microbatch size can change the draws. The CLI default is four; these examples explicitly use eight. A larger collection microbatch, such as 32, may improve throughput if memory permits. Profile once, then keep the chosen value fixed across corresponding discovery and confirmation pools. Full-gradient accumulation has its own, separate `--microbatch` setting.

## Discovery and independent confirmation

Use Fresh run directories for the predetermined seeds (101 and 102), each containing steps 48 and 64. The two collections use disjoint panels. Both have 32 puzzles, 256 draws per sampler at steps 0, 48 and 64, and 512 additional draws per puzzle for the fresh reference (40,960 total).

```sh
python -m maxrl.diagnostics collect --panel discovery --run runs/matrix/main/fresh-101 \
    --manifest data/manifest.json --out runs/discovery --microbatch 8
python -m maxrl.diagnostics score --pools runs/discovery
python -m maxrl.diagnostics analyze --pools runs/discovery --group-size 8
python -m maxrl.diagnostics full --pools runs/discovery --group-size 8

# After recording the mechanism, predicted ordering, conditions and candidate:
python -m maxrl.diagnostics collect --panel confirmation --run runs/matrix/main/fresh-102 \
    --manifest data/manifest.json --out runs/confirmation --prediction predictions.md --microbatch 8
python -m maxrl.diagnostics score --pools runs/confirmation
python -m maxrl.diagnostics analyze --pools runs/confirmation --group-size 8
python -m maxrl.diagnostics analyze --pools runs/confirmation --group-size 32
python -m maxrl.diagnostics full --pools runs/confirmation --group-size 8
```

The confirmation collection saves a copy of the prespecified prediction. Analysis can regroup either panel at $N=32$ with the same fixed prompt set. The primary comparison uses `initial` at $N=8$ (lag 64). Compare `lag16` for the smaller shift and `current` for numerical consistency. In the latter, inspect the reported log-weight quantiles against the expected zero. Kernel, precision, caching and batching differences can change teacher-forced log probabilities slightly. Raw saved probabilities remain the denominator.

## Files and measurements

- Each pool JSONL is ordered by puzzle, then draw. `manifest.json` stores prompt metadata, split and pool hashes, checkpoint identities, allocation and generation settings. Each stage writes to a new output directory. A relocated training directory can be supplied to `score` and `full` with `--run`.
- Collection, scoring and full-model outputs each include `resources.csv` with elapsed stage and total wall times. Collection records sampled tokens and response counts. Supply `--hourly-usd` to record machine cost. Device, library versions, dtype and GPU model accompany the manifests. RMSNorm analysis records elapsed CPU-stage wall time.
- `scores/*.npz` stores FP32 final-RMSNorm score vectors and FP64 full-response log ratios for each prompt and draw. The score is the derivative of the sum of active response-token log probabilities with respect to the final norm gain. EOS contributes when sampled. Failed responses have zero stored score because their coefficient is zero for every method.
- `analysis-N8/exposure.csv` has one row per (source, puzzle, group). It reports $K$, each method’s average successful coefficient, successful-weight ESS, $D/K$, log inverse-weight summaries for both individual weights and their successful mean, and two clipping rates among successes. Empty groups have blank successful-weight statistics. When $D/K$ exceeds FP64 range, its column is blank and `log_d_over_k` retains its magnitude. Raw cap frequency uses $W>2$. LOO final-coefficient cap frequency uses $W/m>2$.
- `analysis-N8/summary.json` reports the cosine of each mean update with the fresh mean, mean squared Euclidean batch error, norm ratios, zero frequencies, and errors after globally scaling each batch gradient to that batch's SNIS norm. Paired differences compare SNIS with Count, and both candidates with each baseline. A norm-matched difference uses only batches where all compared norms are positive. Raw MSE includes every batch. The K histogram gives counts from zero through $N$.
- If an estimator overflows, `numerical_failures` identifies the affected method and first failing group. Its pool-level comparisons are unavailable. The remaining methods use every group. A metric can overflow even when all gradients are finite. Such a point estimate is null with `point_overflow: true`. An overflow in any bootstrap draw is counted and suppresses the interval for that metric.
- Each batch uses the same contiguous draw indices at every prompt. A fresh reference first averages successes **within** each group, assigns zero to an empty group and averages equally across prompts and groups. This keeps the availability factor in the target.
- The 95% percentile intervals resample complete prompt-balanced batches on the fixed panel. Candidate and reference indices are drawn independently and shared across methods for paired differences. They quantify response sampling conditional on these puzzles and checkpoints. The two disjoint halves of each pool give a separate cosine reliability check. The summary marks a direction unresolved when either split-half interval is undefined, includes zero, or has undefined bootstrap draws. Even an interpretable split-half cosine can be biased. `--bootstrap` and `--seed` control interval simulation.
- `full-initial-N8/summary.json` uses ordinary full-model gradients accumulated over successes with the same detached coefficients. It checks Uncorrected, Count, SNIS, capped LOO and the hybrid on one predetermined batch at midpoint or eight per full panel. Batch indices are fixed at `floor(i * total_groups / selected_groups)`. Twice as many independent fresh groups are used for the full-model reference. Small full-model checks can have a noisier reference than the RMSNorm panel. The midpoint check has one candidate batch and therefore supplies descriptive estimates only.
- During the full check, each aggregate batch gradient is written once in FP32. Blockwise FP64 dot products preserve their inner products for raw errors, cosine and bootstrap analysis. `--scratch /path/to/scratch` selects temporary storage. Budget up to seven full vectors at midpoint and 56 on each larger panel (about 17 GB and 135 GB for 0.6B parameters). Create the scratch directory on a local volume and check free space before starting. Working files are removed on normal exit or Python exceptions, but may remain after forced termination. The runner holds one model and its gradient on the accelerator.
