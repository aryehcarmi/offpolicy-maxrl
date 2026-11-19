# Countdown midpoint report numbers

Rebuild from the repository root with `python -m maxrl.report_numbers`. [Results index](README.md)

## Sampled validation accuracy

Pass@1 counts correct responses. The other columns count puzzles by the number of successes in eight draws. The first two rows are separate samples from the same initial model.

| Model and sample | Pass@1 | Pass@8 | No success | Exactly one | Source |
| --- | --- | --- | --- | --- | --- |
| Initial model, standalone | 82/1,024 (8.01%) | 40/128 (31.25%) | 88/128 (68.75%) | 16/128 (12.50%) | [metrics.json](initial-eval/metrics.json) |
| Initial model, pilot step 0 | 86/1,024 (8.40%) | 43/128 (33.59%) | 85/128 (66.41%) | 17/128 (13.28%) | [metrics.jsonl](pilot-lr-0.001/metrics.jsonl) |
| Count at 0.001, step 16 | 0/1,024 (0.00%) | 0/128 (0.00%) | 128/128 (100.00%) | 0/128 (0.00%) | [metrics.jsonl](pilot-lr-0.001/metrics.jsonl) |
| Count at 1.5e-7, step 16 | 90/1,024 (8.79%) | 43/128 (33.59%) | 85/128 (66.41%) | 17/128 (13.28%) | [metrics.json](pilot-lr-1.5e-7/metrics.json) |

### Importance weights around the first update

The original Count pilot used a learning rate of 0.001. Each range spans responses in one training batch. The pilot logged weights before applying that batch's update. [metrics.jsonl](pilot-lr-0.001/metrics.jsonl)

| When measured | All weights | Weights on successes | All log weights |
| --- | --- | --- | --- |
| Batch 1, before the first update | 0.99975795 to 1.0002788 | 0.99981229 to 1.0001285 | -0.000242076 to 0.000278759 |
| Batch 2, after the first update | 9.9707639e-259 to 2.5736223e-14 | 1.2327696e-165 to 1.3698736e-16 | -594.07 to -31.2909 |

## GPU time and estimated cost

Each row covers a complete GPU worker, including setup, verification and copying results. The estimate includes the requested GPU, CPU and memory. The six trajectories shared one worker.

| GPU run | Reported GPU | Worker seconds | Estimated USD | Source |
| --- | --- | --- | --- | --- |
| Initial model evaluation | NVIDIA A10 | 1,289.05 | $0.4736 | [research-log.md](../../docs/research-log.md) · [resources.csv](initial-eval/resources.csv) |
| Original Count pilot | NVIDIA A10 | 7,779.39 | $2.8581 | [attempt.json](pilot-lr-0.001/attempt.json) |
| First-update replay, original rates | NVIDIA A10G | 117.07 | $0.0430 | [attempt.json](replay/original/attempt.json) |
| First-update replay, low rates | NVIDIA A10G | 258.11 | $0.0948 | [attempt.json](replay/low/attempt.json) |
| Reproduction on saved batches | NVIDIA A10 | 308.59 | $0.1134 | [attempt.json](saved-verify/attempt.json) |
| Six trajectories and probes | NVIDIA A10 | 4,363.71 | $1.6032 | [attempt.json](saved-trajectories/attempt.json) |
| Fresh Count validation | NVIDIA A10 | 1,287.89 | $0.4732 | [attempt.json](pilot-lr-1.5e-7/attempt.json) |
| Midpoint diagnostics | NVIDIA A10 | 2,589.21 | $0.9512 | [attempt.json](midpoint-lr-1.5e-7/attempt.json) |
| **Total** |  | **17,993.02** | **$6.6105** | Sum of the rows above |

For the initial evaluation, the evaluator process alone took 1,265.95 s (1,236.28 s per 1,000 responses) and cost $0.4651 at $1.3226 per hour. [resources.csv](initial-eval/resources.csv)

The **$527 projection** ($526.55) was for the original plan's pilot, screening, tuning and main training runs. Diagnostics and Li replication were unpriced in that projection. [budget.json](pilot-lr-0.001/budget.json)

## Six runs on the same saved batches

The six trajectories reused 16 batches from seed 17. Each batch contained 32 puzzles and 8 responses per puzzle. [run.json](saved-trajectories/pilot/run.json)

Each pass@1 estimate averages reward times the response's importance weight across the two initial-model validation pools. Together they contained 168/2,048 correct responses (8.20%). The last three columns are measured at step 16. The final column is the model's probability of opening with the exact token `<`, averaged over the 128 validation prompts. It started at 23.19%. [trajectories.jsonl](saved-trajectories/trajectories.jsonl) · [summary.json](saved-trajectories/summary.json)

| Method | SGD rate | Pass@1 at step 4 | Step 8 | Step 16 | ESS, all | ESS, successes | P(first token `<`) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Count | 1.5e-07 | 8.27% | 8.35% | 8.54% | 1,871.9 | 155.5 | 13.19% |
| Count | 3e-07 | 8.34% | 8.50% | 8.84% | 1,487.6 | 132.0 | 6.49% |
| SNIS | 1.5e-07 | 8.27% | 8.34% | 8.50% | 1,972.4 | 162.5 | 16.81% |
| SNIS | 3e-07 | 8.33% | 8.47% | 8.77% | 1,895.3 | 157.4 | 14.99% |
| Uncorrected | 1.5e-07 | 8.27% | 8.34% | 8.49% | 2,003.6 | 164.7 | 18.54% |
| Uncorrected | 3e-07 | 8.33% | 8.46% | 8.73% | 1,977.9 | 163.2 | 18.65% |

## Midpoint gradient comparisons

At step 16, the 8 fixed Countdown puzzles supplied 512 initial-model responses, 512 current-model responses and 1,024 independent current-model responses for the fresh reference. [Panel manifest](midpoint-lr-1.5e-7/manifest.json)

Cosine measures alignment of the mean update with the fresh reference. Raw mean squared error (MSE) is the average squared distance of a batch update from that reference. Norm-matched MSE first rescales each update to the SNIS length on its own batch. Length / SNIS averages those per-batch length ratios. Both length comparisons use batches on which both lengths are positive.

Brackets give 95% intervals from 500 bootstrap draws. Each draw resamples whole candidate and reference batches independently on the fixed puzzles and uses the same draws for all methods. [Bootstrap settings](midpoint-lr-1.5e-7/analysis-N8/summary.json) · [Diagnostic definitions](../../docs/diagnostics.md)

### Final RMSNorm gain on initial-model responses

The sampler was at step 0. The analysis used 8 candidate batches and 16 reference batches, with 8 responses per puzzle per batch. [RMSNorm summary](midpoint-lr-1.5e-7/analysis-N8/summary.json)

| Correction | Cosine | Raw MSE | Norm-matched MSE | Length / SNIS |
|---|---:|---:|---:|---:|
| Uncorrected | 0.518 [0.024, 0.590] | 0.333 [0.279, 0.503] | 0.399 [0.343, 0.574] | 0.901 [0.842, 0.950] |
| Count | 0.505 [-0.027, 0.601] | 0.502 [0.389, 0.747] | 0.406 [0.346, 0.586] | 1.100 [1.029, 1.170] |
| SNIS | 0.546 [-0.016, 0.627] | 0.396 [0.333, 0.573] | 0.396 [0.333, 0.573] | 1.000 [1.000, 1.000] |
| Capped Count | 0.505 [-0.027, 0.601] | 0.502 [0.389, 0.747] | 0.406 [0.346, 0.586] | 1.100 [1.029, 1.170] |
| Capped SNIS | 0.546 [-0.016, 0.627] | 0.396 [0.333, 0.573] | 0.396 [0.333, 0.573] | 1.000 [1.000, 1.000] |
| Leave-one-out (LOO) | 0.503 [-0.127, 0.625] | 0.624 [0.430, 0.969] | 0.401 [0.336, 0.586] | 1.226 [1.087, 1.378] |
| Capped LOO | 0.538 [-0.074, 0.636] | 0.559 [0.429, 0.807] | 0.400 [0.337, 0.582] | 1.175 [1.084, 1.256] |
| Hybrid | 0.537 [-0.016, 0.621] | 0.462 [0.359, 0.703] | 0.400 [0.340, 0.577] | 1.059 [1.001, 1.121] |

The paired SNIS minus Count difference in raw MSE was -0.106 [-0.191, -0.037]. Matching lengths on 8 shared nonzero batches gave a difference of -0.010 [-0.029, 0.002]. The fresh reference split-half cosine was 0.366 [-0.206, 0.553]. The direction comparison for SNIS was unresolved. [Paired differences and split-half check](midpoint-lr-1.5e-7/analysis-N8/summary.json)

### Final RMSNorm gain on current-model responses

The sampler was at step 16. The analysis used 8 candidate batches and 16 reference batches, with 8 responses per puzzle per batch. [RMSNorm summary](midpoint-lr-1.5e-7/analysis-N8/summary.json)

| Correction | Cosine | Raw MSE | Norm-matched MSE | Length / SNIS |
|---|---:|---:|---:|---:|
| Uncorrected | -0.154 [-0.501, 0.364] | 0.749 [0.592, 1.093] | 0.749 [0.592, 1.093] | 0.999992 [0.999983, 1.000000] |
| Count | -0.154 [-0.501, 0.364] | 0.749 [0.592, 1.093] | 0.749 [0.592, 1.093] | 1.000024 [0.999983, 1.000062] |
| SNIS | -0.154 [-0.501, 0.364] | 0.749 [0.592, 1.093] | 0.749 [0.592, 1.093] | 1.000 [1.000, 1.000] |
| Capped Count | -0.154 [-0.501, 0.364] | 0.749 [0.592, 1.093] | 0.749 [0.592, 1.093] | 1.000024 [0.999983, 1.000062] |
| Capped SNIS | -0.154 [-0.501, 0.364] | 0.749 [0.592, 1.093] | 0.749 [0.592, 1.093] | 1.000 [1.000, 1.000] |
| Leave-one-out (LOO) | -0.154 [-0.501, 0.364] | 0.749 [0.592, 1.093] | 0.749 [0.592, 1.093] | 1.000023 [0.999991, 1.000055] |
| Capped LOO | -0.154 [-0.501, 0.364] | 0.749 [0.592, 1.093] | 0.749 [0.592, 1.093] | 1.000023 [0.999991, 1.000055] |
| Hybrid | -0.154 [-0.501, 0.364] | 0.749 [0.592, 1.093] | 0.749 [0.592, 1.093] | 1.000020 [0.999989, 1.000051] |
| Fresh | -0.154 [-0.501, 0.364] | 0.749 [0.592, 1.093] | 0.749 [0.592, 1.093] | 0.999992 [0.999983, 1.000000] |

### Full model on initial-model responses

The full-model check used 1 candidate batch and 2 reference batches. A single candidate batch gives point estimates only. [Full-model summary](midpoint-lr-1.5e-7/full-initial-N8/summary.json)

| Correction | Cosine | Raw MSE | Norm-matched MSE | Length / SNIS |
|---|---:|---:|---:|---:|
| Uncorrected | 0.097 | 249,838 | 247,906 | 1.006 |
| Count | 0.084 | 286,440 | 251,394 | 1.106 |
| SNIS | 0.113 | 243,770 | 243,770 | 1.000 |
| Capped LOO | 0.128 | 253,935 | 239,762 | 1.046 |
| Hybrid | 0.112 | 250,184 | 243,877 | 1.020 |

## Replay and first-update checks

The first-batch gradient norm was 361.03. Replaying the collapsed pilot gave a maximum absolute gradient-norm difference of 0 across 16 updates and a maximum absolute log-weight difference of 0 across 4,096 responses. [Per-update verification](saved-verify/verification.json)

At the original grid's smallest rate (1e-06), the first update realized 77.90% of its predicted objective increase. Only 86/256 held-out log-weights fell within ±0.1. [Original rate grid](replay/original/rates.csv)

At the selected rate of 1.5e-07, the first update realized 95.53% of its predicted objective increase. 254/256 batch-2 log-weights (99.22%) fell within ±0.1. The float32 step retained 98.9579% of the intended displacement along its direction. The study waived the original 99% rounding threshold. Across 428 successes in the other saved batches, the mean weight was 1.01068 (standard error 0.00239, clustered by puzzle within batch). [Rate table](replay/low/rates.csv) · [Selected checkpoint](saved-trajectories/summary.json) · [Amendment](../../docs/research-log.md)

In the first Count training batch, coefficient-weighted successes opened with `<` 12.71% of the time. The model assigned it 24.04% on the same weighted prompts, for a difference of -11.33 percentage points. [Per-update opening measurements](saved-trajectories/trajectories.jsonl)

Number of batches with a negative opening-token difference at each rate. [Per-update measurements](saved-trajectories/trajectories.jsonl)

| SGD rate | Count | SNIS | Uncorrected |
|---:|---:|---:|---:|
| 1.5e-07 | 13/16 | 9/16 | 9/16 |
| 3e-07 | 13/16 | 9/16 | 9/16 |

At 1e-06, 73/73 responses in the lower cluster of batch 2 log-weights opened with `<` followed by `answer`. That prefix appeared in 0/183 other responses. The table splits the mean change in log probability after one update into contributions from the first and remaining tokens, in nats. [Opening counts](replay/batch-2-opening-cluster.json) · [Token-level decomposition](saved-trajectories/probes/first-token-summary.json)

| First token | Responses | First token change | Remaining change | Whole-response change |
|---|---:|---:|---:|---:|
| `<` | 73 | -0.5958 | +0.0034 | -0.5924 |
| Other | 183 | +0.1265 | -0.0050 | +0.1214 |

The last column counts parameters whose float32 value changed on the first Count step at 1.5e-07. The embedding and output head share weights, counted once. Normalization includes query and key gains. [Parameter counts and gradient norms](saved-trajectories/probes/parameter-types.json)

| Parameter type | Share of squared gradient norm | Parameters of this type that changed |
|---|---:|---:|
| Tied embedding | 2.85% | 0.23% |
| Normalization | 0.15% | 3.64% |
| Attention | 44.54% | 42.64% |
| MLP | 52.47% | 38.35% |
