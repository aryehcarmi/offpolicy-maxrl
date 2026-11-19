# Countdown training on 16 frozen batches

All six trajectories started at the pinned Qwen3-0.6B weights and reused the same 16 seed-17 batches of 32 puzzles × 8 responses. The original Count run at 0.001 reproduced all 16 gradient norms and all 4,096 per-response log importance weights exactly on an A10. The amendment retained 1.5e-7 after its replay measured 98.9579% projected retention in float32. [Amendment](../../docs/research-log.md) · [Reproduction checks](saved-verify/verification.json)

## Validation

Each importance-sampling estimate averages reward times the full-response likelihood ratio over the two saved validation pools. Each pool contains eight original-model draws per puzzle for the same 128 puzzles. The initial pooled success frequency was 168/2,048 (8.20%). The initial probability of the exact opening token `<` was 23.19%.

| Method | SGD rate | Estimated pass@1, step 4 | Step 8 | Step 16 | Step-16 ESS / 2,048 | Step-16 success ESS / 168 | Step-16 P(first token `<`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Count | 1.5e-07 | 8.27% | 8.35% | 8.54% | 1871.9 | 155.5 | 13.19% |
| Count | 3e-07 | 8.34% | 8.50% | 8.84% | 1487.6 | 132.0 | 6.49% |
| SNIS | 1.5e-07 | 8.27% | 8.34% | 8.50% | 1972.4 | 162.5 | 16.81% |
| SNIS | 3e-07 | 8.33% | 8.47% | 8.77% | 1895.3 | 157.4 | 14.99% |
| Uncorrected | 1.5e-07 | 8.27% | 8.34% | 8.49% | 2003.6 | 164.7 | 18.54% |
| Uncorrected | 3e-07 | 8.33% | 8.46% | 8.73% | 1977.9 | 163.2 | 18.65% |

Fresh seed-18 validation of Count at 1.5e-7 produced 90/1,024 correct responses (8.79% pass@1) and solved 43/128 puzzles (33.59% pass@8). The six table rows share the same saved draws. [Fresh evaluation](pilot-lr-1.5e-7/metrics.json) · [All 96 updates and individual-pool estimates](saved-trajectories/trajectories.jsonl)

## Opening tokens and parameter changes

On the first Count batch, 12.71% of the coefficient-weighted successes opened with `<`. The model assigned it 24.04% on the same prompts with the same coefficient weighting. The difference was −11.33 percentage points. The normalized push was negative on 13 of 16 batches for Count at either rate, and on nine for each SNIS and Uncorrected trajectory. The per-update file includes each signed push, its coefficient mass, and the success-weight mean, quantiles and effective sample size. [Per-update measurements](saved-trajectories/trajectories.jsonl)

For the 73 answer-first responses in batch 2, a single Count update at 1e-6 changed whole-response log probability by −0.5924 nats on average. The first token contributed −0.5958 nats, and later tokens offset it by +0.0034. The other 183 responses averaged +0.1265 nats on the first token and −0.0050 on later tokens. These changes use the same response tokens before and after the update. [All 256 decompositions](saved-trajectories/probes/first-token.jsonl) · [Group summaries](saved-trajectories/probes/first-token-summary.json)

| Parameter type | Share of squared first-batch gradient norm | Fraction of that type changed in float32 at 1.5e-7 |
|---|---:|---:|
| Tied input/output embedding | 2.85% | 0.23% |
| Normalization gains | 0.15% | 3.64% |
| Attention | 44.54% | 42.64% |
| MLP | 52.47% | 38.35% |

Tied weights are counted once. The normalization row includes the query and key normalization gains. [Gradient and parameter counts](saved-trajectories/probes/parameter-types.json)

## Midpoint diagnostics

At the selected checkpoint the eight midpoint puzzles supplied 512 initial-model responses, 512 current-model responses and 1,024 independent current-model reference responses. The table compares methods on the initial-model responses against the fresh reference. The RMSNorm analysis uses eight candidate batches and 16 reference batches. The full-model analysis uses one predetermined candidate batch and two reference batches.

| Method | Final RMSNorm cosine | Final RMSNorm raw MSE | Full-model cosine | Full-model raw MSE |
|---|---:|---:|---:|---:|
| Uncorrected | 0.518 | 0.333 | 0.097 | 249,838 |
| Count | 0.505 | 0.502 | 0.084 | 286,440 |
| SNIS | 0.546 | 0.396 | 0.113 | 243,770 |

On this fixed panel, SNIS minus Count had a paired raw RMSNorm MSE difference of -0.106 (95% bootstrap interval -0.191 to -0.037). After matching gradient norms, that difference was -0.010 (-0.029 to 0.002). The fresh RMSNorm reference split-half cosine interval included zero, so the direction comparison remains unresolved. The single full-model candidate batch gives descriptive estimates only. [RMSNorm analysis](midpoint-lr-1.5e-7/analysis-N8/summary.json) · [Full-model analysis](midpoint-lr-1.5e-7/full-initial-N8/summary.json) · [Exposure table](midpoint-lr-1.5e-7/analysis-N8/exposure.csv) · [Panel and checkpoint manifest](midpoint-lr-1.5e-7/manifest.json)

## Run records

All four GPU attempts used a clean checkout of `438cc41115be59c3eabfcc77745a72af076d411e` and an NVIDIA A10 with two CPU cores and 16 GiB of memory. The price was $1.3226 per machine-hour. Times below include each GPU worker’s setup, verification and output copy.

| Stage | GPU worker seconds | Estimated USD |
|---|---:|---:|
| [Pilot reproduction](saved-verify/attempt.json) | 308.6 | $0.1134 |
| [Six trajectories and probes](saved-trajectories/attempt.json) | 4,363.7 | $1.6032 |
| [Fresh validation](pilot-lr-1.5e-7/attempt.json) | 1,287.9 | $0.4732 |
| [Midpoint diagnostics](midpoint-lr-1.5e-7/attempt.json) | 2,589.2 | $0.9512 |
| **Total** | **8,549.4** | **$3.1410** |

The selected checkpoint is stored on Volume `countdown-maxrl` at `saved-train-20261118T235843324334Z-07a21baa5fbb/output/pilot/checkpoints/step-000016/`. [Checkpoint identity](saved-trajectories/pilot/run.json) · [Fresh validation timing](pilot-lr-1.5e-7/resources.csv) · [Midpoint collection timing](midpoint-lr-1.5e-7/resources.csv) · [Scoring timing](midpoint-lr-1.5e-7/scores/resources.csv) · [Full-model timing](midpoint-lr-1.5e-7/full-initial-N8/resources.csv)
