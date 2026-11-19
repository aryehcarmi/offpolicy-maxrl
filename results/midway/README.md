# Countdown midpoint results

Before training, 8.01% and 8.40% of Qwen3-0.6B's responses were correct in two samples from the same 128 Countdown puzzles.
After 16 Count updates at a learning rate of 0.001, it solved none of the 128 puzzles in eight tries each.
Replaying the first update at smaller rates led us to choose 1.5e-7.
We reproduced the failed pilot from its saved responses, then compared Count, SNIS and Uncorrected on those same 16 batches at two rates.
In fresh validation, Count at 1.5e-7 reached 8.79% pass@1 and 33.59% pass@8.
On eight midpoint puzzles, SNIS had lower raw gradient error than Count in the final normalization layer, though the difference was uncertain after matching gradient lengths.

[Report numbers and their sources](report-numbers.md) · [Saved-batch study and midpoint findings](saved-batch-study.md)

## Runs in order

| Run folder | What ran | Status | Headline result | Estimated worker cost |
| --- | --- | --- | --- | ---: |
| [initial-eval/](initial-eval/) | First validation sample from the initial model | used in the report | 82/1,024 correct responses and 40/128 puzzles solved | $0.4736 |
| [pilot-lr-0.001/](pilot-lr-0.001/) | Second initial sample and 16 Count updates | failed and kept for the record | Pass@1 fell from 8.40% to 0.00% | $2.8581 |
| [replay/original/](replay/original/) | Replayed the first update at six rates from 1e-6 to 1e-3 | supporting check | Even at 1e-6, only 86/256 held-out log weights fell within ±0.1 | $0.0430 |
| [replay/low/](replay/low/) | Replayed the first update at seven rates from 5e-8 to 1e-6 | supporting check | At 1.5e-7, 254/256 log weights fell within ±0.1 and the gain was 95.53% of its first-order prediction | $0.0948 |
| [saved-verify/](saved-verify/) | Reproduced the original pilot using its saved batches | supporting check | All 16 gradient norms and all 4,096 log weights matched exactly | $0.1134 |
| [saved-trajectories/](saved-trajectories/) | Compared three methods at 1.5e-7 and 3e-7 on the same batches | used in the report | Six step-16 pass@1 estimates ranged from 8.49% to 8.84% | $1.6032 |
| [pilot-lr-1.5e-7/](pilot-lr-1.5e-7/) | Drew fresh validation responses from the selected Count checkpoint | used in the report | 90/1,024 correct responses and 43/128 puzzles solved | $0.4732 |
| [midpoint-lr-1.5e-7/](midpoint-lr-1.5e-7/) | Compared gradient estimates with a fresh reference on eight puzzles | used in the report | SNIS minus Count in raw final-layer mean squared error was −0.106 (95% interval −0.191 to −0.037) | $0.9512 |

The eight GPU runs cost an estimated **$6.61** in total, including GPU, CPU and memory. The replay workers reported an A10G and the other six reported an A10. The initial evaluation's worker time and cost are in the research log. The other seven are in each run's `attempt.json`. Component timings in `resources.csv` cover the individual operations. [Research log](../../docs/research-log.md) · [Time and cost by run](report-numbers.md#gpu-time-and-estimated-cost)

The [replay folder](replay/) groups the two rate grids and their shared [opening-token counts](replay/batch-2-opening-cluster.json). The low-rate replay retained 98.9579% of the intended update in float32. We waived the original 99% cutoff and kept 1.5e-7, the largest tested rate that met the two remaining criteria. Other subfolders hold inputs or measurements from the run named in their row. [Selection amendment](../../docs/research-log.md)

## Glossary

| Term | Meaning here |
| --- | --- |
| pass@1 | Probability that one sampled response solves a puzzle, estimated here by the fraction of responses that pass across equally sampled puzzles. |
| pass@8 | Fraction of validation puzzles with at least one correct response among eight draws. |
| Importance weight | The probability the current model assigns a whole sampled response divided by the probability the sampling model assigned it, given the same prompt. A weight of 1 means the probability stayed the same. |
| Effective sample size (ESS) | How many equally weighted draws would give the same weight concentration. It equals the squared sum of the weights divided by the sum of their squares. |
| Lag | Number of model updates between the policy that sampled a batch and the policy being evaluated. Responses from the initial model have lag 16 at step 16. |
| Count | Within each puzzle, weight each successful response's gradient by its importance weight, sum those gradients and divide by the number of successes. |
| SNIS | Self-normalized importance sampling. Divide the same weighted gradient sum by the total importance weight of the successes. |
| Uncorrected | Give every successful response equal weight and average their gradients within each puzzle. |

All three methods give a puzzle with no successes a zero contribution to the update. [Method definitions](../../docs/methods.md)

## Adding results

Give each new run its own folder under `results/midway/`, with its attempt record (`attempt.json`), timing file (`resources.csv`) and small result files. Identify the rate in the folder name when it distinguishes pilots or midpoint runs. Add a row to the table in run order. Checkpoints and raw response shards stay on the Modal Volume.

The existing Job 1 launcher will publish the Li et al. initial-model replication as `results/midway/li-initial-<run_id>.json`. Link that file here when it arrives. Its downloaded run log and evaluation artifacts go under the git-ignored `runs/replication/<run_id>/`. [Li replication guide](../../docs/replication.md)
