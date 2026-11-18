# Controlled experiment matrix

Run each command from the repository root. `screen`, `tune`, `main` and `test` print one shell command per GPU job. Execute the printed jobs for each stage before selecting the next grid. The planner accepts `--root`, `--configs`, `--manifest`, `--device`, `--dtype`, `--generation-batch`, `--score-batch` and `--hourly-usd`.

```sh
python -m maxrl.matrix screen
python -m maxrl.matrix tune
python -m maxrl.matrix freeze
python -m maxrl.matrix main
# After the main runs finish and the comparisons are fixed
python -m maxrl.matrix test
```

The Fresh screen uses seed 17 and chooses the best step-16 validation pass@1 across learning rates 0.0001, 0.001 and 0.01. All four methods then run for 96 updates over the common grid of one third, one, and three times the selected rate. Their scores are mean validation pass@1 at steps 64, 80 and 96. Ties select the smaller learning rate in both stages.

`freeze` validates the recorded configurations, manifest fingerprint, panel size, validation steps and failure records. It writes the full grid, selected rates, matched seeds, main configuration and primary contrast to `runs/matrix/frozen.json`. A poor finite score is a valid trial. Missing metrics and operational failures require completion or a documented rerun. Keep failed attempts in a separate directory before rerunning a job at its original path. Numerical failures remain in the frozen record. The trainer has no automatic resume option; an interrupted trajectory remains partial. In `failure.json`, `target_step` is the loop step before that iteration's update: a failure during `training_update` at `target_step: 0` means update 1 was attempted. Use `phase` and `last_saved_checkpoint_step` to identify the failure and the available checkpoint.

`main` emits 128-update commands for Count and SNIS on seeds 101–105, and Fresh and Uncorrected on seeds 101–103. The first two Fresh trajectories supply discovery and confirmation. `test` independently emits the 256-puzzle, 16-response evaluation of each final checkpoint using the same seed as its trajectory. Review any failed main run before dispatching its evaluation. The planner checks the frozen protocol and manifest before emitting test commands and never reads test response files.

## Numerical settings

Choose one dtype, generation batch size and scoring batch size before the screening runs, and pass those flags consistently to each matrix stage. The defaults are `--dtype float32 --generation-batch 8 --score-batch 1`. Trial selection checks these settings in every recorded screen and tuning run. The version-3 `frozen.json` records them and the prompt metadata with the selected learning rates.

Main commands and the final comparison require training records to match all three frozen settings. Final evaluation must match the frozen dtype and generation batch size; its scoring batch size is unused. Device indices and machine prices can differ and remain in the run metadata. Hardware and dependency versions should still be recorded and reviewed when comparing results.

Missing runtime or prompt metadata is an error. Prompt wording and chat-template options must match the current protocol; new wording requires new tuning and training runs. To replace an older freeze, archive it and regenerate it from the recorded trials after checking their settings. Do not fill missing fields with assumed defaults. The final comparison also checks runtime metadata, so changing a printed command cannot silently change the numerical setup of the headline result.

## Numerical failures and optional methods

The default comparison uses uncapped Count and SNIS. If either primary method fails numerically across the full tuning grid, check the implementation and use one shared smaller-rate extension. Persistent failure permits a separate comparison using the same raw-weight cap for both methods, with comparable retuning and the original failures retained. Poor validation accuracy or a large gradient-norm ratio alone does not trigger clipping. The command planner implements the uncapped design; any capped comparison must be documented separately.

At most one of `loo_cap` and `hybrid` can enter further training. Record the prediction from discovery before collecting confirmation results. The candidate must show a mean-direction advantage over both Count and SNIS on the confirmation panel, supported by paired uncertainty and a full-model gradient check. Report the accompanying noise tradeoff. It must also achieve a better stable validation tuning score than both primary methods. Prefer `hybrid` if discovery does not separate the candidates. Five additional training seeds require spare budget after the core experiments are funded.

## Results

After all ten primary evaluations finish, compute the prespecified contrast from the saved response shards.

```sh
python -m maxrl.compare --root runs/matrix > runs/matrix/primary-result.json
```

The report gives each method’s pass@1 and pass@8 on every seed, all five SNIS-minus-Count pass@1 differences and their mean. Its paired two-sided 95% Student t interval (4 degrees of freedom) treats the five seed differences as independent, approximately normal observations conditional on the test panel. A separate percentile interval holds the five trajectories fixed and resamples paired puzzle IDs within puzzle size, preserving the test panel’s size mix and reusing each draw across all methods and seeds. This interval conditions on the collected responses for each puzzle. Values are in probability units. Multiply by 100 for percentage points. An interval spanning zero leaves the direction of the primary effect unresolved.

Under the normal-difference assumption, five independent seed pairs give 80% power at a two-sided 5% significance level for an effect of about 1.68 standard deviations of the paired differences. The corresponding percentage-point effect depends on that paired variation. Secondary outcomes are pass@8, learning curves, numerical failures and the largest validation decline.

## Budget

The default plan schedules **1,353,216 responses**, including the pilot and both full diagnostic panels. The separate initial controlled evaluation adds 1,024 responses; it is outside the default matrix count.

```sh
python -m maxrl.budget
python -m maxrl.budget --pilot runs/pilot
```

Record the actual machine price with `--hourly-usd` during the pilot or supply it to the budget command. The projection uses measured generation, response lengths, backward/update time, evaluation and overhead. It remains partial until `--diagnostic-usd` and `--replication-usd` supply separate estimates for the total costs of the diagnostic and replication workloads. Reprice if response lengths, hardware or success rates change materially.

The **$1,500 ceiling** allocates **$1,200 to planned work** and initially reserves **$300 for recovery and reruns**. Drop optional work first. To price the reduced-control scenario, run:

```sh
python -m maxrl.budget --pilot runs/pilot --reduced-controls
```

This scenario retains all five Count/SNIS pairs, two Fresh trajectories for discovery and confirmation, and one descriptive Uncorrected run. The flag changes the cost projection; it does not change the jobs printed by `maxrl.matrix`.


## Storage and scheduling

Use local storage for checkpoints, response files and gradient scratch. Response publication uses hard links to avoid overwriting an existing shard; some cloud-drive and network mounts do not support them. Copy completed files to shared storage after the process exits.

The controlled configurations save 65 model checkpoints across the pilot, screen, tuning and main runs. At roughly 2.4 GB per FP32 checkpoint, allow about 160 GB, plus model downloads, response files, data and replication outputs. The upstream replication saves extra training state and is not included in that estimate.

A full-model diagnostic holds seven temporary FP32 gradient vectors at midpoint and 56 for a larger panel: approximately 17 GB or 135 GB for 0.6B parameters. Point `full --scratch` to a local volume with enough free space. Temporary vectors are removed on normal exit and ordinary Python exceptions; forced termination can leave files behind. See the diagnostic guide for the command.

Generation handles one puzzle at a time and waits for the longest response in each sampling microbatch. Use the measured `resources.csv` to estimate elapsed time and cost; a nominal per-token GPU speed is insufficient. The trainer cannot resume interrupted trajectories. Prefer uninterrupted jobs for the main runs, and choose concurrent jobs only after checking per-job memory and total cost.
