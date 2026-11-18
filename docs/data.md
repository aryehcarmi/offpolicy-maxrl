# Countdown data and saved responses

## Dataset and verifier

The controlled experiments use the pinned revision of [Jiayi-Pan/Countdown-Tasks-3to4-Unique](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4-Unique/tree/8af28fc6a897ea9f8d343d450e3f3b16b5d4fdfa). Prepare it with:

```bash
python -m maxrl.data --out data --seed 623
```

The command checks the Parquet SHA-256 and writes a manifest and JSONL shards. For an existing copy, add `--parquet /path/to/train-00000-of-00001.parquet`. The output directory must not already exist.

Each puzzle has three or four numbers and a target. A valid response ends in a single `<answer>expression</answer>`, uses each supplied integer exactly once, and reaches the target using binary `+`, `-`, `*`, `/` and parentheses. The verifier uses exact fractions. Negative and fractional intermediate values are allowed. The controlled experiments use the released data without a difficulty or solvability filter.

## Splits and panels

A puzzle's identity is its sorted multiset of numbers and target. Duplicate identities collapse to the lexicographically first observed ordering of the inputs. Splits are seeded, grouped by this identity, and stratified by the number of inputs.

| Population or panel | Allocation |
|---|---|
| Training population | 90% of unique puzzles |
| Development population | 5% of unique puzzles |
| Test population | 5% of unique puzzles |
| Validation panel | 128 development puzzles |
| Discovery panel | 32 development puzzles |
| Confirmation panel | 32 development puzzles, disjoint from discovery and validation |
| Final test panel | 256 test puzzles |
| Midpoint panel | First eight discovery puzzles |

Seed 623 yields 404,613 training, 22,479 development and 22,478 test puzzles. The prepared manifest has digest `abdb27fbeeefdae2c9f6cce441811af401e07937f9b6bd4693f8a5cabb556186`. Runs record its hash, and readers verify panel hashes before loading. Final test evaluation follows the frozen method and learning-rate selections.

Prepared data are excluded from Git. When moving them to another machine, copy `manifest.json` together with all its referenced JSONL shards, or regenerate them with the pinned source and the same split seed and verify the manifest digest.

Li et al.'s [replication](replication.md) uses its own generated data and split.

## Response records

Each version-1 JSONL row contains:

| Fields | Meaning |
|---|---|
| `sample_id`, `puzzle_id`, `pool` | Response, puzzle and sampling-pool identities |
| `checkpoint_id` | Policy checkpoint that generated the response |
| `prompt_ids`, `response_ids` | Original prompt and response token IDs |
| `sampler_logprobs` | Natural-log token probabilities recorded during sampling |
| `response_mask`, `eos_mask` | Active response tokens and sampled EOS position |
| `reward`, `schema_version` | Binary correctness and record format version |

The three token-level arrays align with `response_ids`. The response mask is a nonempty contiguous prefix. If EOS was sampled, it is the last active token and contributes to the sequence score. Padding contributes zero log probability.

Preserve the original token IDs, masks and recorded sampler probabilities. Run metadata also identifies the prompt wording; never combine these records with runs that used a different prompt. Decoded text alone is insufficient for likelihood-ratio replay. Store each shard with its split manifest, run configuration and checkpoint identity. The reader validates the schema and rejects duplicate sample IDs within a shard.
