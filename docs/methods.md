# Estimators and training conventions

## Finite-group target

For one prompt $x$, the sampler $\nu$ generates $N$ independent responses. For response $i$,

$$r_i\in\{0,1\},\qquad W_i=\frac{\pi_\theta(y_i\mid x)}{\nu(y_i\mid x)},\qquad S_i=\nabla_\theta\log\pi_\theta(y_i\mid x).$$

Here $\nu$ is the distribution that generated the response. Let $K=\sum_i r_i$ and $D=\sum_i W_i r_i$. Each estimator uses

$$\widehat g_x=\frac{1}{K}\sum_{i:r_i=1}\lambda_i S_i$$

when $K>0$, and zero otherwise. The batch update averages over all prompts, including those with no successes.

The reference is the expected success-only update from $N$ fresh responses:

$$G_N^{\mathrm{fresh}}=\left[1-(1-p_\theta)^N\right]\nabla_\theta\log p_\theta,$$

where $p_\theta$ is the current policy's success probability. An empirical reference averages fresh groups of size $N$, keeping the zero updates.

## Coefficients

Define $v_i=\min(W_i,2)$. For each successful response, let $m_i$ be the arithmetic mean of the other successful weights; use $m_i=1$ when $K=1$.

| Method key | Coefficient $\lambda_i$ on a success |
|---|---|
| `fresh` | $1$, with responses sampled from the current policy |
| `uncorrected` | $1$, with responses sampled from the stale policy |
| `count` | $W_i$ |
| `snis` | $KW_i/D$ |
| `count_cap` | $v_i$ |
| `snis_cap` | $Kv_i/\sum_j r_jv_j$ |
| `loo` | $W_i/m_i$ |
| `loo_cap` | $\min(W_i/m_i,2)$ |
| `hybrid` | $W_i$ when $K=1$; SNIS when $K\ge2$ |

`count_cap` and `snis_cap` apply the same cap to raw importance weights. `loo_cap` caps the final coefficient after dividing by the peers' mean. These are different operations. The implementation computes peer means with prefix and suffix log-sums to avoid subtracting a dominant weight from a nearly equal total.

All eight stale estimators are available for offline comparisons. The training configuration excludes uncapped LOO. The signed Exact estimator is used only in the [toy experiments](toy.md).

For each nonempty group,

$$g_{\mathrm{Count}}=(D/K)g_{\mathrm{SNIS}}.$$

This identity explains why batch direction matters: different prompts can receive different multipliers. The diagnostics report the direction of the mean update, raw batch error, error after matching the SNIS batch norm, and total success weight. Their [uncertainty estimates](diagnostics.md) are conditional on the fixed puzzle panel and checkpoints.

## Sampling and optimization

Controlled experiments use Qwen3-0.6B at temperature one, without top-k or top-p filtering, and with a 377-token response limit. The pinned Countdown messages in `src/maxrl/prompts.py` use the original system instruction and user wording from [Li et al.'s task module](https://github.com/Skylanding/Training-Inference-Mismatch/blob/98e8f9fa15200f294c9395626969f745771ee460/tim/tasks/countdown.py). The user message asks for work in `<think>` tags and an expression in `<answer>` tags, with a fixed format example. No puzzle solution is included. The tokenizer still uses `enable_thinking=False`; the instruction to show work does not change that chat-template setting. The strict exact-arithmetic verifier is unchanged.

Every controlled run, saved checkpoint, diagnostic panel and learning-rate freeze records the prompt identifier and a digest of its wording and template options. Tuning, checkpoint evaluation, diagnostic collection and final comparison reject incompatible or missing prompt metadata. Keep results from older prompts in separate directories; do not relabel them to pass these checks. Stored response tokens remain the input to gradient replay.

Both EOS IDs declared by the model are accepted. A sampled EOS is included in the sequence probability; a response stopped by the length limit need not contain EOS.

Compute log importance ratios by summing current minus recorded sampler log probabilities over active response tokens. Save the actual probabilities at sampling time. Equal model weights can still yield small replay differences from caching, kernels or precision; inspect the lag-zero diagnostic before interpreting stale ratios.

Scores sum response-token log probabilities. Weights and denominators are detached. The loss is the negative prompt-average update, followed by one SGD step with no momentum or weight decay. There is no control variate in the controlled estimators. Practical centered MaxRL has a different finite-group reference and is outside the main comparison.

Validation and testing average per-puzzle pass@1 and pass@8. For $c$ successes in $m$ responses, the pass@k estimate is

$$1-\frac{\binom{m-c}{k}}{\binom{m}{k}}.$$

See [sources](sources.md) for the MaxRL, self-normalization and pass@k references, and the [training guide](matrix.md) for tuning and comparison rules.
