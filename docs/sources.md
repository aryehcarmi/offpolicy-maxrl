# Provenance and related work

## Controlled environment

The initial CPU checks used CPython 3.12.3 on macOS arm64 and uv 0.13.4. The tested direct dependencies were NumPy 2.2.6, PyArrow 26.0.1, Matplotlib 3.10.7, PyTorch 2.8.0, Transformers 4.57.1 and pytest 8.4.2. The [lockfile](../uv.lock) records transitive versions and hashes. GPU compatibility and performance still need to be measured on the team's hardware.

## External sources

| Component | Pinned source | Use and attribution |
|---|---|---|
| Controlled model and tokenizer | [Qwen3-0.6B at c1899de289a04d12100db370d81485cdf75e47ca](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) | Apache-2.0. Loaded by Transformers with the bundled non-thinking chat template. |
| Controlled data | [Countdown-Tasks-3to4-Unique at 8af28fc6a897ea9f8d343d450e3f3b16b5d4fdfa](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4-Unique/tree/8af28fc6a897ea9f8d343d450e3f3b16b5d4fdfa) | Jiayi Pan's dataset. The pinned card declares no license field. Data are fetched separately. |
| Published replication | [Li, Zhou & Lan at 98e8f9fa15200f294c9395626969f745771ee460](https://github.com/Skylanding/Training-Inference-Mismatch/tree/98e8f9fa15200f294c9395626969f745771ee460) | MIT. The wrapper calls the authors' actual trainer, sampler and evaluator in their checkout. Its LICENSE and file notices travel with that checkout. |
| Controlled prompt | [Countdown prompt at the Li et al. pinned commit](https://github.com/Skylanding/Training-Inference-Mismatch/blob/98e8f9fa15200f294c9395626969f745771ee460/tim/tasks/countdown.py) | System/user wording reproduced in `src/maxrl/prompts.py`. The file credits Martin Marek. [MIT notices](../THIRD_PARTY_NOTICES.md) are retained. |
| Replication Countdown task | [Martin Marek's score-centering at 7c56e9ee2972aa57f446cf564de1a1658d14b321](https://github.com/martin-marek/score-centering/tree/7c56e9ee2972aa57f446cf564de1a1658d14b321) | Li et al. ported this MIT generator and verifier. Their task module retains Marek's notice. |
| Replication runtime | [verl at 8718ca30a3f002f93b7c4fd99b9b2506718681bc](https://github.com/verl-project/verl/tree/8718ca30a3f002f93b7c4fd99b9b2506718681bc), [FA3 at 14c377950125c70b7a9dabf9c561fca53715ac7d](https://github.com/Dao-AILab/flash-attention/tree/14c377950125c70b7a9dabf9c561fca53715ac7d) | Upstream lock, Python 3.12, torch 2.13.0+cu129, vLLM 0.29.0+cu129, Transformers 5.12.1. Linux stack specified by Li et al. |
| Polynomial estimator reference | [MaxRL Statistics at 64a21acb014ee9ae6092242ad731a7985d66827c](https://github.com/nlyu1/maxrl-statistics/tree/64a21acb014ee9ae6092242ad731a7985d66827c) | Mathematical reference. The toy integrator here is independently implemented with Gauss–Legendre quadrature. |

The controlled estimator, data split, exact rational verifier, training, diagnostics and tests are project-specific implementations. Third-party software is installed from its own distribution or cloned at the indicated commit. No upstream source files are vendored; the controlled prompt text is the attributed exception to project-specific wording. The controlled lock is `uv.lock`. The separate upstream lock at `envs/verl-main-cu129/uv.lock` has SHA-256 `a7d016b57e4f0e4055b2d28ee3da2c1c22e2ca7438455dfaf27fa5840d42e79e`.

## Research questions in context

| Existing work | Established idea | How the project uses or tests it |
|---|---|---|
| [Tajwar et al., Maximum Likelihood Reinforcement Learning](https://arxiv.org/html/2602.02710v2) | Finite-sample success conditioning connects ML and RL. | Use the success-only finite-group expectation as the reference. Centered practical MaxRL requires its own target. |
| [Song et al., Training Foundation Models with a Single Algorithm](https://arxiv.org/html/2610.40564v1) | Learning from successful proposals can use trajectories from other generators. | Separate the availability of successes from their response composition under staleness. |
| [Li, Zhou & Lan, Training-Inference Mismatch](https://arxiv.org/html/2610.40945v1) | Staleness and numerical discrepancy can alter the update. Token IS and score centering are published corrections. | Replicate their frozen-sampler accuracy contrast. Measure lag-zero agreement in the controlled stack. |
| [Swaminathan & Joachims, The Self-Normalized Estimator for Counterfactual Learning](https://proceedings.neurips.cc/paper/2015/hash/39027dfad5138c9ca0c474d71db915c3-Abstract.html) | Self-normalization changes finite-sample bias and scale. | Compare raw and self-normalized success weights at the same numerator. |
| [Pandey et al., BRAIn](https://proceedings.mlr.press/v235/pandey24a.html) | Reward-conditioned distribution matching uses samples from a proposal. | Interpret normalization in the conditional-success distribution. |
| [Huang et al., Stable Asynchrony](https://proceedings.mlr.press/v306/huang26ca.html) | Off-policy weight concentration matters for gradient variance. | Measure successful-weight ESS alongside batch gradient direction and error. |
| [Fu et al., AReaL](https://arxiv.org/abs/2505.24298) | Asynchronous rollout and optimization can involve lagged models. | Isolate lag through checkpointed samplers at fixed training updates. |
| [Chen et al., Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) | Estimate pass@k from multiple verified samples per task. | Compute per-puzzle pass@1 and pass@8 on fixed panels and average over puzzles. |

The project tests whether prompt reweighting improves a shared-parameter batch direction, how much single-batch noise persists after global norm matching, and whether the effect predicts training accuracy. The Exact identity and generic self-normalization are established mathematical ingredients. Claims about a new mechanism require independent confirmation and supporting training evidence. [Related leave-one-out derivation](https://nlyu1.github.io/maxrl-statistics/gradient-estimator.html)
