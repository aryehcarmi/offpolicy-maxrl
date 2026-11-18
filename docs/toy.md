# Four-outcome softmax experiments

```bash
python -m maxrl.toy --checks-only
python -m maxrl.toy --output runs/toy
```

Routine output goes to `runs/toy/`, which is ignored by Git. `results/toy/` retains the reference CSV and figures. Numerical checks use tolerances; byte-for-byte agreement is not expected across platforms.

## Model and exact calculation

The outcomes are ordered $00,01,10,11$, with rewards $(1,0,0,1)$ and probabilities

$$\pi(p,s)=\left(ps,\frac{1-p}{2},\frac{1-p}{2},p(1-s)\right).$$

Here $p$ is the success probability and $s$ is the first correct outcome's share of successes. These values specify distributions. Gradients are taken with respect to four free softmax logits, giving $S_j=e_j-\pi$.

The enumerator sums over outcome counts with multinomial probability $N!\prod_j\nu_j^{c_j}/c_j!$. Count vectors with identical successful counts share an update. Fresh samples use probabilities under the current policy.

Write $u=E_\pi[S\mid r=1]$ and $q(p)=1-(1-p)^N$. The expected fresh update is $q(p_\theta)u$. Since $E_\nu[WS\mid r=1]=(p_\theta/p_\nu)u$, Count has mean $q(p_\nu)(p_\theta/p_\nu)u$. It can therefore preserve one prompt's mean direction while changing the relative weights of different prompts.

The toy-only Exact estimator is

$$\widehat g_{\mathrm{Exact}}=
\sum_i W_i r_i S_i\int_0^1\prod_{j\ne i}(1-tW_jr_j)\,dt.$$

For $K$ successes, the integrand has degree at most $K-1$. A Gauss–Legendre rule with $\lceil K/2\rceil$ nodes integrates it up to floating-point error. Signed terms are retained. Independence gives

$$E_\nu[\widehat g_{\mathrm{Exact}}]
=Np_\theta u\int_0^1(1-tp_\theta)^{N-1}\,dt
=q(p_\theta)u.$$

The code checks this identity against the enumerated mean. At unit weights each successful term integrates to $1/K$, so Exact agrees with the ordinary success average for every on-policy group. Empty-success groups contribute zero.

## Check cases

The first panel of `bias-variance.png` uses $N=2$, $p_\nu=0.05$, $p_\theta=0.20$ and $s_\nu=s_\theta=0.50$. Count has lower squared bias than SNIS (0.000576 versus 0.0441), but greater MSE (1.67163 versus 0.14854).

The second panel uses $N=8$, $p_\nu=p_\theta=0.20$, $s_\nu=0.20$ and $s_\theta=0.80$:

| Estimator | Uncapped MSE | Cap operation | Capped MSE |
|---|---:|---|---:|
| Count | 1.519272 | Raw weight at 2 | 0.499363 |
| SNIS | 0.761393 | Raw weight at 2 | 0.755848 |
| LOO | 7.973237 | Final coefficient at 2 | 0.684231 |

Raw-weight capping changes both the numerator and denominator of SNIS. Its capped and uncapped results are distinct estimator variants. LOO's cap is applied after peer normalization. See the [coefficient definitions](methods.md).

## Grid and output fields

`moments.csv` records the mean and target vectors, squared Euclidean bias, trace variance, their sum (MSE), and the cosine between the mean and target. Zero updates contribute to every expectation.

The 27 single-prompt conditions cross $N\in\{2,8,32\}$, sampler success probabilities $p_\nu\in\{0.20,0.15,0.05\}$ with $p_\theta=0.20$, and composition shifts $0.50\to0.50$, $0.45\to0.55$ and $0.20\to0.80$. This grid covers unchanged and increasing success rates.

## Two-prompt geometry

Each batch contains independent groups for two equally weighted prompts. Prompt 2 swaps logit coordinates 2 and 4 (one-based), under both policies. The separate-parameter calculation concatenates the local updates into eight coordinates and divides by two. The shared-parameter calculation uses $(g_1+P^Tg_2)/2$. Both use the same per-prompt sampling distributions. Independence makes the batch trace variance one quarter of the sum of the prompt trace variances in either geometry.

`batch-direction.png` plots the angle of the expected batch update to the expected fresh update at $N=8$. Zero means agreement in direction. The two conditions are:

| Marker | Prompt 1: sampler to current $(p,s)$ | Prompt 2: sampler to current $(p,s)$ |
|---|---|---|
| Blue circle | $(0.05,0.50)\to(0.20,0.50)$ | $(0.50,0.05)\to(0.50,0.20)$ |
| Orange cross | $(0.05,0.20)\to(0.20,0.80)$ | $(0.485,0.02062)\to(0.56,0.28571)$ |

Prompt 2's shifts follow from the coordinate permutation; its composition changes in both conditions. The CSV's $p$ and $s$ fields describe prompt 1. The figure shows the primary methods and the capped LOO and singleton-hybrid candidates. All estimators remain in the results table.
