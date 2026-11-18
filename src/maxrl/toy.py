"""Exact finite-group moments for four free softmax logits and binary rewards."""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np

from .estimators import METHODS, coefficients


SUCCESS = [0, 3]  # 00, 01, 10, 11
SWAP = [0, 3, 2, 1]
TOY_METHODS = (*METHODS, "exact")


def policy(p, s):
    """p sets the success rate, s = P(00 | success). Both are interior."""
    if not (0 < p < 1 and 0 < s < 1):
        raise ValueError("p and s must lie strictly between zero and one")
    return np.array([p*s, (1-p)/2, (1-p)/2, p*(1-s)])


def fresh_target(pi, n):
    p = pi[SUCCESS].sum()
    score = np.eye(4) - pi
    availability = -np.expm1(n * np.log1p(-p))
    return availability * ((pi[SUCCESS] / p) @ score[SUCCESS])


@lru_cache(maxsize=None)
def count_catalog(n):
    """Each row represents n! / prod(c_j!) ordered samples."""
    if n < 1:
        raise ValueError("n must be positive")
    rows = [(a, b, c, n-a-b-c) for a in range(n+1)
            for b in range(n-a+1) for c in range(n-a-b+1)]
    factors = [math.factorial(i) for i in range(n+1)]
    multiplicities = [factors[n] // math.prod(factors[i] for i in row) for row in rows]
    return np.asarray(rows), np.asarray(multiplicities, dtype=np.float64)


@lru_cache(maxsize=None)
def _integration_rule(k):
    # Each leave-one-out integrand has degree <= K-1. An m-node Gauss rule
    # integrates through degree 2m-1; the only approximation is floating point.
    x, w = np.polynomial.legendre.leggauss((k+1)//2)
    return (x+1)/2, w/2


def exact_update(counts, success_weights, success_scores):
    """Integrate the leave-one-out product for a fixed pair of success counts."""
    k = sum(counts)
    update = np.zeros(4)
    if k == 0:
        return update
    nodes, integration_weights = _integration_rule(k)
    factors = 1 - nodes[:, None] * success_weights
    for j, count in enumerate(counts):
        if count:
            exponents = np.array(counts)
            exponents[j] -= 1
            integrand = np.prod(factors ** exponents, axis=1)
            integral = integration_weights @ integrand
            update += count * success_weights[j] * success_scores[j] * integral
    return update


@dataclass
class Moments:
    mean: np.ndarray
    target: np.ndarray
    variance: float  # trace of the covariance

    @property
    def squared_bias(self):
        return float(np.sum((self.mean - self.target)**2))

    @property
    def mse(self):
        return self.variance + self.squared_bias

    @property
    def cosine(self):
        norm = np.linalg.norm(self.mean) * np.linalg.norm(self.target)
        return float(np.clip(self.mean @ self.target / norm, -1, 1)) if norm else None


def enumerate_moments(n, nu, pi):
    """Use multinomial masses over four-outcome counts. Fresh draws use pi."""
    rows, multiplicities = count_catalog(n)
    # All allocations of the two failure outcomes with the same successful
    # counts give the same score update. Sum their multinomial masses once.
    successes, index = np.unique(rows[:, SUCCESS], axis=0, return_inverse=True)
    index = index.reshape(-1)

    def masses(distribution):
        probs = multiplicities * np.prod(distribution ** rows, axis=1)
        return np.bincount(index, weights=probs, minlength=len(successes))

    stale_mass, fresh_mass = masses(nu), masses(pi)
    if not all(math.isclose(m.sum(), 1.0, abs_tol=1e-12) for m in (stale_mass, fresh_mass)):
        raise ArithmeticError("multinomial probability mass does not sum to one")
    scores = (np.eye(4) - pi)[SUCCESS]
    weights = (pi / nu)[SUCCESS]
    log_weights = np.log(weights)
    updates = {method: np.zeros((len(successes), 4)) for method in TOY_METHODS}
    for idx, counts in enumerate(successes):
        k = int(counts.sum())
        if not k:
            continue
        labels = np.repeat([0, 1], counts)
        log_w = log_weights[labels]
        rewards = np.ones(k, dtype=bool)
        for method in METHODS:
            lam = coefficients(method, log_w, rewards)
            updates[method][idx] = (lam / k) @ scores[labels]
        updates["exact"][idx] = exact_update(counts, weights, scores)

    target = fresh_target(pi, n)
    result = {}
    for method, values in updates.items():
        mass = fresh_mass if method == "fresh" else stale_mass
        mean = mass @ values
        variance = mass @ np.sum((values - mean)**2, axis=1)
        result[method] = Moments(mean, target, float(variance))
    return result


def batch_moments(first, second, shared):
    """Average independent prompt updates. Prompt 2 uses swapped logits.

    With separate parameters, the score has two disjoint blocks. With shared
    parameters, the prompt-2 score must be pulled back into theta coordinates.
    """
    def combine(a, b):
        return (a + b[SWAP]) / 2 if shared else np.concatenate((a, b)) / 2

    result = {}
    for method, x in first.items():
        y = second[method]
        result[method] = Moments(combine(x.mean, y.mean), combine(x.target, y.target),
                                 (x.variance + y.variance) / 4)
    return result


def check_examples():
    """Assert independent analytic identities and documented numerical examples."""
    original = enumerate_moments(2, policy(.05, .5), policy(.20, .5))
    conditional_score = np.array([.4, -.4, -.4, .4])
    for method, scale in {"fresh": .36, "exact": .36, "uncorrected": .0975,
                          "snis": .0975, "count": .39}.items():
        np.testing.assert_allclose(original[method].mean, scale * conditional_score, atol=1e-14)
    clipping = enumerate_moments(8, policy(.20, .20), policy(.20, .80))
    for method, expected in {"count": 1.519272, "snis": .761393, "loo": 7.973237,
                             "count_cap": .499363, "snis_cap": .755848,
                             "loo_cap": .684231}.items():
        np.testing.assert_allclose(clipping[method].mse, expected, rtol=0, atol=5.1e-7)
    np.testing.assert_allclose(clipping["exact"].mean, clipping["fresh"].target, atol=1e-14)
    return {"original": original, "clipping": clipping}


def _row(case, n, geometry, method, result):
    return dict(case=case, n=n, geometry=geometry, method=method,
                squared_bias=result.squared_bias, variance=result.variance, mse=result.mse,
                cosine=result.cosine, mean=json.dumps(result.mean.tolist()),
                target=json.dumps(result.target.tolist()))


def run_grid():
    rows = []
    rates = {"equal": .20, "small": .15, "large": .05}
    compositions = {"equal": (.5, .5), "small": (.45, .55), "large": (.20, .80)}
    for (rate, p_nu), (composition, (s_nu, s_theta)), n in product(
        rates.items(), compositions.items(), [2, 8, 32]
    ):
        case = f"rate_{rate}__composition_{composition}"
        nu, pi = policy(p_nu, s_nu), policy(.20, s_theta)
        first = enumerate_moments(n, nu, pi)
        panels = {"one_prompt": first}
        if rate == "large" and composition in ("equal", "large"):
            second = enumerate_moments(n, nu[SWAP], pi[SWAP])
            for shared in (False, True):
                geometry = "shared" if shared else "separate"
                panels[geometry] = batch_moments(first, second, shared)
        for geometry, moments in panels.items():
            for method, result in moments.items():
                row = _row(case, n, geometry, method, result)
                row.update(p_nu=p_nu, p_theta=.20, s_nu=s_nu, s_theta=s_theta)
                rows.append(row)
    return rows


def write_results(output, rows):
    from .toy_plots import plot_batch_direction, plot_moments

    output.mkdir(parents=True, exist_ok=True)
    with (output / "moments.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_moments(rows, output / "bias-variance.png")
    plot_batch_direction(rows, output / "batch-direction.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/toy"))
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    for case, results in check_examples().items():
        print(f"\n{case:10} {'squared bias':>14} {'variance':>14} {'MSE':>14}")
        for method, result in results.items():
            print(f"{method:10} {result.squared_bias:14.6g} {result.variance:14.6g} {result.mse:14.6g}")
    if not args.checks_only:
        rows = run_grid()
        write_results(args.output, rows)
        print(f"\nWrote {len(rows)} rows and two figures to {args.output}")


if __name__ == "__main__":
    main()
