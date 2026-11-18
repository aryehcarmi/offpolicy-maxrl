import math

import numpy as np
import pytest

from maxrl.toy import (SWAP, batch_moments, check_examples, count_catalog, enumerate_moments,
                       exact_update, fresh_target, policy)


def test_frozen_examples():
    cases = check_examples()
    # The common capped numerator changes the SNIS denominator and its MSE.
    assert cases["clipping"]["snis_cap"].mse < cases["clipping"]["snis"].mse
    assert cases["original"]["exact"].variance > cases["original"]["count"].variance


@pytest.mark.parametrize("n", [2, 8, 32])
def test_multinomial_and_off_policy_mean_identities(n):
    nu, pi = policy(.05, .20), policy(.20, .80)
    counts, multiplicities = count_catalog(n)
    assert len(counts) == math.comb(n+3, 3)
    np.testing.assert_allclose(multiplicities @ np.prod(nu**counts, axis=1), 1.)
    moments = enumerate_moments(n, nu, pi)
    p, q = pi[[0,3]].sum(), nu[[0,3]].sum()
    conditional_score = np.array([pi[0]/p, 0, 0, pi[3]/p]) - pi
    np.testing.assert_allclose(moments["count"].mean,
                               (1 - (1-q)**n) * p/q * conditional_score, atol=1e-13)
    for method in ("fresh", "exact"):
        np.testing.assert_allclose(moments[method].mean, fresh_target(pi, n), atol=1e-12)


def test_exact_keeps_signed_terms_and_reduces_to_fresh_at_unit_weights():
    scores = (np.eye(4)-policy(.20, .50))[[0, 3]]
    # For two successes with weight 4, integral(1-4t) = -1.
    np.testing.assert_allclose(exact_update([1, 1], np.array([4, 4]), scores), -4*scores.sum(axis=0))
    np.testing.assert_allclose(exact_update([3, 5], np.ones(2), scores), np.array([3, 5]) @ scores / 8)
    pi = policy(.30, .70)
    moments = enumerate_moments(8, pi, pi)
    for stats in moments.values():
        np.testing.assert_allclose(stats.mean, fresh_target(pi, 8), atol=1e-13)
        np.testing.assert_allclose(stats.variance, moments["fresh"].variance)


def test_free_logit_fresh_target_and_shared_pullback_by_finite_difference():
    theta = np.log(policy(.20, .80))
    pi = np.exp(theta)
    def log_success(logits):
        weights = np.exp(logits)
        return np.log(weights[[0,3]].sum()/weights.sum())
    eps = 1e-5
    for permutation in [[0, 1, 2, 3], SWAP]:
        p = pi[permutation][[0,3]].sum()
        numerical = np.array([
            (log_success((theta+d)[permutation]) - log_success((theta-d)[permutation])) / (2*eps)
            for d in eps*np.eye(4)
        ])
        expected = (1 - (1-p)**8)*numerical
        np.testing.assert_allclose(fresh_target(pi[permutation], 8)[permutation], expected, atol=1e-10)


@pytest.mark.parametrize("shared", [False, True])
def test_batch_statistics_against_joint_count_distribution(shared):
    nu, pi = policy(.05, .5), policy(.20, .5)
    first = enumerate_moments(2, nu, pi)
    second = enumerate_moments(2, nu[SWAP], pi[SWAP])
    result = batch_moments(first, second, shared)["count"]
    counts, multiplicity = count_catalog(2)
    def prompt_law(nu, pi):
        mass = multiplicity * np.prod(nu**counts, axis=1)
        successes = counts[:, [0, 3]]
        weighted = successes * (pi/nu)[[0, 3]]
        scores = (np.eye(4)-pi)[[0, 3]]
        return mass, weighted @ scores / np.maximum(successes.sum(axis=1, keepdims=True), 1)
    mass1, g1 = prompt_law(nu, pi)
    mass2, g2 = prompt_law(nu[SWAP], pi[SWAP])
    means, mass = [], []
    for prob_a, a in zip(mass1, g1):
        for prob_b, b in zip(mass2, g2):
            means.append((a+b[SWAP])/2 if shared else np.r_[a,b]/2)
            mass.append(prob_a*prob_b)
    values, mass = np.array(means), np.array(mass)
    avg = mass @ values
    np.testing.assert_allclose(result.mean, avg)
    np.testing.assert_allclose(result.variance, mass @ np.sum((values-avg)**2, axis=1))
    np.testing.assert_allclose(result.mse, mass @ np.sum((values-result.target)**2, axis=1))
