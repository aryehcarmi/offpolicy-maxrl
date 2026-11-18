import numpy as np
import pytest

from maxrl.estimators import METHODS, coefficients


@pytest.mark.parametrize("method", METHODS)
def test_coefficients_against_direct_arithmetic(method):
    rewards = np.array([1, 0, 1, 1])
    weights = np.array([10., 1024., .5, 1.5])
    w = weights[rewards == 1]
    capped = np.minimum(w, 2.)
    other_mean = np.array([np.delete(w, i).mean() for i in range(len(w))])
    expected = {
        "fresh": np.ones(3), "uncorrected": np.ones(3), "count": w,
        "snis": w/w.mean(), "count_cap": capped, "snis_cap": capped/capped.mean(),
        "loo": w/other_mean, "loo_cap": np.minimum(w/other_mean, 2.),
        "hybrid": w/w.mean(),
    }[method]
    actual = coefficients(method, np.log(weights), rewards)
    np.testing.assert_allclose(actual[rewards == 1], expected)
    assert actual[1] == 0.
    np.testing.assert_array_equal(coefficients(method, np.log(weights), rewards*0), 0.)


@pytest.mark.parametrize("method", METHODS)
def test_single_success(method):
    expected = {"count": 4., "count_cap": 2., "loo": 4., "loo_cap": 2., "hybrid": 4.}
    np.testing.assert_allclose(coefficients(method, [np.log(4.), 1.], [1, 0]),
                               [expected.get(method, 1.), 0.])


def test_normalized_methods_and_final_cap_at_extreme_scales():
    log_w = np.log([1e250, 1e-50, 1e-50])
    rewards = [1, 1, 1]
    np.testing.assert_allclose(coefficients("loo", log_w, rewards),
                               [1e300, 2e-300, 2e-300], atol=0.)
    np.testing.assert_allclose(coefficients("loo_cap", log_w, rewards),
                               [2., 2e-300, 2e-300], atol=0.)
    np.testing.assert_allclose(coefficients("snis", [-900, -900], [1, 1]), 1.)
    np.testing.assert_allclose(coefficients("snis_cap", [900, 899], [1, 1]), 1.)
    with pytest.raises(FloatingPointError):
        coefficients("count", [900.], [1])


def test_count_snis_differ_by_one_scalar_within_prompt():
    rewards = np.array([1, 1, 0, 1])
    w = np.array([.12, 3., 2., .7])
    for raw, normalized, weights in [("count", "snis", w), ("count_cap", "snis_cap", np.minimum(w, 2))]:
        actual = coefficients(raw, np.log(w), rewards)
        reference = coefficients(normalized, np.log(w), rewards)
        np.testing.assert_allclose(actual, weights[rewards == 1].mean()*reference)
        np.testing.assert_allclose(reference.sum()/rewards.sum(), 1.)


@pytest.mark.parametrize(("method", "log_w", "rewards"), [
    ("exact", [0], [1]), ("count", [0], [.5]), ("snis", [np.nan], [1]),
    ("loo", [[0]], [[1]]), ("fresh", [0, 0], [1]),
])
def test_invalid_group(method, log_w, rewards):
    with pytest.raises(ValueError):
        coefficients(method, log_w, rewards)
