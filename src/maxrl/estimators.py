"""Success-only MaxRL coefficients for one prompt's group of responses."""

import numpy as np


STALE_METHODS = (
    "uncorrected", "count", "snis", "count_cap", "snis_cap", "loo", "loo_cap", "hybrid"
)
METHODS = ("fresh", *STALE_METHODS)
CAP = 2.0


def loo_log_ratio(log_w):
    """Log W_i minus log mean of the other successes (zero log mean for K=1)."""
    if len(log_w) == 1:
        return log_w
    # Prefix/suffix log sums avoid losing all the other weights to cancellation.
    prefix = np.r_[-np.inf, np.logaddexp.accumulate(log_w[:-1])]
    suffix = np.r_[np.logaddexp.accumulate(log_w[:0:-1])[::-1], -np.inf]
    return log_w - np.logaddexp(prefix, suffix) + np.log(len(log_w) - 1)


def coefficients(method, log_weights, rewards):
    """Return lambda_i (zero on failures). Divide the weighted score sum by K.

    Inputs are detached 1-D arrays. Log weights are sums of target minus actual
    sampler token log probabilities over the complete masked response. Finite
    log weights assume positive probability under both full-softmax policies.
    Unrepresentable uncapped coefficients raise FloatingPointError.
    """
    if method not in METHODS:
        raise ValueError(f"unknown estimator: {method}")
    log_weights = np.asarray(log_weights, dtype=np.float64)
    rewards = np.asarray(rewards)
    if log_weights.ndim != 1 or rewards.shape != log_weights.shape or not rewards.size:
        raise ValueError("expected nonempty log_weights and rewards vectors of equal size")
    if not np.all(np.isfinite(log_weights)):
        raise ValueError("a response has a non-finite log weight")
    if not np.all((rewards == 0) | (rewards == 1)):
        raise ValueError("rewards must be binary")

    selected = rewards.astype(bool)
    log_w = log_weights[selected]
    k = len(log_w)
    result = np.zeros_like(log_weights)
    if k == 0:
        return result
    if method in ("fresh", "uncorrected"):
        result[selected] = 1.0
        return result
    if method in ("count_cap", "snis_cap"):
        log_w = np.minimum(log_w, np.log(CAP))
    with np.errstate(over="raise", invalid="raise"):
        if method in ("snis", "snis_cap") or (method == "hybrid" and k > 1):
            relative = np.exp(log_w - log_w.max())
            result[selected] = k * relative / relative.sum()
        else:
            if method in ("loo", "loo_cap") and k > 1:
                log_w = loo_log_ratio(log_w)
            if method == "loo_cap":
                log_w = np.minimum(log_w, np.log(CAP))
            result[selected] = np.exp(log_w)
    return result
