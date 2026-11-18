"""Prompt-balanced grouped gradients and paired finite-pool diagnostics."""

import numpy as np

from .estimators import CAP, STALE_METHODS, coefficients, loo_log_ratio


def grouped_updates(rewards, log_weights, scores, n, prompt_ids, methods=STALE_METHODS):
    """Inputs are [prompt, draw, ...]. Batch b uses group b from every prompt."""
    prompts, draws = rewards.shape
    if len(prompt_ids) != prompts or draws % n or n < 1:
        raise ValueError("every prompt must supply complete groups")
    if log_weights.shape != rewards.shape or scores.shape[:2] != rewards.shape:
        raise ValueError("misaligned saved-response arrays")
    updates = {m: np.zeros((draws // n, scores.shape[-1])) for m in methods}
    exposure = []
    failures = {}
    for p, puzzle_id in enumerate(prompt_ids):
        for group in range(draws // n):
            sl = slice(group * n, (group + 1) * n)
            r, lw, s = rewards[p, sl], log_weights[p, sl], scores[p, sl]
            k = int(r.sum())
            row = {"puzzle_id": puzzle_id, "group": group, "K": k}
            if k:
                success_lw = lw[r.astype(bool)]
                relative = np.exp(success_lw - success_lw.max())
                log_d_over_k = float(np.logaddexp.reduce(success_lw) - np.log(k))
                with np.errstate(over="ignore"):
                    d_over_k = np.exp(log_d_over_k)
                row.update(
                    successful_ess=float(relative.sum() ** 2 / (relative @ relative)),
                    d_over_k=float(d_over_k) if np.isfinite(d_over_k) else None,
                    log_d_over_k=log_d_over_k,
                    log_inverse_success_mean=-log_d_over_k,
                    log_inverse_w_median=np.median(-success_lw),
                    log_inverse_w_max=np.max(-success_lw),
                    raw_cap_fraction=np.mean(success_lw > np.log(CAP)),
                    loo_final_cap_fraction=np.mean(loo_log_ratio(success_lw) > np.log(CAP)),
                )
            for method in methods:
                if method in failures:
                    continue
                try:
                    with np.errstate(over="raise", invalid="raise"):
                        scale = coefficients(method, lw, r) / max(1, k)
                        row["exposure_" + method] = scale.sum()
                        updates[method][group] += (scale / prompts) @ s
                        if not np.isfinite(updates[method][group]).all():
                            raise FloatingPointError("nonfinite prompt-balanced update")
                except FloatingPointError as error:
                    failures[method] = {"puzzle_id": puzzle_id, "group": group, "reason": str(error)}
                    del updates[method]
            exposure.append(row)
    return updates, exposure, failures


def _unit_and_norm(vectors):
    scale = np.max(np.abs(vectors), axis=1)
    scaled = vectors / np.where(scale > 0, scale, 1)[:, None]
    length = np.linalg.norm(scaled, axis=1)
    unit = scaled / np.where(length > 0, length, 1)[:, None]
    with np.errstate(over="ignore"):
        return unit, scale * length


def _cosines(a, b):
    unit_a, len_a = _unit_and_norm(a)
    unit_b, len_b = _unit_and_norm(b)
    values = np.einsum("ij,ij->i", unit_a, unit_b).clip(-1, 1)
    return np.where((len_a > 0) & (len_b > 0), values, np.nan)


def _resample_weights(rng, batches, repeats):
    # Row zero gives the original point estimate. Remaining rows represent
    # whole-batch bootstrap draws, paired across estimators on the same pool.
    weights = rng.multinomial(batches, np.full(batches, 1 / batches), size=repeats)
    return np.vstack((np.ones(batches), weights)) / batches


def _metrics(batches, snis_norm, reference, weights):
    units, lengths = _unit_and_norm(batches)
    mean = weights @ batches
    _, ref_norm = _unit_and_norm(reference)

    def mse(values, w):
        # Keep *intermediate* squares representable. Only the final change
        # of units can overflow, in which case record a failed MSE explicitly.
        scale = max(np.max(np.abs(values)), np.max(np.abs(reference))) or 1.0
        v, ref = values / scale, reference / scale
        v_sq = np.einsum("ij,ij->i", v, v)
        error = (w @ v_sq - 2 * np.einsum("ij,ij->i", w @ v, ref)
                 + np.einsum("ij,ij->i", ref, ref))
        with np.errstate(over="ignore"):
            return np.maximum(error, 0) * scale * scale

    valid = (lengths > 0) & (snis_norm > 0)
    mass = weights @ valid
    matched_weights = weights * valid
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        matched_weights /= mass[:, None]
        ratio = np.divide(lengths, snis_norm, out=np.zeros_like(lengths), where=valid)
        raw_ratio = (weights @ lengths) / ref_norm
    return {
        "cosine_of_mean": _cosines(mean, reference),
        "raw_batch_mse": mse(batches, weights),
        "mean_batch_norm_over_reference_norm": np.where(ref_norm > 0, raw_ratio, np.nan),
        "zero_batch_fraction": weights @ (lengths == 0),
        "matched_batch_fraction": mass,
        "norm_matched_mse": mse(units * snis_norm[:, None], matched_weights),
        "mean_batch_norm_over_snis_norm_on_matched": matched_weights @ ratio,
    }


def _interval(values, intervals=True):
    finite = values[1:][np.isfinite(values[1:])]
    # Undefined draws commonly arise for zero updates. Give their count and
    # compute a conditional interval only when both pools have replication.
    overflow = np.isinf(values)
    ci = None
    if intervals and len(finite) and not overflow.any():
        scale = max(np.abs(finite)) or 1
        ci = (np.quantile(finite / scale, [0.025, 0.975]) * scale).tolist()
    return {
        "point_overflow": bool(overflow[0]),
        "overflow_bootstrap_draws": int(overflow[1:].sum()) if intervals else 0,
        "estimate": float(values[0]) if np.isfinite(values[0]) else None,
        "ci95": ci,
        "defined_bootstrap_draws": int(len(finite)) if intervals else 0,
    }


def _split_reliability(batches, rng, repeats):
    a, b = np.array_split(batches, 2)
    if len(b) < 1:
        return {"estimate": None, "ci95": None, "defined_bootstrap_draws": 0}
    a_means = _resample_weights(rng, len(a), repeats) @ a
    b_means = _resample_weights(rng, len(b), repeats) @ b
    return _interval(_cosines(a_means, b_means), min(len(a), len(b)) >= 2)


def summarize(batches, fresh, repeats=500, seed=7001, failures=None):
    """Return point estimates, paired differences, and sampling intervals.

    Puzzle identities are fixed. Independently resample complete batches from
    the candidate and reference pools, sharing both draws across methods.
    """
    if repeats < 1 or not len(fresh) or not batches:
        raise ValueError("positive bootstrap count, reference groups and candidate batches are required")
    n = len(next(iter(batches.values())))
    if any(len(values) != n for values in batches.values()) or not n:
        raise ValueError("all candidate methods need the same batch count")
    rng = np.random.default_rng(seed)
    stale_w = _resample_weights(rng, n, repeats)
    fresh_w = _resample_weights(rng, len(fresh), repeats)
    refs = fresh_w @ fresh
    snis_norm = _unit_and_norm(batches["snis"])[1] if "snis" in batches else np.zeros(n)
    measures = {m: _metrics(g, snis_norm, refs, stale_w) for m, g in batches.items()}
    intervals = min(n, len(fresh)) >= 2
    ref_check = _split_reliability(fresh, rng, repeats)
    result = {"stale_batches": n, "reference_batches": len(fresh), "bootstrap_seed": seed,
              "bootstrap_repeats": repeats, "reference_split_half_cosine": ref_check,
              "methods": {}, "paired_differences": {}, "numerical_failures": failures or {}}

    def reliable(check):
        return (check["ci95"] is not None and check["ci95"][0] > 0 and
                check["defined_bootstrap_draws"] == repeats)

    for method, metrics in measures.items():
        split = _split_reliability(batches[method], rng, repeats)
        direction = "interpretable" if reliable(split) and reliable(ref_check) else "unresolved"
        result["methods"][method] = {
            **{k: _interval(v, intervals) for k, v in metrics.items()},
            "mean_split_half_cosine": split, "mean_direction_status": direction,
        }

    # Count and SNIS share a paired inclusion mask for matched errors. The
    # SNIS error on *all* its nonzero batches can use a different denominator.
    for method, baseline in (("snis", "count"), ("loo_cap", "count"), ("loo_cap", "snis"),
                             ("hybrid", "count"), ("hybrid", "snis")):
        if method not in measures or baseline not in measures:
            continue
        def difference(a, b):
            with np.errstate(over="ignore", invalid="ignore"):
                return np.where(np.isinf(a) | np.isinf(b), np.inf, a - b)

        diffs = {k: _interval(difference(measures[method][k], measures[baseline][k]), intervals)
                 for k in ("cosine_of_mean", "raw_batch_mse")}
        valid = ((snis_norm > 0) & (np.any(batches[method] != 0, axis=1))
                 & (np.any(batches[baseline] != 0, axis=1)))
        shared_norm = snis_norm * valid
        first = _metrics(batches[method], shared_norm, refs, stale_w)
        second = _metrics(batches[baseline], shared_norm, refs, stale_w)
        diffs["norm_matched_mse"] = _interval(difference(first["norm_matched_mse"],
                                                               second["norm_matched_mse"]), intervals)
        diffs["matched_batches"] = int(valid.sum())
        result["paired_differences"][f"{method}_minus_{baseline}"] = diffs
    return result
