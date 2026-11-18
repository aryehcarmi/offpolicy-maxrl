"""Full-model comparisons from streamed, prompt-balanced batch gradients."""

import tempfile
from pathlib import Path

import numpy as np

from .data import digest
from .diagnostics import _dump, _scored_manifest, load_rows, stage_resources, staged_directory, target_model
from .estimators import coefficients
from .gradient_metrics import summarize


FULL_METHODS = ("uncorrected", "count", "snis", "loo_cap", "hybrid")


def group_terms(rows, log_weights, rewards, n, group, method):
    """Yield response and detached coefficient for one prompt-balanced batch."""
    prompts, draws = rewards.shape
    if draws % n or not 0 <= group < draws // n:
        raise ValueError("invalid group index")
    for p in range(prompts):
        start = n * group
        sl = slice(start, start + n)
        r = rewards[p, sl]
        scale = coefficients(method, log_weights[p, sl], r) / (prompts * max(1, int(r.sum())))
        for j in np.flatnonzero(scale):
            yield rows[p * draws + start + j], float(scale[j])


def aggregate(model, terms, microbatch):
    """Accumulate response sums into one full-parameter ascent gradient."""
    import torch
    from .model import sequence_logprobs

    if microbatch < 1:
        raise ValueError("microbatch must be positive")
    model.requires_grad_(True)
    model.eval()
    model.zero_grad(set_to_none=True)
    terms = list(terms)
    for start in range(0, len(terms), microbatch):
        chunk = terms[start:start + microbatch]
        log_probs = sequence_logprobs(model, [row for row, _ in chunk])
        scale = log_probs.new_tensor([weight for _, weight in chunk])
        (log_probs * scale).sum().backward()


def store_gradient(path, model):
    """Save one aggregate vector. Missing gradients are mathematical zeros."""
    with path.open("wb") as f:
        for param in model.parameters():
            if param.grad is None:
                # Sparse-file hole in unallocated regions, including all-zero batches.
                f.seek(param.numel() * 4, 1)
            else:
                array = param.grad.detach().float().cpu().contiguous().numpy()
                if not np.all(np.isfinite(array)):
                    raise FloatingPointError("nonfinite full-model batch gradient")
                array.tofile(f)
        f.truncate(f.tell())
    model.zero_grad(set_to_none=True)


def gram_coordinates(paths, chunk_size=65536):
    """Compact exact inner-product geometry via a blockwise FP64 Gram matrix."""
    arrays = [np.memmap(path, mode="r", dtype=np.float32) for path in paths]
    gram = np.zeros((len(arrays), len(arrays)))
    for start in range(0, len(arrays[0]), chunk_size):
        part = np.array([a[start:start + chunk_size] for a in arrays], dtype=np.float64)
        gram += part @ part.T
    # Eigensolve a correlation matrix to retain capped methods' geometry when
    # a large, *representable* uncapped gradient dominates the unscaled Gram.
    norms = np.sqrt(gram.diagonal())
    denom = np.where(norms > 0, norms, 1.0)
    values, vectors = np.linalg.eigh(gram / denom[:, None] / denom[None, :])
    # Gram is positive semidefinite by construction. Clear roundoff on exact
    # zero rows so the zero-update rule remains exact after eigendecomposition.
    coordinates = norms[:, None] * vectors * np.sqrt(np.maximum(values, 0))
    coordinates[gram.diagonal() == 0] = 0
    return coordinates


def full(args):
    directory = Path(args.pools)
    manifest = _scored_manifest(directory)
    n = args.group_size
    num_batches = 1 if manifest["panel"] == "midpoint" else 8
    source = args.source
    if source not in manifest["pools"]:
        raise ValueError(f"{source} is unavailable in this panel")
    resources, hardware = stage_resources(args)
    model = target_model(args, manifest)
    output = directory / f"full-{source}-N{n}"
    paths, indices, slices, failures = [], {}, {}, {}
    with staged_directory(output) as dest, tempfile.TemporaryDirectory(dir=args.scratch) as temp:
        for name in (source, "reference"):
            rows = load_rows(directory, manifest, name)
            with np.load(directory / f"scores/{name}.npz") as saved:
                lw, r = saved["log_weights"], saved["rewards"]
            total = r.shape[1] // n
            size = num_batches * (2 if name == "reference" else 1)
            if size > total:
                raise ValueError("insufficient independent groups for this full-model check")
            # Grid is fixed before reading scores. Double the reference count
            # for split halves without conditioning selection on success.
            indices[name] = [(i * total) // size for i in range(size)]
            methods = ("fresh",) if name == "reference" else FULL_METHODS
            for method in methods:
                started = resources.clock()
                batch_paths = []
                try:
                    for b in indices[name]:
                        path = Path(temp) / f"{name}-{method}-{b}.f32"
                        aggregate(model, group_terms(rows, lw, r, n, b, method), args.microbatch)
                        store_gradient(path, model)
                        batch_paths.append(path)
                except FloatingPointError as error:
                    model.zero_grad(set_to_none=True)
                    failures[method] = {"pool": name, "group": b, "reason": str(error)}
                else:
                    slices[method] = slice(len(paths), len(paths) + len(batch_paths))
                    paths.extend(batch_paths)
                resources.add(f"{name}_{method}", started)
            del rows
        del model
        started = resources.clock()
        if "fresh" in slices and len(slices) > 1:
            coords = gram_coordinates(paths)
            resources.add("gram", started)
            batches = {m: coords[slices[m]] for m in FULL_METHODS if m in slices}
            result = summarize(batches, coords[slices["fresh"]], args.bootstrap, args.seed, failures)
        else:
            result = {"status": "no finite candidate/reference comparison",
                      "numerical_failures": failures}
        resources.save(dest / "resources.csv")
        _dump(dest / "summary.json", {"group_size": n, "source": source, "batch_indices": indices,
                                      "manifest_sha256": digest(directory / "manifest.json"),
                                      "metrics": result, "hardware": hardware})
