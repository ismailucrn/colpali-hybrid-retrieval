"""Uncertainty and paired tests over per-query metric values (queries are the resampling unit)."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(values: np.ndarray, n_resamples: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_resamples: int = 10_000, alpha: float = 0.05,
                        seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of mean(a - b), resampling queries jointly (paired)."""
    return bootstrap_ci(np.asarray(a) - np.asarray(b), n_resamples, alpha, seed)


def paired_permutation_test(a: np.ndarray, b: np.ndarray, n_permutations: int = 10_000, seed: int = 0) -> float:
    """Two-sided sign-flip randomisation test of mean(a - b) = 0. Returns a p-value with the +1 correction."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    observed = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((1 + np.sum(null >= observed - 1e-12)) / (n_permutations + 1))


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm–Bonferroni adjusted p-values (family-wise error control)."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m, running, adjusted = len(items), 0.0, {}
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted
