"""Tests for the statistics helpers (synthetic data)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.stats import bootstrap_ci, holm, paired_bootstrap_ci, paired_permutation_test  # noqa: E402


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    v = np.random.default_rng(0).uniform(size=200)
    lo, hi = bootstrap_ci(v)
    assert lo < v.mean() < hi
    assert (lo, hi) == bootstrap_ci(v)


def test_identical_systems_give_p_one_and_zero_ci():
    a = np.random.default_rng(1).uniform(size=100)
    assert paired_permutation_test(a, a) == pytest.approx(1.0)
    assert paired_bootstrap_ci(a, a) == (0.0, 0.0)


def test_clear_difference_is_significant():
    rng = np.random.default_rng(2)
    b = rng.uniform(size=200)
    a = b + 0.2
    assert paired_permutation_test(a, b) < 0.001
    lo, hi = paired_bootstrap_ci(a, b)
    assert lo > 0


def test_holm_adjustment():
    adj = holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert adj["x"] == pytest.approx(0.03)   # 3 * 0.01
    assert adj["z"] == pytest.approx(0.06)   # 2 * 0.03
    assert adj["y"] == pytest.approx(0.06)   # max(1 * 0.04, previous 0.06)
