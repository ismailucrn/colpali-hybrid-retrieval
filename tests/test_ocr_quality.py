"""Tests for the OCR-quality helpers (CPU, synthetic values)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ocr_quality import average_ranks, page_proxy, spearman  # noqa: E402


def test_average_ranks_with_ties():
    assert average_ranks(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [3.5, 1.0, 3.5, 2.0]


def test_spearman_known_values():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, x * 10) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
    # ties: hand-computed Pearson correlation of average ranks [1, 2, 3, 4] vs [1.5, 1.5, 3.5, 3.5]
    assert spearman(x, np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(0.894427, abs=1e-6)


def test_page_proxy():
    vocab = {"total", "cost"}
    assert page_proxy("TOTAL cost x 12 qzx", vocab) == (pytest.approx(2 / 3), 3)  # "x" (len 1) and "12" ignored
    assert page_proxy("", vocab) == (None, 0)
