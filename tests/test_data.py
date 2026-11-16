"""Rows that share a page (TAT-DQA, TabFQuAD, SyntheticDocQA): official max-merge semantics."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import build_task, max_over_rows  # noqa: E402


def test_duplicate_pages_merge_to_first_occurrence_order():
    task = build_task(["a", None, "b", "a"], ["p1", "p0", "p1", "p2"], allow_duplicate_pages=True)
    assert task.page_ids == ["p1", "p0", "p2"]
    assert task.row_page == [0, 1, 0, 2]
    assert task.queries == ["a", "b"]
    assert task.qrels == {"a": {"p2": 1}, "b": {"p1": 1}}  # last occurrence wins, as in the official dict overwrite


def test_unique_pages_give_identity_row_map():
    task = build_task(["a", "b"], ["p0", "p1"])
    assert task.row_page == [0, 1]


def test_max_over_rows():
    rows = np.array([[1.0, 5.0, 2.0, 0.5], [3.0, 0.0, 4.0, 1.0]])
    out = max_over_rows(rows, [0, 1, 0, 2], 3)
    assert out.tolist() == [[2.0, 5.0, 0.5], [4.0, 0.0, 1.0]]


def test_max_over_rows_refuses_negative_scores():
    with pytest.raises(ValueError):
        max_over_rows(np.array([[-1.0, 2.0]]), [0, 1], 2)
