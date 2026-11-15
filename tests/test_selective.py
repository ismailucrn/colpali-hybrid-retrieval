import numpy as np
import pytest

from src.selective import gated, relative_margin


def test_relative_margin():
    s = np.array([[10.0, 5.0, 1.0], [4.0, 4.0, 0.0], [0.0, 0.0, 0.0], [2.0, 8.0, 6.0]])
    np.testing.assert_allclose(relative_margin(s), [0.5, 0.0, 0.0, 0.25])


def test_gated_selects_rows():
    fused = np.array([[1.0, 2.0], [3.0, 4.0]])
    base = np.array([[9.0, 8.0], [7.0, 6.0]])
    out = gated(fused, base, np.array([0.9, 0.1]), tau=0.5)
    np.testing.assert_array_equal(out, [[1.0, 2.0], [7.0, 6.0]])


def test_gated_inf_keeps_base():
    fused, base = np.ones((3, 4)), np.zeros((3, 4))
    np.testing.assert_array_equal(gated(fused, base, np.array([1.0, 0.5, 0.0]), tau=np.inf), base)


def test_gated_shape_check():
    with pytest.raises(ValueError):
        gated(np.ones((2, 3)), np.ones((2, 3)), np.ones(3), 0.5)
