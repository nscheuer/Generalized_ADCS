"""Rank-one Cholesky updates on factors with structurally zero rows.

These arise whenever a covariance has an exactly-zero channel (a perfect
actuator's control covariance, a zero-process-noise block). The kernels used
to write NaN for every such row, which pushed each square-root covariance with
a zero channel onto the dense eigen-decomposition fallback.
"""

from __future__ import annotations

import numpy as np
import pytest

from ADCS.helpers.cholesky_update import choldowndate, cholupdate


def _gram(upper: np.ndarray) -> np.ndarray:
    return upper.T @ upper


@pytest.mark.parametrize(
    "diagonal, vector",
    [
        ([0.0, 1.0, 1.0], [0.0, 1.0, 0.0]),   # zero row untouched by the update
        ([0.0, 1.0, 1.0], [1.0, 1.0, 0.0]),   # zero row receives the update (swap rotation)
        ([1.0, 0.0, 2.0], [0.5, 0.0, 0.25]),  # zero row in the middle, untouched
        ([1.0, 0.0, 2.0], [0.5, 3.0, 0.25]),  # zero row in the middle, receives the update
    ],
)
def test_cholupdate_handles_structurally_zero_rows(diagonal, vector):
    upper = np.diag(np.asarray(diagonal, float))
    x = np.asarray(vector, float)
    expected = _gram(upper) + np.outer(x, x)
    updated = upper.copy()
    cholupdate(updated, x.copy())
    assert np.all(np.isfinite(updated)), updated
    np.testing.assert_allclose(_gram(updated), expected, rtol=0.0, atol=1.0e-14)
    assert np.allclose(updated, np.triu(updated))


@pytest.mark.parametrize(
    "diagonal, vector",
    [
        ([0.0, 1.0, 1.0], [0.0, 0.5, 0.0]),
        ([0.5, 0.0], [0.1, 0.0]),   # the control_covariance = diag([0.25, 0.0]) structure
    ],
)
def test_choldowndate_skips_untouched_zero_rows(diagonal, vector):
    upper = np.diag(np.asarray(diagonal, float))
    x = np.asarray(vector, float)
    expected = _gram(upper) - np.outer(x, x)
    updated = upper.copy()
    choldowndate(updated, x.copy())
    assert np.all(np.isfinite(updated)), updated
    np.testing.assert_allclose(_gram(updated), expected, rtol=0.0, atol=1.0e-14)


def test_choldowndate_still_flags_a_genuinely_indefinite_result():
    upper = np.diag([0.0, 1.0])
    updated = upper.copy()
    choldowndate(updated, np.array([0.3, 0.0]))  # removes mass from an empty channel
    assert np.isnan(updated).any()
