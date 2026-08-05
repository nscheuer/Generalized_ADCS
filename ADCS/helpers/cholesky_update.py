r"""In-tree rank-1 Cholesky update and downdate.

Replaces the external ``choldate`` package, which has no PyPI release, must be
installed from git with ``--no-build-isolation``, and therefore made the whole
package impossible to ``pip install`` cleanly.

Convention (identical to ``choldate``): ``R`` is **upper triangular** with

.. math::

    A = R^\top R

``cholupdate(R, x)`` overwrites ``R`` in place with the factor of
:math:`A + x x^\top`; ``choldowndate(R, x)`` with the factor of
:math:`A - x x^\top`. Both use ``x`` as scratch space, so a copy is taken
internally and the caller's array is left untouched (``choldate`` clobbers it).

The algorithm is the standard LINPACK sequence of Givens rotations: one
rotation per row, each annihilating a component of ``x`` into the factor. It is
:math:`O(n^2)` and allocation-free.

**One deliberate difference from choldate.** On a downdate whose result is not
positive definite, ``choldate`` neither raises nor returns NaN -- it silently
computes :math:`\sqrt{|r^2|}` and returns a plausible-looking but wrong factor.
This implementation writes NaN instead, which is what the SRUAKF's existing
``np.any(np.isnan(...))`` guard was already written to catch.
"""

__all__ = ["cholupdate", "choldowndate"]

import numpy as np
from numba import njit


@njit(cache=True)
def _cholupdate_impl(R: np.ndarray, x: np.ndarray) -> None:
    n = R.shape[0]
    for k in range(n):
        Rkk = R[k, k]
        xk = x[k]
        r = np.sqrt(Rkk * Rkk + xk * xk)
        if Rkk == 0.0:
            # Degenerate factor: the rotation is undefined. Propagate NaN
            # rather than dividing by zero silently.
            for i in range(k, n):
                R[k, i] = np.nan
            return
        c = r / Rkk
        s = xk / Rkk
        R[k, k] = r
        for i in range(k + 1, n):
            R[k, i] = (R[k, i] + s * x[i]) / c
            x[i] = c * x[i] - s * R[k, i]


@njit(cache=True)
def _choldowndate_impl(R: np.ndarray, x: np.ndarray) -> None:
    n = R.shape[0]
    for k in range(n):
        Rkk = R[k, k]
        xk = x[k]
        r2 = Rkk * Rkk - xk * xk
        if Rkk == 0.0 or r2 <= 0.0:
            # A - x x^T is not positive definite; there is no real factor.
            # Fill with NaN so callers' finite-checks trip instead of
            # silently accepting a wrong factor.
            for i in range(k, n):
                for j in range(n):
                    R[i, j] = np.nan
            return
        r = np.sqrt(r2)
        c = r / Rkk
        s = xk / Rkk
        R[k, k] = r
        for i in range(k + 1, n):
            R[k, i] = (R[k, i] - s * x[i]) / c
            x[i] = c * x[i] - s * R[k, i]


def cholupdate(R: np.ndarray, x: np.ndarray) -> None:
    r"""Rank-1 update: overwrite ``R`` with the factor of :math:`A + x x^\top`.

    :param R: Upper-triangular Cholesky factor, modified **in place**.
    :type R: numpy.ndarray
    :param x: Update vector. Not modified (a working copy is taken).
    :type x: numpy.ndarray
    :return: None
    :rtype: None
    """
    _cholupdate_impl(R, np.ascontiguousarray(x, dtype=np.float64).copy())


def choldowndate(R: np.ndarray, x: np.ndarray) -> None:
    r"""Rank-1 downdate: overwrite ``R`` with the factor of
    :math:`A - x x^\top`.

    If the downdated matrix is not positive definite, ``R`` is filled with NaN.

    :param R: Upper-triangular Cholesky factor, modified **in place**.
    :type R: numpy.ndarray
    :param x: Downdate vector. Not modified (a working copy is taken).
    :type x: numpy.ndarray
    :return: None
    :rtype: None
    """
    _choldowndate_impl(R, np.ascontiguousarray(x, dtype=np.float64).copy())
