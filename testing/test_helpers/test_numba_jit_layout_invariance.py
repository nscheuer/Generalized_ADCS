"""
Regression test for Numba JIT helpers that may receive strided NumPy views.

It verifies that each kernel returns the same result for a non-contiguous
input and its contiguous copy, guarding against layout-dependent bugs.
"""

import warnings

import numpy as np
import pytest

from ADCS.helpers import math_helpers as H

_RNG = np.random.default_rng(1)


def _noncontig(n):
    """A strided, non-C-contiguous length-n view (like a slice of a larger
    state/covariance buffer)."""
    big = _RNG.standard_normal(4 * n)
    v = big[1:1 + 2 * n:2]
    assert v.shape == (n,) and not v.flags["C_CONTIGUOUS"]
    return v


def _cases():
    v3 = _noncontig(3)
    q = _noncontig(4)
    q /= np.linalg.norm(q)   # in-place: keep the strided (non-contiguous) view
    cases = [
        ("normalize", H.normalize, v3),
        ("norm", H.norm, v3),
        ("skewsym", H.skewsym, v3),
        ("rot_mat", H.rot_mat, q),
        ("quat_inv", H.quat_inv, q),
        ("quat_mult", lambda a: H.quat_mult(a, q), q),
        ("mrp_to_quat", H.mrp_to_quat, v3),
        ("cayley_to_quat", H.cayley_to_quat, v3),
    ]
    for m in (0, 1, 2, 6):
        cases.append((f"quat_to_vec3[mode={m}]",
                       lambda a, mm=m: H.quat_to_vec3(a, mm), q))
        cases.append((f"vec3_to_quat[mode={m}]",
                       lambda a, mm=m: H.vec3_to_quat(a, mm), v3))
    return cases


@pytest.mark.parametrize("name,fn,arg", _cases(), ids=lambda x: x if isinstance(x, str) else "")
def test_jit_kernel_is_layout_invariant(name, fn, arg):
    """Result on a non-contiguous (strided) input must equal the result on
    its contiguous copy -- the contiguous value is the independent reference."""
    contiguous = np.ascontiguousarray(arg)
    assert not arg.flags["C_CONTIGUOUS"] and contiguous.flags["C_CONTIGUOUS"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the NumbaPerformanceWarning is expected & benign
        r_contig = np.asarray(fn(contiguous), dtype=float)
        r_strided = np.asarray(fn(arg), dtype=float)
    np.testing.assert_allclose(
        r_strided, r_contig, atol=1e-10, rtol=1e-9,
        err_msg=f"{name}: non-contiguous result differs from contiguous "
                f"-> a layout-dependent bug was introduced")
