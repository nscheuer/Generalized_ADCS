"""
Numba @njit(cache=True) memory-layout invariance guard (test-hardening #12).

math_helpers.py has ~30 @njit(cache=True) kernels. The estimators/dynamics
routinely pass NON-CONTIGUOUS inputs to them -- e.g. the SR-/UAKF passes
strided slices like state1[3:6] or quaternion sub-vectors -- which is what
triggers the `NumbaPerformanceWarning: np.dot() is faster on contiguous
arrays`. That warning prompted backlog #12: do the cached kernels make
latent contiguity/dtype assumptions that produce WRONG results, and is the
on-disk cache a correctness hazard?

PROBED -- NEGATIVE RESULT (recorded so #12 is not re-investigated as a bug):
- Non-contiguous inputs: every kernel returns BIT-IDENTICAL results to the
  contiguous case (numba specialises on the array *type*, layout included,
  and computes correctly on strided arrays -- the warning is PURELY a
  performance advisory, not a correctness issue).
- Unsupported dtype (float32) into the f64 kernels raises a numba
  TypingError -- it fails LOUD, not silently wrong; and float32 is not a
  real code path (state/quaternion/covariance are float64 everywhere).
- cache=True is keyed on a content hash of the function, so a source change
  (or a different worktree's copy) invalidates it -- stale binaries are not
  silently reused; the documented cross-worktree concern is operational
  (recompiles / iCloud paths), not a wrong-result hazard.

So there is no correctness defect. This file LOCKS the verified property:
the JIT math kernels must be layout-invariant. The contiguous result is the
independent reference (not the kernel compared to itself). It is GREEN on
main and goes RED if a future change introduces a layout-dependent bug
(manual stride math, a `.ravel()`/`reshape` assumption, etc.).
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
