"""
Finite-difference verification of the second-derivative (Hessian) helpers
in ADCS/helpers/math_helpers.py (backlog #5).

ddrotmatTvecdqdq, normed_vec_hess, vec_norm_hess feed dynamics_Hessians
(the one-step-MPC scaffolding) and planner second-order terms and had ZERO
test coverage. Their first-derivative counterparts (drotmatTvecdq,
normed_vec_jac, vec_norm_jac) were independently FD-verified earlier, so the
Hessian = d(Jacobian)/d(input) is the right ground truth. Each Hessian is
checked by central-differencing the verified Jacobian, plus index symmetry
and (for the norm) the exact closed form.
"""

import numpy as np
import pytest

from ADCS.helpers.math_helpers import (
    rot_mat, drotmatTvecdq, ddrotmatTvecdqdq,
    normed_vec_jac, normed_vec_hess,
    vec_norm_jac, vec_norm_hess, normalize,
)

EPS = 1e-6
RNG = np.random.default_rng(0)


def _uquat():
    q = RNG.normal(size=4)
    q = q / np.linalg.norm(q)
    return q * (1.0 if q[0] >= 0 else -1.0)


@pytest.mark.parametrize("trial", range(15))
def test_ddrotmatTvecdqdq_matches_fd_of_drotmatTvecdq(trial):
    q = _uquat()
    v = RNG.normal(size=3)
    H = np.asarray(ddrotmatTvecdqdq(q, v), float)        # (4,4,3) = d2(R^T v)/dq2
    assert H.shape == (4, 4, 3)
    fd = np.zeros((4, 4, 3))
    for i in range(4):
        dq = np.zeros(4); dq[i] = EPS
        Jp = np.asarray(drotmatTvecdq(q + dq, v), float)  # (4,3) = d(R^T v)/dq
        Jm = np.asarray(drotmatTvecdq(q - dq, v), float)
        fd[i] = (Jp - Jm) / (2.0 * EPS)
    err = np.max(np.abs(H - fd))
    assert err < 1e-5, f"ddrotmatTvecdqdq vs FD: max err {err:.2e}"
    assert np.allclose(H, np.transpose(H, (1, 0, 2)), atol=1e-9), "not q-symmetric"


@pytest.mark.parametrize("trial", range(15))
def test_normed_vec_hess_predicts_curvature(trial):
    """Strict, layout-pinned: the analytic Hessian must make the 2nd-order
    Taylor model of normalize(v) accurate to O(|delta|^3). Layout is
    [in, in, out] (same as ddrotmatTvecdqdq, verified strictly above);
    contracting it with delta(x)delta must reproduce the true curvature, so
    a wrong index convention or wrong values fail (no permissive min)."""
    v = RNG.normal(size=3) * RNG.uniform(0.5, 3.0)
    J = np.asarray(normed_vec_jac(v), float)
    H = np.asarray(normed_vec_hess(v), float)
    assert H.shape == (3, 3, 3)
    assert np.allclose(H, np.transpose(H, (1, 0, 2)), atol=1e-9), \
        "Hessian not symmetric in its two input indices"
    base = normalize(v)
    h = 1e-3
    max_rel = 0.0
    for _ in range(8):
        d = h * RNG.standard_normal(3)
        # normed_vec_jac layout is [in, out] here, so J.T @ d is the linear
        # term; H is [in, in, out].
        lin = J.T @ d
        quad = 0.5 * np.einsum("ijk,i,j->k", H, d, d)
        pred = base + lin + quad
        true = normalize(v + d)
        # 2nd-order model error must be cubic-small vs the quadratic term.
        resid = np.linalg.norm(true - pred)
        max_rel = max(max_rel, resid / (np.linalg.norm(quad) + 1e-30))
    assert max_rel < 0.05, f"2nd-order Taylor residual too large: {max_rel:.3e}"


@pytest.mark.parametrize("trial", range(15))
def test_vec_norm_hess_matches_fd_and_closed_form(trial):
    v = RNG.normal(size=3) * RNG.uniform(0.3, 4.0)
    H = np.asarray(vec_norm_hess(v), float)              # d2|v|/dv2  (3,3)
    assert H.shape == (3, 3)
    # Exact closed form: I/|v| - v v^T / |v|^3.
    n = np.linalg.norm(v)
    closed = np.eye(3) / n - np.outer(v, v) / n ** 3
    assert np.allclose(H, closed, atol=1e-9), f"vs closed form:\n{H}\n{closed}"
    # And central FD of the (verified) gradient vec_norm_jac = v/|v|.
    fd = np.zeros((3, 3))
    for i in range(3):
        dv = np.zeros(3); dv[i] = EPS
        gp = np.asarray(vec_norm_jac(v + dv), float).reshape(3)
        gm = np.asarray(vec_norm_jac(v - dv), float).reshape(3)
        fd[:, i] = (gp - gm) / (2.0 * EPS)
    assert np.allclose(H, fd, atol=1e-5), f"vs FD: max {np.abs(H-fd).max():.2e}"
    assert np.allclose(H, H.T, atol=1e-12), "Hessian not symmetric"
