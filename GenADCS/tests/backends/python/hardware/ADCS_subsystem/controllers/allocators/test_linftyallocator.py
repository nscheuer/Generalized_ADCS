"""Tests for LInfinityAllocator."""

import numpy as np

from GenADCS.backends.python.hardware.ADCS_subsystem.controllers.allocator import (
    LInfinityAllocator,
    allocate_l2,
    allocate_linfinity,
)


def test_linfinity_allocator_minimizes_max_wheel_torque_on_4_wheels() -> None:
    s3 = 1.0 / np.sqrt(3.0)
    a_mat = np.array(
        [[1.0, 0.0, 0.0, s3], [0.0, 1.0, 0.0, s3], [0.0, 0.0, 1.0, s3]],
        dtype=np.float64,
    )
    a_pinv = np.linalg.pinv(a_mat)
    null_proj = np.eye(4) - a_pinv @ a_mat
    _, _, vh = np.linalg.svd(a_mat, full_matrices=True)
    null_basis = vh[3:].T
    max_t = np.full(4, 1.0)
    tau_c = np.array([0.3, 0.0, 0.0])

    u_l2 = allocate_l2(tau_c, a_pinv, null_proj, np.zeros(4), max_t)
    u_linf = allocate_linfinity(tau_c, a_pinv, null_basis, null_proj, np.zeros(4), max_t)
    assert LInfinityAllocator().kind == 1
    assert np.allclose(a_mat @ u_linf, tau_c, atol=1e-9)
    assert np.max(np.abs(u_linf)) < np.max(np.abs(u_l2))


def test_linfinity_allocator_general_5_wheel_nullspace() -> None:
    a_mat = np.array(
        [
            [1.0, 0.0, 0.0, 0.5, -0.5],
            [0.0, 1.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0, 0.70710678, 0.70710678],
        ],
        dtype=np.float64,
    )
    a_pinv = np.linalg.pinv(a_mat)
    null_proj = np.eye(5) - a_pinv @ a_mat
    _, _, vh = np.linalg.svd(a_mat, full_matrices=True)
    null_basis = vh[3:].T
    tau_c = np.array([0.4, 0.1, -0.1])
    max_t = np.full(5, 1.0)
    u_l2 = allocate_l2(tau_c, a_pinv, null_proj, np.zeros(5), max_t)
    u_linf = allocate_linfinity(tau_c, a_pinv, null_basis, null_proj, np.zeros(5), max_t)
    assert np.allclose(a_mat @ u_linf, tau_c, atol=1e-8)
    assert np.max(np.abs(u_linf)) <= np.max(np.abs(u_l2)) + 1e-9
