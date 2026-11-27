"""Tests for L2Allocator."""

import numpy as np

from GenADCS.backends.python.hardware.ADCS_subsystem.controllers.allocator import (
    L2Allocator,
    allocate_l2,
)


def test_l2_allocator_solves_minimum_norm_and_saturates() -> None:
    s3 = 1.0 / np.sqrt(3.0)
    a_mat = np.array(
        [[1.0, 0.0, 0.0, s3], [0.0, 1.0, 0.0, s3], [0.0, 0.0, 1.0, s3]],
        dtype=np.float64,
    )
    a_pinv = np.linalg.pinv(a_mat)
    null_proj = np.eye(4) - a_pinv @ a_mat
    tau_c = np.array([0.3, -0.1, 0.2])
    u = allocate_l2(tau_c, a_pinv, null_proj, np.zeros(4), np.full(4, 1.0))
    assert L2Allocator().kind == 0
    assert np.allclose(a_mat @ u, tau_c, atol=1e-12)

    # Saturation check
    u_sat = allocate_l2(np.array([2.0, 0.0, 0.0]), a_pinv, null_proj, np.zeros(4), np.full(4, 0.2))
    assert np.all(np.abs(u_sat) <= 0.2 + 1e-12)
