"""Tests for MRPFeedbackLinearized."""

import numpy as np
import pytest

from GenADCS.backends.python.hardware.ADCS_subsystem.controllers.feedback import (
    MRPFeedbackLinearized,
    mrp_feedback_linearized_step,
)


def test_mrp_feedback_linearized_gain_validation() -> None:
    fb = MRPFeedbackLinearized(kp=0.5, kd=(1.0, 1.2, 1.4))
    kp, kd = fb.to_matrices()
    assert kp.shape == (3, 3) and kd.shape == (3, 3)
    with pytest.raises(ValueError):
        MRPFeedbackLinearized(kp=0.0, kd=1.0)
    with pytest.raises(ValueError):
        MRPFeedbackLinearized(kp=0.5, kd=-0.1)


def test_mrp_feedback_linearized_step_produces_restoring_torque() -> None:
    q = np.array([np.cos(np.pi / 6.0), np.sin(np.pi / 6.0), 0.0, 0.0])
    w = np.zeros(3)
    qd = np.array([1.0, 0.0, 0.0, 0.0])
    wd = np.zeros(3)
    inertia = np.diag((0.15, 0.12, 0.08))
    tau = mrp_feedback_linearized_step(
        q, w, qd, wd, wd, wd, np.zeros(3), inertia, inertia, np.eye(3), 0.5 * np.eye(3), 1.4 * np.eye(3)
    )
    assert tau[0] > 0.0
    assert np.allclose(tau[1:], 0.0, atol=1e-12)
