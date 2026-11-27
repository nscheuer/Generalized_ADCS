"""Tests for Passthrough feedforward."""

import numpy as np

from GenADCS.backends.python.hardware.ADCS_subsystem.controllers.feedforward import (
    Passthrough,
    passthrough_step,
)


def test_passthrough_forwards_goal_and_handles_nan() -> None:
    ff = Passthrough()
    assert ff.to_params().shape == (1,)

    q_cur = np.array([1.0, 0.0, 0.0, 0.0])
    q_g = np.array([0.0, 1.0, 0.0, 0.0])
    w_g = np.array([0.01, -0.02, 0.03])
    qd, wd, wdd, fft = passthrough_step(q_g, w_g, q_cur)
    assert np.allclose(qd, q_g)
    assert np.allclose(wd, w_g)
    assert np.allclose(wdd, 0.0)
    assert np.allclose(fft, 0.0)

    nan_q = np.full(4, np.nan)
    nan_w = np.full(3, np.nan)
    qd2, wd2, _, _ = passthrough_step(nan_q, nan_w, q_cur)
    assert np.allclose(qd2, q_cur)
    assert np.allclose(wd2, 0.0)
