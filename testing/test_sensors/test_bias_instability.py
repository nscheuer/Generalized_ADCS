"""Bias = constant offset + random walk + bias instability.

The offset is the value the device was built with and never changes; the
random walk (rate ``std_bias``) starts at zero; the bias instability is a
first-order Gauss-Markov process with standard deviation ``instability_std``
and correlation time ``instability_time``, the Allan-variance floor.
"""

from __future__ import annotations

import numpy as np
import pytest

from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.errors import Bias
from ADCS.satellite_hardware.sensors import Gyro

T0 = 0.22
STEP_S = 1.0
STEP = STEP_S * TimeConstants.sec2cent


def _run(bias: Bias, steps: int) -> np.ndarray:
    bias._update_bias(T0)  # first call only records the time
    return np.array([(bias._update_bias(T0 + (k + 1) * STEP), bias.bias.copy())[1] for k in range(steps)])


def test_the_offset_is_constant_and_the_walk_starts_at_zero():
    np.random.seed(2)
    bias = Bias(bias=0.01, std_bias=1.0e-4)
    history = _run(bias, 500)
    np.testing.assert_allclose(bias.offset, [0.01])
    np.testing.assert_allclose(history[:, 0], 0.01 + np.cumsum(np.diff(np.concatenate([[0.01], history[:, 0]]))))
    assert abs(history[0, 0] - 0.01) < 5.0e-4  # the walk starts from the offset, not from somewhere else
    np.testing.assert_allclose(bias.bias, bias.offset + bias.walk + bias.instability)


def test_without_instability_the_sequence_is_the_old_random_walk():
    np.random.seed(5)
    bias = Bias(bias=0.01, std_bias=1.0e-4)
    history = _run(bias, 50)[:, 0]
    np.random.seed(5)
    reference, value = [], 0.01
    for _ in range(50):
        value = np.random.normal(loc=value, scale=1.0e-4 * np.sqrt(STEP_S))
        reference.append(value)
    # the same draws; the step length round-trips through Julian centuries, which resolves 1 s to about 1e-8 s
    np.testing.assert_allclose(history, reference, rtol=1.0e-6)


def test_the_instability_is_stationary_with_the_configured_statistics():
    np.random.seed(9)
    sigma, tau = 1.0e-3, 50.0
    bias = Bias(bias=0.0, std_bias=0.0, instability_std=sigma, instability_time=tau)
    history = _run(bias, 40000)[:, 0]
    assert np.std(history) == pytest.approx(sigma, rel=0.1)
    assert abs(np.mean(history)) < 3.0 * sigma / np.sqrt(40000 / (2 * tau))  # mean of a correlated series
    lag_one = np.corrcoef(history[:-1], history[1:])[0, 1]
    assert lag_one == pytest.approx(np.exp(-STEP_S / tau), abs=0.02)
    np.testing.assert_allclose(bias.offset, [0.0])
    np.testing.assert_allclose(bias.walk, [0.0])


def test_the_estimator_rate_covers_the_instability():
    bias = Bias(bias=0.0, std_bias=2.0e-4, instability_std=1.0e-3, instability_time=50.0)
    np.testing.assert_allclose(bias.process_psd().as_matrix(), [[4.0e-8 + 2.0 * 1.0e-6 / 50.0]])
    np.testing.assert_allclose(Bias(std_bias=2.0e-4).process_psd().as_matrix(), [[4.0e-8]])


def test_configuration_is_validated_and_carried():
    with pytest.raises(ValueError, match="instability_time"):
        Bias(instability_std=1.0e-3)
    with pytest.raises(ValueError, match="non-negative"):
        Bias(instability_std=-1.0, instability_time=10.0)
    with pytest.raises(ValueError, match="shape"):
        Bias(bias=np.zeros(3), instability_std=np.zeros(2), instability_time=10.0)
    assert bool(Bias(instability_std=1.0e-3, instability_time=10.0)) and not bool(Bias())
    sized = Bias(instability_std=1.0e-3, instability_time=10.0).broadcast_to(3)
    np.testing.assert_allclose(sized.instability_std, np.full(3, 1.0e-3))
    assert sized.instability_time == 10.0 and sized.copy().instability_time == 10.0
    gyro = Gyro(np.array([1.0, 0.0, 0.0]), bias=Bias(bias=1.0e-4, instability_std=1.0e-5, instability_time=100.0))
    assert gyro.bias.instability_time == 100.0
