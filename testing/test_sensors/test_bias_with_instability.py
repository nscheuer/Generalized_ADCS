"""BiasWithInstability = the base Bias (offset + random walk) + a bias instability.

The instability is a first-order Gauss-Markov process with standard
deviation ``instability_std`` and correlation time ``instability_time``, the
Allan-variance floor. With it switched off the subclass draws exactly the
base class's random walk. The base ``Bias`` is untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.errors import Bias, BiasWithInstability
from ADCS.satellite_hardware.sensors import Gyro

T0 = 0.22
STEP_S = 1.0
STEP = STEP_S * TimeConstants.sec2cent


def _run(bias: Bias, steps: int) -> np.ndarray:
    bias._update_bias(T0)  # first call only records the time
    return np.array([(bias._update_bias(T0 + (k + 1) * STEP), bias.bias.copy())[1] for k in range(steps)])


def test_without_instability_the_sequence_is_the_base_random_walk():
    np.random.seed(5)
    ours = _run(BiasWithInstability(bias=0.01, std_bias=1.0e-4, instability_std=0.0, instability_time=10.0), 50)
    np.random.seed(5)
    base = _run(Bias(bias=0.01, std_bias=1.0e-4), 50)
    np.testing.assert_array_equal(ours, base)


def test_the_instability_is_stationary_with_the_configured_statistics():
    np.random.seed(9)
    sigma, tau = 1.0e-3, 50.0
    bias = BiasWithInstability(bias=0.0, std_bias=0.0, instability_std=sigma, instability_time=tau)
    history = _run(bias, 40000)[:, 0]
    assert np.std(history) == pytest.approx(sigma, rel=0.1)
    assert abs(np.mean(history)) < 3.0 * sigma / np.sqrt(40000 / (2 * tau))  # mean of a correlated series
    lag_one = np.corrcoef(history[:-1], history[1:])[0, 1]
    assert lag_one == pytest.approx(np.exp(-STEP_S / tau), abs=0.02)
    np.testing.assert_allclose(bias.bias, bias.instability)  # no offset, no walk: the bias is the instability


def test_an_estimator_write_back_is_kept():
    np.random.seed(3)
    bias = BiasWithInstability(bias=0.0, std_bias=0.0, instability_std=1.0e-3, instability_time=50.0)
    bias._update_bias(T0)
    bias._update_bias(T0 + STEP)
    bias.bias = np.array([0.5])  # what EstimatedSatellite.match_estimate does
    before = bias.instability.copy()
    bias._update_bias(T0 + 2 * STEP)
    np.testing.assert_allclose(bias.bias, 0.5 + (bias.instability - before))


def test_the_estimator_rate_covers_the_instability():
    bias = BiasWithInstability(bias=0.0, std_bias=2.0e-4, instability_std=1.0e-3, instability_time=50.0)
    np.testing.assert_allclose(bias.process_psd().as_matrix(), [[4.0e-8 + 2.0 * 1.0e-6 / 50.0]])
    np.testing.assert_allclose(bias.process_covariance(2.0).as_matrix(), [[2.0 * (4.0e-8 + 2.0 * 1.0e-6 / 50.0)]])


def test_the_psd_term_is_the_short_step_variance_rate_of_the_instability():
    # The Gauss-Markov step has variance sigma_g^2 (1 - exp(-2 dt / tau)). For dt << tau
    # that is (2 sigma_g^2 / tau) dt: a rate in bias^2 per second, the same units as
    # std_bias^2, which is what process_psd adds to the random walk.
    sigma_g, tau = 1.0e-3, 50.0
    bias = BiasWithInstability(bias=0.0, std_bias=0.0, instability_std=sigma_g, instability_time=tau)
    dt = tau / 1000.0
    step_variance = sigma_g**2 * (1.0 - np.exp(-2.0 * dt / tau))
    np.testing.assert_allclose(bias.process_psd().as_matrix(), [[step_variance / dt]], rtol=2.0e-3)
    # The alternative 2 sigma_g^2 tau is 2500x off here and has units of bias^2 * seconds.
    assert not np.isclose(bias.process_psd().as_matrix()[0, 0], 2.0 * sigma_g**2 * tau, rtol=0.5)


def test_the_psd_term_matches_the_simulated_instability_growth():
    # Draw many one-step instability increments from a known start and compare the sample
    # variance per second with the PSD term. Monte Carlo, seeded; 5 % tolerance.
    sigma_g, tau, dt_s = 1.0e-3, 50.0, 0.05
    rng_state = np.random.get_state()
    try:
        np.random.seed(7)
        draws = []
        for _ in range(20000):
            bias = BiasWithInstability(bias=0.0, std_bias=0.0, instability_std=sigma_g, instability_time=tau)
            bias._update_bias(T0)
            bias._update_bias(T0 + dt_s * TimeConstants.sec2cent)
            draws.append(bias.bias[0])
    finally:
        np.random.set_state(rng_state)
    rate_from_sim = np.var(draws) / dt_s
    np.testing.assert_allclose(rate_from_sim, bias.process_psd().as_matrix()[0, 0], rtol=0.05)


def test_configuration_is_validated_and_carried():
    with pytest.raises(ValueError, match="instability_time"):
        BiasWithInstability(instability_std=1.0e-3, instability_time=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        BiasWithInstability(instability_std=-1.0, instability_time=10.0)
    with pytest.raises(ValueError, match="shape"):
        BiasWithInstability(bias=np.zeros(3), instability_std=np.zeros(2), instability_time=10.0)
    assert bool(BiasWithInstability(instability_std=1.0e-3, instability_time=10.0))
    assert not bool(BiasWithInstability(instability_std=0.0, instability_time=10.0))
    sized = BiasWithInstability(instability_std=1.0e-3, instability_time=10.0).broadcast_to(3)
    assert isinstance(sized, BiasWithInstability)
    np.testing.assert_allclose(sized.instability_std, np.full(3, 1.0e-3))
    assert sized.instability_time == 10.0 and sized.copy().instability_time == 10.0
    gyro = Gyro(np.array([1.0, 0.0, 0.0]), bias=BiasWithInstability(bias=1.0e-4, instability_std=1.0e-5, instability_time=100.0))
    assert isinstance(gyro.bias, BiasWithInstability) and gyro.bias.instability_time == 100.0


def test_plain_bias_is_unchanged():
    assert not hasattr(Bias(std_bias=1.0e-4), "instability_std")
    np.testing.assert_allclose(Bias(std_bias=2.0e-4).process_psd().as_matrix(), [[4.0e-8]])
