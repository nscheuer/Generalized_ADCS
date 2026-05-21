"""
Regression tests for sensor validity and stochastic-update contracts.

This module checks three behaviors that are easy to regress because they sit at
the boundary between measurement models and estimator assumptions:

1. Sun-sensor eclipse handling. For both ``SunSensor`` and ``SunPair``, the
   clean measurement and its base-state Jacobian must agree on the eclipse
   sentinel: both should return ``NaN`` when ``Orbital_State.is_sunlit()`` is
   false, and both should stay finite when sunlit.

2. First-sample noise injection. ``Sensor.reading()`` should evolve stochastic
   noise before using it, so the very first ``MTM`` reading already reflects
   the configured noise standard deviation. The test estimates that spread over
   many random seeds and compares it to the configured ``Noise(std_noise=...)``.

3. Bias diffusion over elapsed time. With a random-walk ``Bias`` enabled and
   ``J2000`` advanced between samples, repeated ``MTM`` readings should drift
   rather than repeat, confirming that bias updates are applied on the current
   step instead of lagging by one measurement.
"""

import numpy as np
import pytest

from ADCS.satellite_hardware.sensors import SunSensor
from ADCS.satellite_hardware.sensors.sunpair import SunPair
from ADCS.satellite_hardware.sensors.magnetometer import MTM
from ADCS.satellite_hardware.errors import Noise, Bias, ErrorMode
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_helpers import normalize

EPHEM = Ephemeris()


def _os(sunlit):
    o = Orbital_State(ephem=EPHEM, J2000=0.22,
                      R=np.array([7000.0, 0.0, 0.0]),
                      V=np.array([0.0, 7.5, 0.0]),
                      B=np.array([2e-5, -1e-5, 3e-5]),
                      S=np.array([1.5e8, 0.0, 0.0]))
    o._sunlit = bool(sunlit)        # Orbital_State honours an explicit _sunlit
    return o


X = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("make", [
    lambda: SunSensor(axis=normalize(np.array([1.0, 0.2, -0.3])), efficiency=1.0),
    lambda: SunPair(axis=normalize(np.array([1.0, 0.2, -0.3])), efficiency=(1.0, 1.0)),
])
def test_sun_sensor_eclipse_contract_is_consistent(make):
    s = make()

    sun = _os(sunlit=True)
    assert np.isfinite(s.clean_reading(X, sun)), "sunlit reading must be finite"
    assert np.isfinite(s.basestate_jac(X, sun)).all(), "sunlit Jacobian finite"

    dark = _os(sunlit=False)
    assert not dark.is_sunlit()
    r = s.clean_reading(X, dark)
    j = s.basestate_jac(X, dark)
    # The sentinel must be NaN for BOTH the measurement and its Jacobian, so
    # the estimator's np.isnan deactivation is internally consistent (the old
    # finite-zeros Jacobian silently meant "measurement is exactly 0").
    assert np.isnan(np.asarray(r)).all(), f"eclipse reading must be NaN, got {r}"
    assert np.isnan(np.asarray(j)).all(), f"eclipse Jacobian must be NaN, got {j}"
    assert np.asarray(j).shape == (7, 1)


def test_sensor_first_reading_contains_noise():
    """The first reading must already carry noise of the configured std (the
    one-step-lag bug made it exactly noise-free)."""
    dmode = ErrorMode(add_bias=False, add_noise=True,
                      update_bias=False, update_noise=True)
    os_ = _os(sunlit=True)
    std = 3.0e-6

    first = []
    for seed in range(400):
        np.random.seed(seed)
        m = MTM(axis=np.array([1.0, 0.0, 0.0]),
                noise=Noise(std_noise=std))
        clean = m.clean_reading(X, os_)
        r = m.reading(X, os_, dmode=dmode)
        first.append(float(np.ravel(r - clean)[0]))

    first = np.array(first)
    # Bug: every first sample was exactly 0.0 -> std == 0.
    assert np.std(first) > 0.3 * std, \
        f"first-reading noise std {np.std(first):.2e} ~ 0 (configured {std:.2e})"
    assert np.isclose(np.std(first), std, rtol=0.25), \
        f"first-reading noise std {np.std(first):.2e} != configured {std:.2e}"


def test_sensor_bias_diffuses_as_time_advances():
    """With a random-walk bias and advancing time, successive readings must
    diverge (the random walk is driven by elapsed dt; the ordering fix must
    not suppress it)."""
    dmode = ErrorMode(add_bias=True, add_noise=False,
                      update_bias=True, update_noise=False)
    os_ = _os(sunlit=True)
    np.random.seed(7)
    m = MTM(axis=np.array([1.0, 0.0, 0.0]),
            bias=Bias(bias=0.0, std_bias=1.0e-3))
    vals = []
    for _ in range(5):
        vals.append(float(np.ravel(m.reading(X, os_, dmode=dmode))[0]))
        os_.J2000 += 1.0e-7          # advance ~0.3 s of centuries per step
    vals = np.array(vals)
    assert np.std(vals) > 0.0, "bias did not diffuse as time advanced"
    assert len(np.unique(vals)) == len(vals), "readings not all distinct"
