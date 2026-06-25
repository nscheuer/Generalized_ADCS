r"""
Phase 1 guard: the refactored ``simulate()`` (which now runs its per-step loop
through ``SatelliteAgent``) is deterministic when the satellite is given a
seeded RNG, and two different seeds produce independent stochastic histories.
This is the reproducibility foundation the formation orchestrator relies on.
"""

import numpy as np

from ADCS.CONOPS.goals import No_Goal
from ADCS.helpers.math_constants import MathConstants
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM, Gyro
from ADCS.simulate import simulate

UV = MathConstants.unitvecs
EPHEM = Ephemeris()


def _build_sat():
    acts = [MTQ(axis=UV[i], max_torque=0.1) for i in range(3)]
    sensors = [
        MTM(axis=UV[i], noise=Noise(noise=0.0, std_noise=1e-7), bias=Bias(bias=0.0, std_bias=1e-9))
        for i in range(3)
    ] + [
        Gyro(axis=UV[i], noise=Noise(noise=0.0, std_noise=1e-4), bias=Bias(bias=0.0, std_bias=1e-6))
        for i in range(3)
    ]
    return Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]), actuators=acts, sensors=sensors)


def _os0():
    return Orbital_State(
        ephem=EPHEM, J2000=0.22,
        R=-7000.0 * np.array([0.0, np.sqrt(0.5), np.sqrt(0.5)]),
        V=np.array([8.0, 0.0, 0.0]),
        B=np.array([0.0, 0.1, 0.0]),
        S=np.array([1e5 + 1.0, 0.0, 0.0]),
        rho=5e-12,
    )


def _run(seed):
    sat = _build_sat()
    sat.seed(seed)
    x0 = np.concatenate([np.array([0.01, -0.02, 0.015]), [1.0, 0.0, 0.0, 0.0]])
    res = simulate(x=x0, satellite=sat, goal=No_Goal(), os0=_os0(), dt=1.0, tf=20.0)[0]
    return np.asarray(res.sensor_hist, dtype=float), np.asarray(res.state_hist, dtype=float)


def test_simulate_is_bit_reproducible_under_same_seed():
    s_a, x_a = _run(7)
    s_b, x_b = _run(7)
    assert np.array_equal(s_a, s_b)
    assert np.array_equal(x_a, x_b)


def test_simulate_stochastic_history_differs_across_seeds():
    s_a, _ = _run(7)
    s_c, _ = _run(8)
    assert not np.allclose(s_a, s_c)


def test_simulate_returns_one_run_with_expected_history_length():
    s_a, x_a = _run(7)
    assert x_a.shape[0] == 20  # tf/dt steps
    assert np.all(np.isfinite(x_a))
    assert np.all(np.isfinite(s_a))
