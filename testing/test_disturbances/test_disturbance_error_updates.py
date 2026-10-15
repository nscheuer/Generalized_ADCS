"""Per-step disturbance error updates: the plant walks the parameters and draws the noise once per step.

Nothing in the simulation loop ever called a disturbance's ``update()``, and
``update()`` itself only re-applied the noise model's stored sample, which
nothing ever redrew. So a dipole or propulsion disturbance given a noise model
stayed at its nominal value for the whole run, and ``parameter_std_rate`` only
told the estimator to expect a random walk the true disturbance never made.
"""

from __future__ import annotations

import numpy as np

import ADCS
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.disturbances import Dipole_Disturbance, Prop_Disturbance, Torque_Disturbance
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite, Satellite
from ADCS.state import State

T0 = 0.22
STEP = 10.0 * TimeConstants.sec2cent


def _orbital_state(j2000: float = T0) -> Orbital_State:
    return Orbital_State(ephem=Ephemeris(), J2000=j2000, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]))


def test_noise_is_drawn_once_per_step_and_held_within_it():
    np.random.seed(3)
    nominal = np.array([1.0e-3, 0.0, 0.0])
    dipole = Dipole_Disturbance(nominal.copy(), noise=Noise(std_noise=1.0e-4))
    os = _orbital_state()
    assert np.array_equal(dipole.current_torque, nominal)  # untouched until the first step
    dipole.update_errors(T0)
    first = dipole.current_torque.copy()
    assert not np.array_equal(first, nominal)
    assert np.array_equal(dipole.torque(State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0]), os), np.cross(first, os.B))
    assert np.array_equal(dipole.current_torque, first)  # calling torque() did not draw again
    dipole.update_errors(T0 + STEP)
    assert not np.array_equal(dipole.current_torque, first)
    assert np.array_equal(dipole.torque_nominal, nominal)  # the nominal value is not what moves


def test_noise_is_independent_per_axis():
    np.random.seed(5)
    prop = Prop_Disturbance(np.zeros(3), noise=Noise(std_noise=1.0e-4))
    samples = []
    for k in range(200):
        prop.update_errors(T0 + k * STEP)
        samples.append(prop.torque(None, None).copy())
    samples = np.array(samples)
    assert np.all(np.std(samples, axis=0) > 5.0e-5)
    assert abs(np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]) < 0.3  # not the same scalar on every axis


def test_parameters_random_walk_at_the_configured_rate():
    # Many independent single steps: the increment's standard deviation is rate * sqrt(dt).
    np.random.seed(7)
    rate, dt = 1.0e-6, 10.0
    increments = []
    for _ in range(2000):
        torque = Torque_Disturbance(np.array([1.0e-5, 0.0, 0.0]), estimate_dist=True)
        torque.parameter_std_rate = np.full(3, rate)
        torque.update_errors(T0)  # first call only records the time
        assert np.array_equal(torque.main_param, [1.0e-5, 0.0, 0.0])
        torque.update_errors(T0 + dt * TimeConstants.sec2cent)
        increments.append(torque.main_param - np.array([1.0e-5, 0.0, 0.0]))
    increments = np.array(increments)
    np.testing.assert_allclose(np.std(increments, axis=0), rate * np.sqrt(dt), rtol=0.1)
    assert np.all(np.abs(np.mean(increments, axis=0)) < 3.0 * rate * np.sqrt(dt) / np.sqrt(2000))


def test_zero_rate_and_zero_noise_change_nothing_and_consume_no_randomness():
    state_before = np.random.get_state()
    torque = Torque_Disturbance(np.array([1.0e-5, 2.0e-5, 0.0]), estimate_dist=True)
    dipole = Dipole_Disturbance(np.array([1.0e-3, 0.0, 0.0]))
    for k in range(3):
        torque.update_errors(T0 + k * STEP)
        dipole.update_errors(T0 + k * STEP)
    assert np.array_equal(torque.main_param, [1.0e-5, 2.0e-5, 0.0])
    assert np.array_equal(dipole.current_torque, [1.0e-3, 0.0, 0.0])
    state_after = np.random.get_state()
    assert np.array_equal(state_before[1], state_after[1]) and state_before[2] == state_after[2]


def test_a_step_backwards_or_in_place_does_not_walk():
    np.random.seed(9)
    torque = Torque_Disturbance(np.zeros(3), estimate_dist=True)
    torque.parameter_std_rate = np.full(3, 1.0e-6)
    torque.update_errors(T0)
    torque.update_errors(T0)  # same time
    torque.update_errors(T0 - STEP)  # earlier time
    assert np.array_equal(torque.main_param, np.zeros(3))


def test_simulate_steps_the_true_disturbances_but_never_inside_the_solver():
    np.random.seed(11)
    satellite = Satellite(J_0=np.diag([0.5, 0.8, 1.2]), disturbances=[Dipole_Disturbance(np.array([1.0e-3, 0.0, 0.0]), noise=Noise(std_noise=1.0e-4))])
    estimated = EstimatedSatellite.from_satellite(satellite)
    dipole = satellite.disturbances[0]
    seen = []
    draws_inside = {"count": 0}
    inside = {"flag": False}
    original_rhs = Satellite.dynamics_for_solver
    original_normal = np.random.normal

    def rhs(self, *args, **kwargs):
        inside["flag"] = True
        try:
            seen.append(dipole.current_torque.copy())
            return original_rhs(self, *args, **kwargs)
        finally:
            inside["flag"] = False

    def normal(*args, **kwargs):
        if inside["flag"]:
            draws_inside["count"] += 1
        return original_normal(*args, **kwargs)

    Satellite.dynamics_for_solver = rhs
    np.random.normal = normal
    try:
        ADCS.simulate(x=State(w=[0.01, 0.0, 0.0], q=[1.0, 0.0, 0.0, 0.0]), satellite=satellite, est_satellite=estimated,
                      os0=_orbital_state(), dt=10.0, tf=50.0)
    finally:
        Satellite.dynamics_for_solver = original_rhs
        np.random.normal = original_normal
    assert draws_inside["count"] == 0
    distinct = {tuple(value) for value in seen}
    assert len(distinct) == 5  # one realisation per step, held across every solver evaluation of that step
    assert np.array_equal(estimated.disturbances[0].current_torque, [1.0e-3, 0.0, 0.0])  # the estimator's copy is a model, not the plant
