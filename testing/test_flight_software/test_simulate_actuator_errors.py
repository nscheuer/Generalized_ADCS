"""simulate() draws the plant's actuator noise once per step (zero-order hold).

Before this change five consecutive steps with ``MTQ`` ``std_noise=1e-3``
applied exactly zero noise: the samples were never drawn, so ``std_noise``
on an actuator changed what the filters believed and never what the plant did.
"""

from __future__ import annotations

import numpy as np

import ADCS
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite, Satellite
from ADCS.state import State


def _run(std_noise: float, seed: int):
    np.random.seed(seed)
    satellite = Satellite(
        J_0=np.diag([0.5, 0.8, 1.2]),
        actuators=[MTQ(axis, max_torque=1.0, noise=Noise(noise=0.0, std_noise=std_noise)) for axis in np.eye(3)],
    )
    est_satellite = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), actuators=[MTQ(axis, max_torque=1.0) for axis in np.eye(3)])
    orbital_state = Orbital_State(ephem=Ephemeris(), J2000=0.22, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]))
    result = ADCS.simulate(
        x=State(w=np.array([0.01, -0.02, 0.015]), q=[1.0, 0.0, 0.0, 0.0]), satellite=satellite,
        est_satellite=est_satellite, os0=orbital_state, dt=10.0, tf=30.0,
    ).first()
    return np.asarray(result.state_hist[-1].w, dtype=float).copy(), satellite


def test_simulate_plant_draws_actuator_noise_every_step():
    noiseless, _ = _run(0.0, 1)
    noisy, satellite = _run(0.05, 1)
    assert not np.allclose(noisy, noiseless), "actuator noise never reached the plant"
    assert all(float(np.max(np.abs(actuator.noise.get_noise()))) != 0.0 for actuator in satellite.actuators)


def test_simulate_stays_reproducible_under_a_seed():
    first, _ = _run(0.05, 2)
    second, _ = _run(0.05, 2)
    np.testing.assert_allclose(first, second)
