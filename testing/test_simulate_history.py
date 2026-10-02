"""Every entry of a simulation record belongs to the start of its step.

``simulate()`` and ``simulate_mc()`` used to record the real state *after*
propagating it, while the time, the orbital state, the readings, the estimate
and the control of the same record all came from the start of the step. So
``state_hist[k]`` was the state at ``time_s[k] + dt``, one step ahead of the
estimate it was plotted against.
"""

from __future__ import annotations

import numpy as np

import ADCS
from ADCS.helpers.math_helpers import normalize, quat_mult
from ADCS.mc.simulate_mc import simulate_mc
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite import EstimatedSatellite, Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.state import State

RATE = np.array([0.02, -0.01, 0.015])
START = normalize(np.array([0.9, 0.2, -0.3, 0.1]))
DT = 10.0


def _orbital_state() -> Orbital_State:
    return Orbital_State(ephem=Ephemeris(), J2000=0.22, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]))


def _truth_at(seconds: float) -> np.ndarray:
    # A spherical satellite with no torques spins at a constant rate about a fixed body axis.
    angle = np.linalg.norm(RATE) * seconds
    return quat_mult(START, np.concatenate([[np.cos(angle / 2.0)], np.sin(angle / 2.0) * RATE / np.linalg.norm(RATE)]))


def _satellite() -> Satellite:
    return Satellite(J_0=np.eye(3), sensors=[MTM(axis) for axis in np.eye(3)])


def _check_run(run) -> None:
    assert len(run.time_s) == len(run.state_hist) == len(run.os_hist) == len(run.clean_sensor_hist) == 5
    np.testing.assert_allclose(run.time_s, DT * np.arange(5))
    np.testing.assert_allclose(run.state_hist[0].q, START, atol=0.0)  # the first record is the initial state
    for seconds, state, os, clean in zip(run.time_s, run.state_hist, run.os_hist, run.clean_sensor_hist):
        np.testing.assert_allclose(state.q, _truth_at(seconds), atol=1.0e-8)
        np.testing.assert_allclose(state.w, RATE, atol=1.0e-12)
        # the recorded readings were taken from the recorded state in the recorded orbital state
        np.testing.assert_allclose(clean, _satellite().noiseless_sensor_readings(state, os), rtol=1.0e-12)


def test_simulate_records_the_state_of_the_same_instant_as_the_rest_of_the_record():
    run = ADCS.simulate(x=State(w=RATE, q=START), satellite=_satellite(), os0=_orbital_state(), dt=DT, tf=50.0).first()
    _check_run(run)


def test_simulate_mc_records_the_state_of_the_same_instant_as_the_rest_of_the_record():
    result = simulate_mc(
        x=State(w=RATE, q=START), satellite=_satellite(), os0=_orbital_state(), dt=DT, tf=50.0,
        num_runs=1, max_workers=1, base_seed=0,
    )
    _check_run(result.runs[0])


def test_estimate_time_and_orbital_state_share_the_step():
    class Stamping:
        """A stand-in estimator: its estimate encodes the step and it remembers the orbital state it saw."""

        def __init__(self):
            self.step_index = 0
            self.seen = []

        def update(self, u, sensors, os):
            self.seen.append(os)
            estimate = State(w=[float(self.step_index), 0.0, 0.0], q=[1.0, 0.0, 0.0, 0.0])
            self.step_index += 1
            return estimate

    stamping = Stamping()
    satellite = _satellite()
    run = ADCS.simulate(
        x=State(w=RATE, q=START), satellite=satellite,
        est_satellite=EstimatedSatellite(J_0=np.eye(3), sensors=[MTM(axis) for axis in np.eye(3)]),
        estimator=stamping, os0=_orbital_state(), dt=DT, tf=50.0,
    ).first()
    for k, (seconds, estimate, os, state) in enumerate(zip(run.time_s, run.est_state_hist, run.os_hist, run.state_hist)):
        assert estimate.w[0] == k
        assert seconds == k * DT
        assert os.J2000 == stamping.seen[k].J2000
        np.testing.assert_allclose(state.q, _truth_at(seconds), atol=1.0e-8)
