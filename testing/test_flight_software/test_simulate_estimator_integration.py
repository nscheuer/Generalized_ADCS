"""simulate() with the new-generation estimators, alone and with an orbit estimator.

Regressions from the September 2026 estimator audit: the loop handed the
orbit estimate's wrapper object to the attitude estimator and the goals
(``AttributeError: get_state_vector`` / ``R``), recorded no covariance history
for any new filter (it read an attribute only the deleted legacy class had),
and could not drive a one-call estimator such as the remote proxy.
"""

from __future__ import annotations

import numpy as np

import ADCS
from ADCS.estimators.attitude_estimators import EKF, MEKF
from ADCS.estimators.orbit_estimators import Orbit_GPS
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite, Satellite
from ADCS.satellite_hardware.sensors import GPS, Gyro, StarTrackerQuaternion
from ADCS.state import EstimatorState, State


def _tracker() -> StarTrackerQuaternion:
    tracker = StarTrackerQuaternion(noise=Noise(std_noise=np.full(4, 1.0e-3)))

    def clean(*args, **kwargs):
        state = kwargs.get("x", kwargs.get("state"))
        if state is None and args:
            state = args[0]
        return np.asarray(state.q, float).copy()

    tracker.clean_reading = clean
    return tracker


def _sensors():
    return ([Gyro(axis, noise=Noise(std_noise=1.0e-4)) for axis in np.eye(3)] + [_tracker()]
            + [GPS(noise=Noise(noise=np.zeros(6), std_noise=np.full(6, 1.0e-2)))])


def _orbital_state() -> Orbital_State:
    return Orbital_State(ephem=Ephemeris(), J2000=0.22, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]))


def _estimate(size: int) -> EstimatorState:
    return EstimatorState(w=[0.02, -0.015, 0.01], q=[0.9, 0.2, -0.3, 0.1], cov=np.eye(size) * 1.0e-3,
                          int_cov=np.zeros((size, size))).normalized()


TRUTH = State(w=np.array([0.02, -0.015, 0.01]), q=np.array([0.9, 0.2, -0.3, 0.1])).normalized()


def test_simulate_records_covariance_history_for_new_filters():
    result = ADCS.simulate(
        x=TRUTH, satellite=Satellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=_sensors()),
        est_satellite=EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=_sensors()),
        estimator=MEKF(EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=_sensors()), _estimate(6), dt=10.0),
        os0=_orbital_state(), dt=10.0, tf=40.0,
    ).first()
    assert result.state_cov_hist is not None and len(result.state_cov_hist) > 0
    assert all(np.asarray(cov).shape == (6, 6) for cov in result.state_cov_hist)


def test_simulate_runs_an_attitude_estimator_together_with_an_orbit_estimator():
    os0 = _orbital_state()
    est_sat = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=_sensors())
    result = ADCS.simulate(
        x=TRUTH, satellite=Satellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=_sensors()),
        est_satellite=est_sat,
        estimator=EKF(est_sat, _estimate(7), dt=10.0),
        orbit_estimator=Orbit_GPS(est_sat=est_sat, J2000=0.22, os_template=os0),
        os0=os0, dt=10.0, tf=40.0,
    ).first()  # used to raise AttributeError: 'EstimatedOrbital_State' object has no attribute 'get_state_vector'
    assert len(result.est_state_hist) > 0
    assert result.state_cov_hist is not None and len(result.state_cov_hist) > 0


def test_simulate_falls_back_to_the_one_call_protocol_for_step_less_estimators():
    """A remote proxy only speaks update(u, sensors, os)."""

    class OneCall:
        def __init__(self, inner):
            self.inner = inner
            self.calls = 0

        def update(self, u, sensors, os):
            self.calls += 1
            return self.inner.update(u, sensors, os)

    est_sat = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=_sensors())
    proxy = OneCall(MEKF(est_sat, _estimate(6), dt=10.0))
    result = ADCS.simulate(
        x=TRUTH, satellite=Satellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=_sensors()),
        est_satellite=est_sat, estimator=proxy, os0=_orbital_state(), dt=10.0, tf=40.0,
    ).first()
    assert proxy.calls == len(result.est_state_hist) > 0
