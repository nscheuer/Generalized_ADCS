"""The augmented filters' actuator-bias estimate must reach the propagated mean.

``Satellite.noiseless_rk4`` hard-coded ``ErrorMode(add_bias=False, ...)`` while
``dynJacCore`` linearized about the biased command, so the covariance coupled to
a parameter the dynamics ignored: the linearized filters wound the bias up and
the sigma-point filters froze it (issue #173, related to #62).
"""

from __future__ import annotations

import numpy as np

from ADCS.estimators.attitude_estimators import AugmentedMEKF, AugmentedSRUKF, AugmentedUKF
from ADCS.estimators.process_model import propagate_state
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import Gyro, StarTrackerQuaternion
from ADCS.state import EstimatorState

EPHEM = Ephemeris()
TRUE_BIAS = np.array([0.05, -0.03, 0.02])


def _orbital_state(j2000: float, along_track_s: float = 0.0) -> Orbital_State:
    """Circular-orbit sample ``along_track_s`` seconds past the reference point."""
    angle = along_track_s * 7.5 / 7000.0
    R = 7000.0 * np.array([np.cos(angle), np.sin(angle), 0.0])
    V = 7.5 * np.array([-np.sin(angle), np.cos(angle), 0.0])
    return Orbital_State(ephem=EPHEM, J2000=j2000, R=R, V=V, fast=True)


def _tracker() -> StarTrackerQuaternion:
    tracker = StarTrackerQuaternion(noise=Noise(std_noise=np.full(4, 1.0e-3)))

    def clean(*args, **kwargs):
        state = kwargs.get("x", kwargs.get("state"))
        if state is None and args:
            state = args[0]
        return np.asarray(state.q, float).copy()

    tracker.clean_reading = clean
    return tracker


def _actuators(*, bias=None, estimate=False):
    return [MTQ(axis, max_torque=0.5, noise=Noise(std_noise=1.0e-9),
                bias=None if bias is None and not estimate else
                Bias(bias=np.array([0.0 if bias is None else bias[i]]), std_bias=np.array([1.0e-7])),
                estimate_bias=estimate) for i, axis in enumerate(np.eye(3))]


def _sensors():
    return [Gyro(axis, noise=Noise(std_noise=1.0e-6)) for axis in np.eye(3)] + [_tracker()]


def test_augmented_filters_recover_a_magnetorquer_bias():
    truth_sat = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=_sensors(), actuators=_actuators(bias=TRUE_BIAS))
    est_sat = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=_sensors(), actuators=_actuators(estimate=True))
    assert est_sat.act_bias_len == 3

    for cls in (AugmentedMEKF, AugmentedUKF, AugmentedSRUKF):
        size = 6 + 3
        cov = np.diag([1.0e-6] * 3 + [1.0e-4] * 3 + [1.0e-2] * 3)
        state = EstimatorState(w=[0.02, -0.015, 0.01], q=[0.9, 0.2, -0.3, 0.1], act_bias=np.zeros(3),
                               cov=cov, int_cov=np.zeros((size, size))).normalized()
        estimator = cls(est_sat, state, dt=10.0, unmodeled_dynamics_psd=np.array([1.0e-16] * 3 + [1.0e-9] * 3))
        truth = EstimatorState(w=[0.02, -0.015, 0.01], q=[0.9, 0.2, -0.3, 0.1]).normalized()
        os_prev = _orbital_state(0.22)
        rng = np.random.default_rng(3)
        errors = []
        for k in range(40):
            os_k = _orbital_state(0.22 + (k + 1) * 10.0 * TimeConstants.sec2cent, along_track_s=(k + 1) * 10.0)
            control = 0.2 * rng.standard_normal(3)  # excite the bias through a varying command
            truth = propagate_state(truth, truth_sat, control, 10.0, os_prev, os_k)
            estimator.predict(control, os_prev, os_k)
            estimator.correct(truth_sat.noiseless_sensor_readings(truth, os_k), os_k)
            errors.append(np.linalg.norm(estimator.state.act_bias - TRUE_BIAS) / np.linalg.norm(TRUE_BIAS))
            os_prev = os_k
        assert errors[-1] < 0.25, f"{cls.__name__}: relative bias error {errors[0]:.2f} -> {errors[-1]:.2f}"
        assert errors[-1] < errors[0]
