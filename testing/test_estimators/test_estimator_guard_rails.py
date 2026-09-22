"""Guard rails: calls and configurations that used to fail silently or late.

* ``step(control, measurements)`` (the pre-2.0 calling convention) was routed
  to ``correct(measurements, orbital_state)``; with seven controls and a
  seven-row measurement stack the control vector was corrected against as if
  it were the measurement.
* ``update(sensors=, os=)`` without a control vector predicted with none.
* ``predict()`` never compared ``dt`` with the orbital-state gap; a 10x
  mismatch took a 16 degree error to 47.
* A scalar ``Noise``/``Bias`` on a multi-axis sensor stayed size 1 and failed
  inside the filter instead of at construction.
* A quaternion star tracker with ``estimate_bias=True`` registered a
  four-wide bias block and failed later inside the augmented filters.
* The pre-2.0 ``UAKF``/``SRUAKF`` alias stubs still shipped although their
  constructor arguments no longer worked.
"""

from __future__ import annotations

import numpy as np
import pytest

from ADCS.estimators.attitude_estimators import MEKF, AugmentedMEKF
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware import disturbances as D
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import EarthHorizonSensor, Gyro, StarTrackerQuaternion
from ADCS.state import EstimatorState


EPHEM = Ephemeris()


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


def _satellite(disturbances=()) -> EstimatedSatellite:
    return EstimatedSatellite(
        J_0=np.diag([0.5, 0.8, 1.2]),
        sensors=[Gyro(axis, noise=Noise(std_noise=1.0e-4)) for axis in np.eye(3)] + [_tracker()],
        disturbances=list(disturbances),
    )


def _state(size: int = 6, **blocks) -> EstimatorState:
    return EstimatorState(
        w=[0.02, -0.015, 0.01], q=[0.9, 0.2, -0.3, 0.1],
        cov=np.eye(size) * 1.0e-3, int_cov=np.zeros((size, size)), **blocks,
    ).normalized()


def test_scalar_noise_on_a_vector_sensor_is_broadcast_per_axis():
    horizon = EarthHorizonSensor(noise=Noise(std_noise=0.01))  # three outputs, scalar model
    assert horizon.noise.std_noise.shape == (3,)
    assert horizon.measurement_covariance().shape == (3, 3)

    tracker = StarTrackerQuaternion()  # default models must size to four
    assert tracker.noise.std_noise.shape == (4,)
    assert tracker.bias.bias.shape == (4,)
    tracker.clean_reading = _tracker().clean_reading
    satellite = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=[tracker])
    estimator = MEKF(satellite, _state(), dt=1.0)
    os0 = _orbital_state(0.22)
    estimator.predict(np.empty(0), os0, os0)
    estimator.correct(satellite.measurement_stack.predict(_state(), os0), os0)  # used to fail on a 1x1 R

    with pytest.raises(ValueError, match="entries"):
        EarthHorizonSensor(noise=Noise(std_noise=np.full(2, 0.01)))


def test_quaternion_star_tracker_bias_estimation_is_rejected_at_assembly():
    tracker = StarTrackerQuaternion(bias=Bias(bias=np.zeros(4), std_bias=np.full(4, 1.0e-9)), estimate_bias=True)
    with pytest.raises(ValueError, match="cannot be estimated"):
        EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=[tracker])


def test_two_argument_step_rejects_a_legacy_control_measurement_call():
    """With 7 controls and a 7-row stack the old step(control, measurements)
    used to be corrected silently with the control as the measurement."""
    from ADCS.satellite_hardware.actuators import MTQ
    satellite = EstimatedSatellite(
        J_0=np.diag([0.5, 0.8, 1.2]),
        sensors=[Gyro(axis, noise=Noise(std_noise=1.0e-4)) for axis in np.eye(3)] + [_tracker()],
        actuators=[MTQ(np.eye(3)[i % 3], max_torque=0.05, noise=Noise(std_noise=1.0e-6)) for i in range(7)],
    )
    estimator = MEKF(satellite, _state(), dt=1.0)
    with pytest.raises(TypeError, match="Orbital_State"):
        estimator.step(np.full(7, 0.3), np.zeros(7))
    with pytest.raises(TypeError, match="update requires"):
        estimator.update(sensors=np.zeros(7), os=_orbital_state(0.22))


def test_predict_warns_when_dt_disagrees_with_the_orbital_state_gap():
    satellite = _satellite()
    os_start = _orbital_state(0.22)
    os_end = _orbital_state(0.22 + 100.0 * TimeConstants.sec2cent, along_track_s=100.0)
    estimator = MEKF(satellite, _state(), dt=10.0)
    with pytest.warns(UserWarning, match="apart"):
        estimator.predict(np.empty(0), os_start, os_end)  # 10 s step over a 100 s gap
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MEKF(satellite, _state(), dt=100.0).predict(np.empty(0), os_start, os_end)  # matched
        MEKF(satellite, _state(), dt=10.0).predict(np.empty(0), os_start, os_start)  # static orbit


def test_legacy_estimator_package_is_gone():
    """0.2.x ships UKF/SRUKF only; the pre-2.0 UAKF/SRUAKF stubs were removed."""
    import importlib.util

    for name in ("ADCS.estimators.old_attitude_estimators.attitude_UAKF",
                 "ADCS.estimators.old_attitude_estimators.attitude_SRUAKF",
                 "ADCS.estimators.old_attitude_estimators.attitude_estimator"):
        assert importlib.util.find_spec(name) is None, name
