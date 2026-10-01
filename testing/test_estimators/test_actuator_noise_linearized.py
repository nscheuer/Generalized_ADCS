"""The linearized filters carry actuator command noise (September 2026 audit).

Before the fix EKF/MEKF built their process noise from ``unmodeled_dynamics_psd``
and the bias random walks only, so ``Actuator.control_covariance()`` never
reached their prediction: with 1e-4 N m of reaction-wheel torque noise the
UKF's rate-block variance after one 10 s step was 6.4e-6 while the EKF's and
MEKF's stayed at the prior's own 3.3e-10 (19,500x apart). Twenty-run NEES
was 1e5 at dt 1 s and 4e6 at dt 10 s with factory actuator noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from ADCS.estimators.attitude_estimators import EKF, MEKF, SRUKF, UKF
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import Gyro, StarTrackerQuaternion
from ADCS.state import EstimatorState

DT = 10.0
OS0 = Orbital_State(ephem=Ephemeris(), J2000=0.22, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]), fast=True)
OS1 = Orbital_State(ephem=Ephemeris(), J2000=0.22 + DT * TimeConstants.sec2cent, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]), fast=True)
COMMAND = np.array([1.0e-3, -2.0e-3, 5.0e-4, 0.01, -0.02, 0.005])  # inside every actuator limit: no one-sided clipping of sigma points
MTQ_NOISE = 0.005
RW_NOISE = 1.0e-4


def _satellite(rw_noise: float, mtq_noise: float) -> EstimatedSatellite:
    tracker = StarTrackerQuaternion(noise=Noise(std_noise=np.full(4, 1.0e-3)))
    tracker.clean_reading = lambda *a, **k: np.asarray((k.get("x") or k.get("state") or a[0]).q, float).copy()
    actuators = ([RW(axis=axis, max_torque=0.02, J=1.0e-3, h=0.0, h_max=1.0, noise=Noise(std_noise=rw_noise)) for axis in np.eye(3)]
                 + [MTQ(axis, max_torque=0.05, noise=Noise(std_noise=mtq_noise)) for axis in np.eye(3)])
    sensors = [Gyro(axis, noise=Noise(std_noise=1.0e-4)) for axis in np.eye(3)] + [tracker]
    return EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=sensors, actuators=actuators)


def _prior(full_quaternion: bool) -> EstimatorState:
    """A near-zero prior with zero PSD, so any covariance growth is actuator noise."""
    size = 3 + (4 if full_quaternion else 3) + 3
    diagonal = [1.0e-10] * 3 + ([0.0] + [1.0e-10] * 3 if full_quaternion else [1.0e-10] * 3) + [1.0e-10] * 3
    return EstimatorState(w=[0.02, -0.015, 0.01], q=[1.0, 0.0, 0.0, 0.0], h=np.zeros(3), cov=np.diag(diagonal),
                          int_cov=np.zeros((size, size)))


def _rate_variance(filter_class, rw_noise: float, mtq_noise: float) -> float:
    estimator = filter_class(_satellite(rw_noise, mtq_noise), _prior(filter_class is EKF), dt=DT)
    predicted = estimator.predict(COMMAND, OS0, OS1)
    block = predicted.slice("angular_velocity", coordinates=estimator.covariance_coordinates)
    return float(np.trace(predicted.covariance.as_matrix()[block, block]))


@pytest.mark.parametrize("filter_class", [EKF, MEKF])
def test_linearized_prediction_grows_with_actuator_command_noise(filter_class):
    perfect = _rate_variance(filter_class, 0.0, 0.0)
    noisy = _rate_variance(filter_class, RW_NOISE, MTQ_NOISE)
    assert perfect < 1.0e-8, "a near-zero prior with zero PSD should barely grow with perfect actuators"
    assert noisy > 1.0e3 * perfect, f"{filter_class.__name__} prediction ignores the actuator noise ({noisy:.3e} vs {perfect:.3e})"


@pytest.mark.parametrize("filter_class", [EKF, MEKF])
def test_linearized_actuator_noise_matches_the_unscented_control_sigma_points(filter_class):
    """The zero-order-hold term and the UKF's control sigma points model the same plant.

    Compared on the growth over the perfect-actuator baseline so the prior's own
    propagation drops out. The UKF's control points carry the second-order
    gyroscopic response to a 4e-4 N m torque perturbation held for 10 s (a rate
    change of 40% of the rate itself), which the linear term omits: measured
    1.2% apart at 10 s and shrinking with the step.
    """
    linearized = _rate_variance(filter_class, RW_NOISE, MTQ_NOISE) - _rate_variance(filter_class, 0.0, 0.0)
    unscented = _rate_variance(UKF, RW_NOISE, MTQ_NOISE) - _rate_variance(UKF, 0.0, 0.0)
    assert linearized == pytest.approx(unscented, rel=5.0e-2)


def test_unscented_family_is_unchanged_by_the_linearized_term():
    """The UKF path must not add the term on top of its control sigma points (no double count)."""
    assert _rate_variance(SRUKF, RW_NOISE, MTQ_NOISE) == pytest.approx(_rate_variance(UKF, RW_NOISE, MTQ_NOISE), rel=1.0e-6)
