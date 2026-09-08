"""End-to-end tracking contracts for the augmented EKF and MEKF."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import block_diag

from ADCS.estimators.attitude_estimators import AugmentedEKF, AugmentedMEKF
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.disturbances import Dipole_Disturbance
from ADCS.satellite_hardware.satellite import EstimatedSatellite, Satellite
from ADCS.satellite_hardware.sensors import Gyro, MTM, SunPair
from ADCS.state import EstimatorState, State


TRUE_GYRO_BIAS = np.array([6.0e-4, -4.0e-4, 5.0e-4])
DT = 20.0


def _sensors(gyro_bias: np.ndarray, *, estimate_bias: bool) -> list:
    mtm_noise = Noise(std_noise=5.0e-8)
    gyro_noise = Noise(std_noise=5.0e-7)
    sun_noise = Noise(std_noise=1.0e-3)
    sensors = [MTM(axis, noise=mtm_noise.copy()) for axis in np.eye(3)]
    sensors += [
        Gyro(
            axis,
            noise=gyro_noise.copy(),
            bias=Bias(bias=value, std_bias=1.0e-10 if estimate_bias else 0.0),
            estimate_bias=estimate_bias,
        )
        for axis, value in zip(np.eye(3), gyro_bias)
    ]
    sensors += [SunPair(axis, efficiency=0.3, noise=sun_noise.copy()) for axis in np.eye(3)]
    return sensors


def _orbital_state() -> Orbital_State:
    return Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([5000.0, 0.0, 5000.0]),
        V=np.array([0.0, -7.5, 0.0]),
    )


@pytest.mark.parametrize("filter_type", [AugmentedEKF, AugmentedMEKF])
def test_augmented_filters_recover_constant_gyro_biases_in_tracking_regime(filter_type):
    """Bias states converge once a coarse attitude solution has been acquired.

    A linearized EKF/MEKF is a tracking filter.  The initial attitude is five
    degrees from truth, which keeps the vector-sensor measurement Jacobians in
    their local validity region while still requiring a real correction.
    """
    np.random.seed(7)
    satellite = Satellite(
        mass=3000.0,
        J_0=np.diag([500.0, 1500.0, 1500.0]),
        sensors=_sensors(TRUE_GYRO_BIAS, estimate_bias=False),
    )
    estimated_satellite = EstimatedSatellite(
        mass=3200.0,
        J_0=np.diag([450.0, 1400.0, 1400.0]),
        sensors=_sensors(np.zeros(3), estimate_bias=True),
    )

    truth = State.from_array([0.001, 0.001, -0.002, 0.2588, 0.0, 0.9659, 0.0])
    quaternion = [np.cos(np.deg2rad(72.5)), 0.0, np.sin(np.deg2rad(72.5)), 0.0]
    if filter_type is AugmentedEKF:
        covariance = np.diag([0.01**2] * 3 + [0.15**2] * 4 + [0.05**2] * 3)
        process_psd = np.array(
            [1.0e-16 / DT] * 3
            + [0.0, 1.0e-8 / DT, 1.0e-8 / DT, 1.0e-8 / DT]
        )
    else:
        covariance = block_diag(
            np.eye(3) * 0.01**2,
            np.eye(3) * 0.15**2,
            np.eye(3) * 0.05**2,
        )
        process_psd = np.array([1.0e-16 / DT] * 3 + [1.0e-8 / DT] * 3)
    state = EstimatorState(
        w=[0.0012, 0.0008, -0.0018],
        q=quaternion,
        sens_bias=np.zeros(3),
        cov=covariance,
        int_cov=np.zeros_like(covariance),
    )
    estimator = filter_type(
        estimated_satellite,
        state,
        dt=DT,
        unmodeled_dynamics_psd=process_psd,
    )
    orbital_state = _orbital_state()
    control = np.empty(0)

    for index in range(100):
        measurements = satellite.sensor_readings(truth, orbital_state)
        if index == 0:
            estimate = estimator.correct(measurements, orbital_state)
        else:
            estimate = estimator.step(
                control,
                measurements,
                orbital_state,
                orbital_state,
                midpoint_orbital_state=orbital_state,
            )
        truth = satellite.noiseless_rk4(
            truth, control, DT, orbital_state, orbital_state, quat_as_vec=True
        ).normalized()

    np.testing.assert_allclose(estimate.sens_bias, TRUE_GYRO_BIAS, atol=1.0e-4)
    assert estimate.covariance.dimension == covariance.shape[0]


@pytest.mark.parametrize("filter_type", [AugmentedEKF, AugmentedMEKF])
def test_augmented_filters_recover_active_dipole_disturbance(filter_type):
    """A nonzero residual dipole is recovered through the dynamics path."""
    np.random.seed(7)
    true_dipole = np.array([0.4, -0.3, 0.2])
    satellite = Satellite(
        mass=4.0,
        J_0=np.diag([3.4, 2.9, 1.3]),
        sensors=_sensors(np.zeros(3), estimate_bias=False),
        disturbances=[Dipole_Disturbance(true_dipole)],
    )
    estimated_satellite = EstimatedSatellite(
        mass=4.2,
        J_0=np.diag([3.5, 3.0, 1.4]),
        sensors=_sensors(np.zeros(3), estimate_bias=False),
        disturbances=[Dipole_Disturbance(np.zeros(3), estimate_dist=True)],
    )
    estimated_satellite.disturbances[0].parameter_std_rate = np.full(3, 1.0e-5)

    truth = State.from_array([0.0012, 0.0008, -0.0018, 0.2588, 0.0, 0.9659, 0.0])
    quaternion = [np.cos(np.deg2rad(72.5)), 0.0, np.sin(np.deg2rad(72.5)), 0.0]
    if filter_type is AugmentedEKF:
        covariance = np.diag([0.01**2] * 3 + [0.15**2] * 4 + [0.5**2] * 3)
        process_psd = np.array(
            [1.0e-16 / 10.0] * 3 + [0.0, 1.0e-8 / 10.0, 1.0e-8 / 10.0, 1.0e-8 / 10.0]
        )
    else:
        covariance = block_diag(
            np.eye(3) * 0.01**2, np.eye(3) * 0.15**2, np.eye(3) * 0.5**2
        )
        process_psd = np.array([1.0e-16 / 10.0] * 3 + [1.0e-8 / 10.0] * 3)
    state = EstimatorState(
        w=[0.0012, 0.0008, -0.0018], q=quaternion, dist_param=np.zeros(3),
        cov=covariance, int_cov=np.zeros_like(covariance),
    )
    estimator = filter_type(
        estimated_satellite, state, dt=10.0, unmodeled_dynamics_psd=process_psd
    )
    orbital_state = _orbital_state()
    control = np.empty(0)

    for index in range(200):
        measurements = satellite.sensor_readings(truth, orbital_state)
        if index == 0:
            estimate = estimator.correct(measurements, orbital_state)
        else:
            estimate = estimator.step(
                control, measurements, orbital_state, orbital_state,
                midpoint_orbital_state=orbital_state,
            )
        truth = satellite.noiseless_rk4(
            truth, control, 10.0, orbital_state, orbital_state, quat_as_vec=True
        ).normalized()

    np.testing.assert_allclose(estimate.dist_param, true_dipole, atol=0.08)
    assert estimate.covariance.dimension == covariance.shape[0]
