from __future__ import annotations

import numpy as np

from ADCS.estimators.old_attitude_estimators import SRUAKF
from testing.test_estimators.ukf.scenarios import (
    SimulationResult,
    make_baseline_sensors,
    make_estimate_guess,
    make_mtqs,
    make_orbital_sequence,
    make_orbital_state,
    make_rws,
    make_satellites,
    make_state,
    run_sequence,
    seed,
)
from testing.test_estimators.srukf.helpers import make_srukf


def bias_scenario() -> SimulationResult:
    seed(7)
    bias = np.array([8.0e-4, -6.0e-4, 5.0e-4])
    real_sensors = make_baseline_sensors(gyro_bias=bias, estimate_gyro_bias=False)
    est_sensors = make_baseline_sensors(gyro_bias=np.zeros(3), estimate_gyro_bias=True)
    real_sat, est_sat = make_satellites(sensors=real_sensors, estimated_sensors=est_sensors, actuators=make_mtqs(), estimated_actuators=make_mtqs())
    ukf = make_srukf(est_sat, x_hat=make_estimate_guess(est_sat), dt=5.0, cross_term=False)
    x_true = make_state(w=np.array([1.0e-3, -2.0e-3, 1.5e-3]), q=np.array([1.0, 0.0, 0.0, 0.0]))
    os_sequence = make_orbital_sequence(count=16, dt=5.0, base=make_orbital_state())
    return run_sequence(real_sat, ukf, x_true=x_true, os_sequence=os_sequence)


def reaction_wheel_scenario() -> SimulationResult:
    seed(11)
    sensors = make_baseline_sensors()
    real_sat, est_sat = make_satellites(
        sensors=sensors,
        estimated_sensors=make_baseline_sensors(),
        actuators=make_mtqs() + make_rws(h=0.8),
        estimated_actuators=make_mtqs() + make_rws(h=0.0),
        disturbances=[],
        estimated_disturbances=[],
    )
    x_hat = make_estimate_guess(est_sat, with_rw=True)
    x_hat.h[:] = 0.0
    ukf = make_srukf(est_sat, x_hat=x_hat, dt=5.0, cross_term=False)
    x_true = make_state(w=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]), h=np.array([0.8, 0.8, 0.8]))
    os_sequence = make_orbital_sequence(count=7, dt=5.0, base=make_orbital_state())
    return run_sequence(real_sat, ukf, x_true=x_true, os_sequence=os_sequence)


def startracker_dropout_scenario() -> SimulationResult:
    seed(19)
    from ADCS.satellite_hardware.errors import AnisotropicNoise, Noise
    from ADCS.satellite_hardware.sensors import Gyro, MTM, StarTracker

    real_sensors = [
        *[MTM(axis=axis, noise=Noise(noise=0.0, std_noise=1.0e-8)) for axis in np.eye(3)],
        *[Gyro(axis=axis, noise=Noise(noise=0.0, std_noise=1.0e-5)) for axis in np.eye(3)],
        StarTracker(
            boresight=np.array([0.0, 0.0, 1.0]),
            fov=np.deg2rad(120.0),
            sun_exclusion=np.deg2rad(5.0),
            anisotropic_noise=AnisotropicNoise(std_cross=1.0e-6, std_roll=2.0e-6),
        ),
    ]
    est_sensors = [
        *[MTM(axis=axis, noise=Noise(noise=0.0, std_noise=1.0e-8)) for axis in np.eye(3)],
        *[Gyro(axis=axis, noise=Noise(noise=0.0, std_noise=1.0e-5)) for axis in np.eye(3)],
        StarTracker(
            boresight=np.array([0.0, 0.0, 1.0]),
            fov=np.deg2rad(120.0),
            sun_exclusion=np.deg2rad(5.0),
            anisotropic_noise=AnisotropicNoise(std_cross=1.0e-6, std_roll=2.0e-6),
        ),
    ]
    real_sat, est_sat = make_satellites(sensors=real_sensors, estimated_sensors=est_sensors, actuators=make_mtqs(), estimated_actuators=make_mtqs())
    ukf = make_srukf(est_sat, dt=5.0, cross_term=False)
    x_true = make_state(w=np.array([2.0e-3, -1.0e-3, 1.2e-3]), q=np.array([0.98, 0.08, -0.05, 0.16]))
    os_sequence = make_orbital_sequence(count=14, dt=5.0, base=make_orbital_state())

    def measurement_hook(index, sensors, os, x):
        sensors[6:9] = np.array([0.0, 0.0, 1.0])
        if index % 2 == 1:
            sensors[6:9] = np.nan
        return sensors

    return run_sequence(real_sat, ukf, x_true=x_true, os_sequence=os_sequence, measurement_hook=measurement_hook)
