"""Compare warm and cold starts of the same augmented MEKF."""

import matplotlib.pyplot as plt
import numpy as np

import ADCS
from ADCS.estimators.initialization import WarmStart
from ADCS.helpers.math_helpers import quaternion_angle


dt = 10.0
axes = np.vstack((np.eye(3), -np.eye(3)))
gyro_bias = [0.002, -0.0015, 0.001]
true_attitude = np.array([np.sqrt(0.5), 0.5, 0.5, 0.0])
process_noise = np.diag([1e-8] * 3 + [1e-6] * 3 + [1e-9] * 3)


def make_sensors(bias_values, estimate_bias=False):
    sun = [
        ADCS.SunSensor(axis=axis, efficiency=1.0, sample_time=dt,
                       noise=ADCS.Noise(std_noise=0.03))
        for axis in axes
    ]
    gyros = [
        ADCS.Gyro(axis=axis, sample_time=dt,
                  noise=ADCS.Noise(std_noise=0.001),
                  bias=ADCS.Bias(bias=bias), estimate_bias=estimate_bias)
        for axis, bias in zip(np.eye(3), bias_values)
    ]
    magnetic = [
        ADCS.MTM(axis=axis, sample_time=dt,
                 noise=ADCS.Noise(std_noise=2e-6))
        for axis in np.eye(3)
    ]
    return sun + gyros + magnetic


def simulate(warm):
    np.random.seed(17)  # Same noise realization in both runs.
    satellite = ADCS.Satellite(
        mass=4.0, J_0=np.eye(3) * 0.02,
        sensors=make_sensors(gyro_bias), disturbances=[],
    )
    estimated_satellite = ADCS.EstimatedSatellite(
        mass=4.0, J_0=np.eye(3) * 0.02,
        sensors=make_sensors([0.0, 0.0, 0.0], estimate_bias=True),
        disturbances=[],
    )
    orbit = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(), J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]),
    )
    if warm:
        initial = WarmStart(method="quest", sens_bias_std=0.004,
                            process_noise=process_noise)
    else:
        initial = ADCS.EstimatorState(
            w=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]),
            sens_bias=np.zeros(3),
            cov=np.diag([0.003**2] * 3 + [0.5**2] * 3 + [0.004**2] * 3),
            int_cov=process_noise,
        )
    estimator = ADCS.AugmentedMEKF(estimated_satellite, initial, dt=dt)
    return ADCS.simulate(
        x=ADCS.State(w=np.array([0.01, -0.008, 0.012]), q=true_attitude),
        satellite=satellite, est_satellite=estimated_satellite,
        estimator=estimator, os0=orbit, dt=dt, tf=600.0,
    ).first()


cold = simulate(warm=False)
warm = simulate(warm=True)


def angle_errors(run):
    return np.rad2deg([
        quaternion_angle(truth.q, estimate.q)
        for truth, estimate in zip(run.state_hist, run.est_state_hist)
    ])


plt.figure(figsize=(9, 4.5))
plt.plot(cold.time_s, angle_errors(cold), label="Cold start")
plt.plot(warm.time_s, angle_errors(warm), label="Warm start")
plt.xlabel("Time [s]")
plt.ylabel("Attitude angle error [deg]")
plt.title("MEKF initial attitude: cold versus warm start")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
