"""Shared basic and gyro-bias scenarios for augmented UKF-family filters."""

from __future__ import annotations

import numpy as np

import ADCS
from debug.debug_estimators.augmented_disturbance_scenario import SimulationAugmentedEstimator


def _orbital_state():
    return ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(), J2000=0.22,
        R=np.array([5000.0, 0.0, 5000.0]), V=np.array([0.0, -7.5, 0.0]),
    )


def _sensors(biases, *, estimate_bias):
    sensors = [ADCS.MTM(axis, noise=ADCS.Noise(std_noise=5e-8)) for axis in np.eye(3)]
    sensors += [ADCS.Gyro(axis, noise=ADCS.Noise(std_noise=5e-7),
                           bias=ADCS.Bias(bias=value, std_bias=1e-10 if estimate_bias else 0.0),
                           estimate_bias=estimate_bias)
                for axis, value in zip(np.eye(3), biases)]
    sensors += [ADCS.SunPair(axis, efficiency=0.3, noise=ADCS.Noise(std_noise=1e-3))
                for axis in np.eye(3)]
    return sensors


def _state(filter_type, *, bias=False):
    tangent = filter_type in (ADCS.AugmentedUKF, ADCS.AugmentedSRUKF)
    covariance = (np.diag([0.01**2] * 3 + [0.15**2] * (3 if tangent else 4)
                          + ([0.05**2] * 3 if bias else [])))
    return ADCS.EstimatorState(
        w=[0.0012, 0.0008, -0.0018],
        q=[np.cos(np.deg2rad(72.5)), 0.0, np.sin(np.deg2rad(72.5)), 0.0],
        sens_bias=np.zeros(3) if bias else np.empty(0),
        cov=covariance,
        int_cov=np.zeros_like(covariance),
    )


def run_basic(filter_type, title):
    dt = 20.0
    satellite = ADCS.Satellite(
        mass=3000.0, J_0=np.diag([500.0, 1500.0, 1500.0]),
        sensors=_sensors(np.zeros(3), estimate_bias=False),
        disturbances=[ADCS.disturbances.GG_Disturbance()],
    )
    estimated = ADCS.EstimatedSatellite(
        mass=3200.0, J_0=np.diag([450.0, 1400.0, 1400.0]),
        sensors=_sensors(np.zeros(3), estimate_bias=False),
        disturbances=[ADCS.disturbances.GG_Disturbance()],
    )
    estimator = SimulationAugmentedEstimator(
        filter_type, estimated, _state(filter_type), dt=dt,
        unmodeled_dynamics_psd=np.array([1e-16 / dt] * 3 + [1e-8 / dt] * 3),
    )
    return _simulate(satellite, estimated, estimator, title, dt)


def run_gyro_bias(filter_type, title):
    dt = 20.0
    true_bias = np.array([6e-4, -4e-4, 5e-4])
    satellite = ADCS.Satellite(
        mass=3000.0, J_0=np.diag([500.0, 1500.0, 1500.0]),
        sensors=_sensors(true_bias, estimate_bias=False),
        disturbances=[ADCS.disturbances.GG_Disturbance()],
    )
    estimated = ADCS.EstimatedSatellite(
        mass=3200.0, J_0=np.diag([450.0, 1400.0, 1400.0]),
        sensors=_sensors(np.zeros(3), estimate_bias=True),
        disturbances=[ADCS.disturbances.GG_Disturbance()],
    )
    estimator = SimulationAugmentedEstimator(
        filter_type, estimated, _state(filter_type, bias=True), dt=dt,
        unmodeled_dynamics_psd=np.array([1e-16 / dt] * 3 + [1e-8 / dt] * 3),
    )
    results = _simulate(satellite, estimated, estimator, title, dt)
    final = results.first().est_state_hist[-1].sens_bias
    print("true gyro bias:", true_bias)
    print("final estimated gyro bias:", final)
    np.testing.assert_allclose(final, true_bias, atol=1e-4)
    return results


def _simulate(satellite, estimated, estimator, title, dt):
    x = ADCS.State.from_array([0.001, 0.001, -0.002, 0.2588, 0.0, 0.9659, 0.0])
    results = ADCS.simulate(x=x, satellite=satellite, est_satellite=estimated,
                            estimator=estimator, os0=_orbital_state(), dt=dt, tf=2000.0)
    ADCS.plot(results, ADCS.plots.AttitudePlot(sources=["real", "estimated"]),
              layout=(1, 1), title=title)
    return results
