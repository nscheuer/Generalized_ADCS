"""Shared, reproducible scenario for estimator comparison examples."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

import numpy as np

import ADCS
from ADCS.helpers.math_helpers import quaternion_angle


DT = 20.0
TF = 600.0
SEED = 2021


def _sensors():
    sensors = [ADCS.MTM(axis, noise=ADCS.Noise(std_noise=5.0e-8)) for axis in np.eye(3)]
    sensors += [ADCS.Gyro(axis, noise=ADCS.Noise(std_noise=5.0e-7)) for axis in np.eye(3)]
    sensors += [ADCS.SunPair(axis, efficiency=0.3, noise=ADCS.Noise(std_noise=1.0e-3)) for axis in np.eye(3)]
    return sensors


def _initial_state(filter_type, half_angle_deg=50.0):
    # The same imperfect initial estimate is used for all four filters by default.
    attitude = [np.cos(np.deg2rad(half_angle_deg)), 0.0, np.sin(np.deg2rad(half_angle_deg)), 0.0]
    if filter_type is ADCS.EKF:
        return ADCS.EstimatorState(
            w=[0.0012, 0.0008, -0.0018], q=attitude,
            cov=np.diag([0.01**2] * 3 + [0.15**2] * 4),
            int_cov=np.diag([1.0e-16] * 3 + [0.0, 1.0e-8, 1.0e-8, 1.0e-8]),
        )
    return ADCS.EstimatorState(
        w=[0.0012, 0.0008, -0.0018], q=attitude,
        cov=np.diag([0.01**2] * 3 + [0.15**2] * 3),
        int_cov=np.diag([1.0e-16] * 3 + [1.0e-8] * 3),
    )


def run(filter_type, warm_start=None, *, initial_half_angle_deg=50.0):
    """Simulate one filter on identical truth, orbit and noise realizations."""
    np.random.seed(SEED)
    satellite = ADCS.Satellite(
        mass=3000.0, J_0=np.diag([500.0, 1500.0, 1500.0]),
        sensors=_sensors(), disturbances=[ADCS.disturbances.GG_Disturbance()],
    )
    estimated = ADCS.EstimatedSatellite(
        mass=3200.0, J_0=np.diag([450.0, 1400.0, 1400.0]),
        sensors=_sensors(), disturbances=[ADCS.disturbances.GG_Disturbance()],
    )
    initial_state = (
        warm_start if warm_start is not None
        else _initial_state(filter_type, half_angle_deg=initial_half_angle_deg)
    )
    estimator = filter_type(estimated, initial_state, dt=DT)
    ephem = ADCS.Ephemeris()
    probe = ADCS.Orbital_State(
        ephem=ephem, J2000=0.22, R=[7000.0, 0.0, 0.0], V=[0.0, 7.5, 0.0],
    )
    sun = probe.S / np.linalg.norm(probe.S)
    velocity = np.cross(sun, [0.0, 0.0, 1.0])
    velocity = 7.5 * velocity / np.linalg.norm(velocity)
    os0 = ADCS.Orbital_State(ephem=ephem, J2000=0.22, R=7000.0 * sun, V=velocity)
    truth = ADCS.State.from_array([0.001, 0.001, -0.002, 0.2588, 0.0, 0.9659, 0.0])
    result = ADCS.simulate(
        x=truth, satellite=satellite, est_satellite=estimated, estimator=estimator,
        os0=os0, dt=DT, tf=TF,
    ).first()
    error = np.rad2deg([
        quaternion_angle(real.q, estimate.q)
        for real, estimate in zip(result.state_hist, result.est_state_hist)
    ])
    # Both the initial guess and the first logged estimate refer to t=0.
    # A warm start obtains that guess from the first sensor readings.
    guess_q = (
        estimator.warm_start_attitude.quaternion
        if warm_start is not None else initial_state.q
    )
    initial_error = np.rad2deg(quaternion_angle(truth.q, guess_q))
    return np.insert(np.asarray(result.time_s), 0, 0.0), np.insert(error, 0, initial_error)
