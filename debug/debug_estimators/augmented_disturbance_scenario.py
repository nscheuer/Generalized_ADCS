"""Shared end-to-end scenario for augmented dipole-disturbance estimation."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import ADCS as ADCS
from ADCS.helpers.plotting.plot_estimator import plot_error_and_sun


class SimulationAugmentedEstimator:
    """Adapt a new estimator to the simulation update protocol."""

    def __init__(self, estimator_type, *args, **kwargs) -> None:
        self.estimator = estimator_type(*args, **kwargs)
        self._previous_orbital_state = None

    def __getattr__(self, name):
        return getattr(self.estimator, name)

    def update(self, u: np.ndarray, sensors: np.ndarray, os: ADCS.Orbital_State) -> ADCS.EstimatorState:
        if self._previous_orbital_state is None:
            self._previous_orbital_state = os
            return self.estimator.correct(sensors, os)
        os_start = self._previous_orbital_state
        self._previous_orbital_state = os
        return self.estimator.step(u, sensors, os_start, os, midpoint_orbital_state=os)


def run(estimator_type, title: str, *, direct_torque: bool = False):
    np.random.seed(7)
    dt = 10.0

    # The residual dipole is deliberately nonzero in truth but starts at zero
    # in the estimated model.  No sensor or actuator biases are estimated here.
    true_dipole = np.array([0.4, -0.3, 0.2])
    mtm_noise = ADCS.Noise(std_noise=5.0e-8)
    gyro_noise = ADCS.Noise(std_noise=5.0e-7)
    sun_noise = ADCS.Noise(std_noise=1.0e-3)

    real_sensors = [ADCS.MTM(axis, noise=mtm_noise.copy()) for axis in np.eye(3)]
    real_sensors += [ADCS.Gyro(axis, noise=gyro_noise.copy()) for axis in np.eye(3)]
    real_sensors += [ADCS.SunPair(axis, efficiency=0.3, noise=sun_noise.copy()) for axis in np.eye(3)]
    truth_disturbances = (
        [ADCS.disturbances.Dipole_Disturbance(true_dipole), ADCS.disturbances.GG_Disturbance()]
        if direct_torque else [ADCS.disturbances.Dipole_Disturbance(true_dipole)]
    )
    satellite = ADCS.Satellite(
        mass=4.0,
        J_0=np.diag([3.4, 2.9, 1.3]),
        sensors=real_sensors,
        disturbances=truth_disturbances,
    )
    x_0 = ADCS.State.from_array(np.array([0.0012, 0.0008, -0.0018, 0.2588, 0.0, 0.9659, 0.0]))

    est_sensors = [ADCS.MTM(axis, noise=mtm_noise.copy()) for axis in np.eye(3)]
    est_sensors += [ADCS.Gyro(axis, noise=gyro_noise.copy()) for axis in np.eye(3)]
    est_sensors += [ADCS.SunPair(axis, efficiency=0.3, noise=sun_noise.copy()) for axis in np.eye(3)]
    disturbance_type = ADCS.disturbances.Torque_Disturbance if direct_torque else ADCS.disturbances.Dipole_Disturbance
    est_disturbance = disturbance_type(np.zeros(3), estimate_dist=True)
    est_disturbance.parameter_std_rate = np.full(3, 1.0e-5)
    est_satellite = ADCS.EstimatedSatellite(
        mass=4.2,
        J_0=np.diag([3.5, 3.0, 1.4]),
        sensors=est_sensors,
        disturbances=[est_disturbance],
    )
    assert est_satellite.act_bias_len == 0
    assert est_satellite.att_sens_bias_len == 0
    assert est_satellite.dist_param_len == 3

    # The filter starts inside the local attitude-estimation basin (5 deg),
    # but has no prior knowledge of the residual dipole.
    if estimator_type is ADCS.AugmentedEKF:
        x_hat = ADCS.EstimatorState(
            w=[0.0012, 0.0008, -0.0018],
            q=[np.cos(np.deg2rad(72.5)), 0.0, np.sin(np.deg2rad(72.5)), 0.0],
            dist_param=np.zeros(3),
            cov=np.diag([0.01**2] * 3 + [0.15**2] * 4 + [0.5**2] * 3),
            int_cov=np.diag([1.0e-16] * 3 + [0.0, 1.0e-8, 1.0e-8, 1.0e-8] + [1.0e-10] * 3),
        )
        process_psd = np.array(
            [1.0e-16 / dt] * 3 + [0.0, 1.0e-8 / dt, 1.0e-8 / dt, 1.0e-8 / dt]
        )
    else:
        x_hat = ADCS.EstimatorState(
            w=[0.0012, 0.0008, -0.0018],
            q=[np.cos(np.deg2rad(72.5)), 0.0, np.sin(np.deg2rad(72.5)), 0.0],
            dist_param=np.zeros(3),
            cov=block_diag(np.eye(3) * 0.01**2, np.eye(3) * 0.15**2, np.eye(3) * 0.5**2),
            int_cov=block_diag(np.eye(3) * 1.0e-16, np.eye(3) * 1.0e-8, np.eye(3) * 1.0e-10),
        )
        process_psd = np.array([1.0e-16 / dt] * 3 + [1.0e-8 / dt] * 3)

    estimator = SimulationAugmentedEstimator(
        estimator_type, est_satellite, x_hat, dt=dt, unmodeled_dynamics_psd=process_psd
    )
    os0 = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(), J2000=0.22,
        R=np.array([5000.0, 0.0, 5000.0]), V=np.array([0.0, -7.5, 0.0]),
    )
    results = ADCS.simulate(
        x=x_0, satellite=satellite, est_satellite=est_satellite,
        estimator=estimator, os0=os0, dt=dt, tf=2000.0,
    )

    final_estimate = results.first().est_state_hist[-1].dist_param
    if direct_torque:
        print(f"{title} final direct-torque estimate: {final_estimate}")
    else:
        print(f"{title} final dipole estimate: {final_estimate}; truth: {true_dipole}")

    ADCS.plot(
        results,
        ADCS.plots.AttitudePlot(sources=["real", "estimated"]),
        layout=(1, 1), title=title,
    )
    ADCS.plot(
        results,
        ADCS.plots.QuaternionPlot(sources=["real", "estimated"]),
        ADCS.plots.AngularVelocityPlotCombined(sources=["real", "estimated"]),
        ADCS.plots.DisturbanceParameterPlot(
            labels=[r"$\tau_x$", r"$\tau_y$", r"$\tau_z$"] if direct_torque else [r"$m_x$", r"$m_y$", r"$m_z$"],
            units="N m" if direct_torque else "A m²", plot_torque=direct_torque,
        ),
        ADCS.plots.SensorsPlot(title="Sensor Readings", sources=["real", "clean"]),
        layout=(2, 2), title=f"{title}: State and Disturbance Estimate",
    )
    run_result = results.first()
    plot_error_and_sun(run_result.time_s, run_result.state_hist, run_result.est_state_hist, run_result.os_hist)
    plt.show()
    return results
