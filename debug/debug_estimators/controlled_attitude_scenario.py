"""Shared controller-in-the-loop scenario for the new attitude estimators."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import ADCS
from ADCS.helpers.plotting.plot_estimator import plot_error_and_sun


def run(estimator_type, title: str) -> None:
    """Run one controlled, no-bias attitude-estimation scenario."""
    np.random.seed(7)
    dt = 2.0
    mtm_noise = ADCS.Noise(std_noise=5.0e-8)
    gyro_noise = ADCS.Noise(std_noise=5.0e-7)
    sun_noise = ADCS.Noise(std_noise=1.0e-3)

    def sensors():
        result = [ADCS.MTM(axis, noise=mtm_noise.copy()) for axis in np.eye(3)]
        result += [ADCS.Gyro(axis, noise=gyro_noise.copy()) for axis in np.eye(3)]
        result += [
            ADCS.SunPair(axis, efficiency=0.3, noise=sun_noise.copy())
            for axis in np.eye(3)
        ]
        return result

    def actuators():
        return [ADCS.MTQ(axis, max_torque=0.1) for axis in np.eye(3)]

    satellite = ADCS.Satellite(
        mass=4.0,
        J_0=np.diag([3.4, 2.9, 1.3]),
        actuators=actuators(),
        sensors=sensors(),
        disturbances=[ADCS.disturbances.GG_Disturbance()],
    )
    est_satellite = ADCS.EstimatedSatellite(
        mass=4.2,
        J_0=np.diag([3.5, 3.0, 1.4]),
        actuators=actuators(),
        sensors=sensors(),
        disturbances=[ADCS.disturbances.GG_Disturbance()],
    )

    # The estimator starts close enough for local EKF/MEKF linearization,
    # while the controller still has a real pointing error to remove.
    x_0 = ADCS.State(
        w=[0.002, -0.001, 0.0015],
        q=[np.cos(np.deg2rad(10.0)), 0.0, np.sin(np.deg2rad(10.0)), 0.0],
    )
    x_hat = ADCS.EstimatorState(
        w=[0.0022, -0.0012, 0.0013],
        q=[np.cos(np.deg2rad(7.5)), 0.0, np.sin(np.deg2rad(7.5)), 0.0],
        cov=np.diag(
            [0.01**2] * 3
            + [0.15**2] * (4 if estimator_type is ADCS.EKF else 3)
        ),
        int_cov=np.zeros((7 if estimator_type is ADCS.EKF else 6,) * 2),
    )
    process_psd = np.array([1.0e-16 / dt] * 3 + [1.0e-8 / dt] * 3)
    if estimator_type is ADCS.EKF:
        process_psd = np.array(
            [1.0e-16 / dt] * 3
            + [0.0, 1.0e-8 / dt, 1.0e-8 / dt, 1.0e-8 / dt]
        )
    estimator = estimator_type(
        est_satellite,
        x_hat,
        dt=dt,
        unmodeled_dynamics_psd=process_psd,
    )
    controller = ADCS.controller.MTQ_Wisniewski(
        est_sat=est_satellite,
        lambda_s=np.diag([0.01, 0.01, 0.01]),
        lambda_q=np.diag([0.002, 0.002, 0.002]),
    )
    goal = ADCS.goals.Fixed_Attitude_Goal(q_ref=np.array([1.0, 0.0, 0.0, 0.0]))
    os0 = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(),
        J2000=0.22,
        R=np.array([5000.0, 0.0, 5000.0]),
        V=np.array([0.0, -7.5, 0.0]),
    )
    results = ADCS.simulate(
        x=x_0,
        satellite=satellite,
        est_satellite=est_satellite,
        estimator=estimator,
        controller=controller,
        goal=goal,
        os0=os0,
        dt=dt,
        tf=1000.0,
    )
    run_result = results.first()
    q_true = run_result.state_hist[-1].q
    q_est = run_result.est_state_hist[-1].q
    attitude_error_deg = np.rad2deg(
        2.0 * np.arccos(np.clip(abs(float(np.dot(q_true, q_est))), -1.0, 1.0))
    )
    print(f"{title}: final attitude error [deg]", attitude_error_deg)
    print(f"{title}: final estimated angular rate", run_result.est_state_hist[-1].w)
    print(f"{title}: maximum commanded actuator input", np.max(np.abs(run_result.control_hist)))
    ADCS.plot(
        results,
        ADCS.plots.AttitudePlot(sources=["real", "estimated", "reference"]),
        layout=(1, 1),
        title=f"{title}: Controlled Attitude",
    )
    ADCS.plot(
        results,
        ADCS.plots.QuaternionPlot(sources=["real", "estimated"]),
        ADCS.plots.AngularVelocityPlotCombined(sources=["real", "estimated"]),
        ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
        ADCS.plots.SensorsPlot(title="Sensor Readings", sources=["real", "clean"]),
        layout=(2, 2),
        title=f"{title}: Estimator with Controller",
    )
    plot_error_and_sun(
        run_result.time_s,
        run_result.state_hist,
        run_result.est_state_hist,
        run_result.os_hist,
    )
    plt.show()
