"""Run the basic attitude-estimation scenario with the non-augmented SRUKF."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
import ADCS as ADCS
from ADCS.helpers.plotting.plot_estimator import plot_error_and_sun


def main() -> None:
    dt = 20.0
    mtm_noise = ADCS.Noise(std_noise=5.0e-8)
    gyro_noise = ADCS.Noise(std_noise=5.0e-7)
    sun_noise = ADCS.Noise(std_noise=1.0e-3)

    def sensors() -> list:
        result = [ADCS.MTM(axis, noise=mtm_noise.copy()) for axis in np.eye(3)]
        result += [ADCS.Gyro(axis, noise=gyro_noise.copy()) for axis in np.eye(3)]
        result += [
            ADCS.SunPair(axis, efficiency=0.3, noise=sun_noise.copy())
            for axis in np.eye(3)
        ]
        return result

    satellite = ADCS.Satellite(
        mass=3000.0,
        J_0=np.diag([500.0, 1500.0, 1500.0]),
        sensors=sensors(),
        disturbances=[ADCS.disturbances.GG_Disturbance()],
    )
    est_satellite = ADCS.EstimatedSatellite(
        mass=3200.0,
        J_0=np.diag([450.0, 1400.0, 1400.0]),
        sensors=sensors(),
        disturbances=[ADCS.disturbances.GG_Disturbance()],
    )
    x_0 = ADCS.State.from_array(
        np.array([0.001, 0.001, -0.002, 0.2588, 0.0, 0.9659, 0.0])
    )
    covariance = ADCS.Covariance(
        block_diag(np.eye(3) * 0.01**2, np.eye(3)),
        form="sqrt",
        coordinates="state_tangent",
        psd_policy="allow_indefinite",
    )
    x_hat = ADCS.EstimatorState(
        w=np.zeros(3),
        q=[1.0, 0.0, 0.0, 0.0],
        covariance=covariance,
        process_noise=ADCS.Covariance.zeros(
            6,
            form="sqrt",
            coordinates="state_tangent",
            psd_policy="allow_indefinite",
        ),
    )
    estimator = ADCS.SRUKF(
        est_satellite,
        x_hat,
        dt=dt,
        unmodeled_dynamics_psd=np.concatenate(
            (np.full(3, 1.0e-16 / dt), np.full(3, 1.0e-8 / dt))
        ),
    )
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
        os0=os0,
        dt=dt,
        tf=2000.0,
    )
    ADCS.plot(
        results,
        ADCS.plots.AttitudePlot(sources=["real", "estimated"]),
        layout=(1, 1),
        title="SRUKF Estimator: Gyro + MTM + SunPair",
    )
    ADCS.plot(
        results,
        ADCS.plots.QuaternionPlot(sources=["real", "estimated"]),
        ADCS.plots.AngularVelocityPlotCombined(sources=["real", "estimated"]),
        ADCS.plots.SensorsPlot(title="Sensor Readings", sources=["real", "clean"]),
        ADCS.plots.IlluminationPlot(),
        layout=(2, 2),
        title="SRUKF Estimator: Gyro + MTM + SunPair",
    )
    run = results.first()
    plot_error_and_sun(run.time_s, run.state_hist, run.est_state_hist, run.os_hist)
    plt.show()


if __name__ == "__main__":
    main()
