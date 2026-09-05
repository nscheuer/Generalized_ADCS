"""Run the basic attitude-estimation scenario with the additive EKF."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
import ADCS as ADCS
from ADCS.helpers.plotting.plot_estimator import plot_error_and_sun


class SimulationEKF(ADCS.EKF):
    """Compatibility shim for the legacy simulation estimator interface."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._previous_orbital_state = None

    def update(self, u: np.ndarray, sensors: np.ndarray, os: ADCS.Orbital_State) -> ADCS.EstimatorState:
        # First call: no elapsed time yet, so correct in place rather than
        # predicting a full dt from ``os`` to ``os`` (which left the estimate
        # permanently one step ahead of every measurement).
        if self._previous_orbital_state is None:
            self._previous_orbital_state = os
            return self.correct(sensors, os)
        os_start = self._previous_orbital_state
        self._previous_orbital_state = os
        return self.step(u, sensors, os_start, os, midpoint_orbital_state=os)


def main() -> None:
    dt = 20.0

    mtm_noise = ADCS.Noise(std_noise=5.0e-8)
    gyro_noise = ADCS.Noise(std_noise=5.0e-7)
    sun_noise = ADCS.Noise(std_noise=1.0e-3)

    real_sensors = [ADCS.MTM(axis, noise=mtm_noise.copy()) for axis in np.eye(3)]
    real_sensors += [ADCS.Gyro(axis, noise=gyro_noise.copy()) for axis in np.eye(3)]
    real_sensors += [ADCS.SunPair(axis, efficiency=0.3, noise=sun_noise.copy()) for axis in np.eye(3)]

    satellite = ADCS.Satellite(
        mass=3000.0,
        J_0=np.diag([500.0, 1500.0, 1500.0]),
        sensors=real_sensors,
        disturbances=[ADCS.disturbances.GG_Disturbance()],
    )
    x_0 = ADCS.State.from_array(np.array([0.001, 0.001, -0.002, 0.2588, 0.0, 0.9659, 0.0]))

    est_sensors = [ADCS.MTM(axis, noise=mtm_noise.copy()) for axis in np.eye(3)]
    est_sensors += [ADCS.Gyro(axis, noise=gyro_noise.copy()) for axis in np.eye(3)]
    est_sensors += [ADCS.SunPair(axis, efficiency=0.3, noise=sun_noise.copy()) for axis in np.eye(3)]

    est_satellite = ADCS.EstimatedSatellite(
        mass=3200.0,
        J_0=np.diag([450.0, 1400.0, 1400.0]),
        sensors=est_sensors,
        disturbances=[ADCS.disturbances.GG_Disturbance()],
    )

    x_hat = ADCS.EstimatorState(
        w=np.zeros(3),
        q=[1.0, 0.0, 0.0, 0.0],
        cov=np.diag([0.01**2, 0.01**2, 0.01**2, 0.0, 1.0, 1.0, 1.0]),
        int_cov=np.diag([1.0e-16, 1.0e-16, 1.0e-16, 0.0, 1.0e-8, 1.0e-8, 1.0e-8]),
    )

    estimator = SimulationEKF(
        est_satellite,
        x_hat,
        dt=dt,
        unmodeled_dynamics_psd=1.0e-16,
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
        title="EKF Estimator: Gyro + MTM + SunPair",
    )

    ADCS.plot(
        results,
        ADCS.plots.QuaternionPlot(sources=["real", "estimated"]),
        ADCS.plots.AngularVelocityPlotCombined(sources=["real", "estimated"]),
        ADCS.plots.SensorsPlot(title="Sensor Readings", sources=["real", "clean"]),
        ADCS.plots.IlluminationPlot(),
        layout=(2, 2),
        title="EKF Estimator: Gyro + MTM + SunPair",
    )

    run = results.first()
    plot_error_and_sun(run.time_s, run.state_hist, run.est_state_hist, run.os_hist)

    plt.show()


if __name__ == "__main__":
    main()
