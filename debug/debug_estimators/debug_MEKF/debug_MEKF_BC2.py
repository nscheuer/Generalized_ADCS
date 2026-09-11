"""Run a BeaverCube-2 attitude-estimation scenario with the multiplicative EKF."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
import ADCS as ADCS
from ADCS.helpers.plotting.plot_estimator import plot_error_and_sun
from ADCS.orbits.universal_constants import TimeConstants


class SimulationMEKF(ADCS.MEKF):
    """Compatibility shim for the legacy simulation estimator interface."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._previous_orbital_state = None

    def update(self, u: np.ndarray, sensors: np.ndarray, os: ADCS.Orbital_State) -> ADCS.EstimatorState:
        if self._previous_orbital_state is None:
            self._previous_orbital_state = os
            return self.correct(sensors, os)
        os_start = self._previous_orbital_state
        self._previous_orbital_state = os
        return self.step(u, sensors, os_start, os, midpoint_orbital_state=os)


def main() -> None:
    np.random.seed(37)
    dt = 20.0
    tf = 1000.0

    rw_h0 = -9.76622366e-05
    satellite = ADCS.satellite_factory.create_beavercube2_cubesat(
        estimated=False,
        include_biases=False,
    )
    satellite.rw_actuators[0].h = rw_h0
    est_satellite = ADCS.EstimatedSatellite.from_satellite(satellite)

    x_0 = ADCS.State(
        w=np.array([5.0e-4, -3.0e-4, 2.0e-4]),
        q=np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]),
        h=np.array([rw_h0]),
    ).normalized()

    x_hat = ADCS.EstimatorState(
        w=np.zeros(3),
        q=[1.0, 0.0, 0.0, 0.0],
        h=np.array([rw_h0]),
        cov=block_diag(
            np.eye(3) * np.deg2rad(1.0) ** 2,
            np.eye(3) * np.deg2rad(45.0) ** 2,
            np.eye(1) * 1.0e-8,
        ),
        int_cov=block_diag(
            np.eye(3) * 1.0e-9,
            np.eye(3) * 1.0e-8,
            np.eye(1) * 1.0e-14,
        ),
    )

    estimator = SimulationMEKF(
        est_satellite,
        x_hat,
        dt=dt,
        unmodeled_dynamics_psd=1.0e-9,
    )

    os0 = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(),
        J2000=0.22 - TimeConstants.sec2cent,
        R=7000.0 * np.array([0.0, np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0]),
        V=np.array([8.0, 0.0, 0.0]),
    )

    results = ADCS.simulate(
        x=x_0,
        satellite=satellite,
        est_satellite=est_satellite,
        estimator=estimator,
        os0=os0,
        dt=dt,
        tf=tf,
    )

    ADCS.plot(
        results,
        ADCS.plots.AttitudePlot(sources=["real", "estimated"]),
        layout=(1, 1),
        title="MEKF Estimator: BeaverCube-2",
    )

    ADCS.plot(
        results,
        ADCS.plots.QuaternionPlot(sources=["real", "estimated"]),
        ADCS.plots.AngularVelocityPlotCombined(sources=["real", "estimated"]),
        ADCS.plots.SensorsPlot(title="Sensor Readings", sources=["real", "clean"]),
        ADCS.plots.IlluminationPlot(),
        layout=(2, 2),
        title="MEKF Estimator: BeaverCube-2",
    )

    run = results.first()
    plot_error_and_sun(run.time_s, run.state_hist, run.est_state_hist, run.os_hist)

    plt.show()


if __name__ == "__main__":
    main()
