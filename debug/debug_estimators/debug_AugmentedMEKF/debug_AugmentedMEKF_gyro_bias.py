"""End-to-end AugmentedMEKF simulation with estimable gyro biases."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
import ADCS as ADCS
from ADCS.helpers.plotting.plot_estimator import plot_error_and_sun


class SimulationAugmentedMEKF(ADCS.AugmentedMEKF):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._previous_orbital_state = None

    def update(self, u, sensors, os):
        if self._previous_orbital_state is None:
            self._previous_orbital_state = os
            return self.correct(sensors, os)
        start = self._previous_orbital_state
        self._previous_orbital_state = os
        return self.step(u, sensors, start, os, midpoint_orbital_state=os)


def _sensors(mtm_noise, gyro_noise, sun_noise, gyro_bias, *, estimate_bias):
    sensors = [ADCS.MTM(axis, noise=mtm_noise.copy()) for axis in np.eye(3)]
    sensors += [
        ADCS.Gyro(
            axis,
            noise=gyro_noise.copy(),
            bias=ADCS.Bias(bias=value, std_bias=0.0 if not estimate_bias else 1.0e-10),
            estimate_bias=estimate_bias,
        )
        for axis, value in zip(np.eye(3), gyro_bias)
    ]
    sensors += [ADCS.SunPair(axis, efficiency=0.3, noise=sun_noise.copy()) for axis in np.eye(3)]
    return sensors


def main() -> None:
    np.random.seed(7)
    dt = 20.0
    gyro_bias = np.array([6.0e-4, -4.0e-4, 5.0e-4])
    mtm_noise = ADCS.Noise(std_noise=5.0e-8)
    gyro_noise = ADCS.Noise(std_noise=5.0e-7)
    sun_noise = ADCS.Noise(std_noise=1.0e-3)

    satellite = ADCS.Satellite(
        mass=3000.0, J_0=np.diag([500.0, 1500.0, 1500.0]),
        sensors=_sensors(mtm_noise, gyro_noise, sun_noise, gyro_bias, estimate_bias=False),
        disturbances=[ADCS.disturbances.GG_Disturbance()],
    )
    est_satellite = ADCS.EstimatedSatellite(
        mass=3200.0, J_0=np.diag([450.0, 1400.0, 1400.0]),
        sensors=_sensors(mtm_noise, gyro_noise, sun_noise, np.zeros(3), estimate_bias=True),
        disturbances=[ADCS.disturbances.GG_Disturbance()],
    )
    assert est_satellite.att_sens_bias_len == 3
    assert est_satellite.act_bias_len == 0
    assert est_satellite.dist_param_len == 0

    x_0 = ADCS.State.from_array(np.array([0.001, 0.001, -0.002, 0.2588, 0.0, 0.9659, 0.0]))
    x_hat = ADCS.EstimatorState(
        # The MEKF tracks about a coarse attitude solution. Its local tangent
        # error is not valid for the former 150 deg initial attitude mismatch.
        w=[0.0012, 0.0008, -0.0018],
        q=[np.cos(np.deg2rad(72.5)), 0.0, np.sin(np.deg2rad(72.5)), 0.0],
        sens_bias=np.zeros(3),
        cov=block_diag(np.eye(3) * 0.01**2, np.eye(3) * 0.15**2, np.eye(3) * 0.05**2),
        int_cov=block_diag(np.eye(3) * 1.0e-16, np.eye(3) * 1.0e-8, np.eye(3) * 1.0e-12),
    )
    # Match the legacy UAKF example: Q_w=1e-16 and Q_attitude=1e-8 per
    # timestep. The new filters take continuous PSDs, so divide by dt.
    estimator = SimulationAugmentedMEKF(
        est_satellite,
        x_hat,
        dt=dt,
        unmodeled_dynamics_psd=np.array(
            [1.0e-16 / dt] * 3 + [1.0e-8 / dt] * 3
        ),
    )
    os0 = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(), J2000=0.22,
        R=np.array([5000.0, 0.0, 5000.0]), V=np.array([0.0, -7.5, 0.0]),
    )
    results = ADCS.simulate(
        x=x_0, satellite=satellite, est_satellite=est_satellite,
        estimator=estimator, os0=os0, dt=dt, tf=2000.0,
    )
    run = results.first()
    estimated_bias = np.vstack([state.sens_bias for state in run.est_state_hist])
    print("true gyro bias:", gyro_bias)
    print("final estimated gyro bias:", estimated_bias[-1])
    np.testing.assert_allclose(estimated_bias[-1], gyro_bias, atol=1.0e-4)

    ADCS.plot(results, ADCS.plots.AttitudePlot(sources=["real", "estimated"]),
              layout=(1, 1), title="Augmented MEKF: Gyro-Bias Estimation")
    ADCS.plot(
        results,
        ADCS.plots.QuaternionPlot(sources=["real", "estimated"]),
        ADCS.plots.AngularVelocityPlotCombined(sources=["real", "estimated"]),
        ADCS.plots.SensorsPlot(title="Sensor Readings", sources=["real", "clean"]),
        ADCS.plots.BiasPlot(
            kind="sensor",
            sources=["real", "estimated"],
            title="Estimated Gyro Biases",
        ),
        ADCS.plots.IlluminationPlot(), layout=(3, 3),
        title="Augmented MEKF: Gyro Bias",
    )
    plot_error_and_sun(run.time_s, run.state_hist, run.est_state_hist, run.os_hist)
    plt.show()


if __name__ == "__main__":
    main()
