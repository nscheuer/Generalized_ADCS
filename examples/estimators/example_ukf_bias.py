import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import ADCS as ADCS


def main() -> None:
    dt = 20.0
    true_gyro_bias = np.array([6.0e-4, -4.0e-4, 5.0e-4])
    true_mtm_bias = np.zeros(3)
    true_sun_bias = np.zeros(3)

    mtm_noise = ADCS.Noise(std_noise=5.0e-8)
    gyro_noise = ADCS.Noise(std_noise=5.0e-7)
    sun_noise = ADCS.Noise(std_noise=1.0e-3)

    real_sensors = [
        ADCS.MTM(
            axis,
            noise=mtm_noise.copy(),
            bias=ADCS.Bias(bias=true_mtm_bias[index], std_bias=0.0),
        )
        for index, axis in enumerate(np.eye(3))
    ]
    real_sensors += [
        ADCS.Gyro(
            axis,
            noise=gyro_noise.copy(),
            bias=ADCS.Bias(bias=true_gyro_bias[index], std_bias=0.0),
        )
        for index, axis in enumerate(np.eye(3))
    ]
    real_sensors += [
        ADCS.SunPair(
            axis,
            efficiency=0.3,
            noise=sun_noise.copy(),
            bias=ADCS.Bias(bias=true_sun_bias[index], std_bias=0.0),
        )
        for index, axis in enumerate(np.eye(3))
    ]

    satellite = ADCS.Satellite(
        mass=3000.0,
        J_0=np.diag([500.0, 1500.0, 1500.0]),
        sensors=real_sensors,
        disturbances=[ADCS.disturbances.GG_Disturbance()],
    )
    x_0 = np.array([0.001, 0.001, -0.002, 0.2588, 0.0, 0.9659, 0.0])

    est_sensors = [
        ADCS.MTM(
            axis,
            noise=mtm_noise.copy(),
            bias=ADCS.Bias(bias=0.0, std_bias=1.0e-10),
            estimate_bias=True,
        )
        for axis in np.eye(3)
    ]
    est_sensors += [
        ADCS.Gyro(
            axis,
            noise=gyro_noise.copy(),
            bias=ADCS.Bias(bias=0.0, std_bias=1.0e-10),
            estimate_bias=True,
        )
        for axis in np.eye(3)
    ]
    est_sensors += [
        ADCS.SunPair(
            axis,
            efficiency=0.3,
            noise=sun_noise.copy(),
            bias=ADCS.Bias(bias=0.0, std_bias=1.0e-6),
            estimate_bias=True,
        )
        for axis in np.eye(3)
    ]

    est_satellite = ADCS.EstimatedSatellite(
        mass=3200.0,
        J_0=np.diag([450.0, 1400.0, 1400.0]),
        sensors=est_sensors,
        disturbances=[ADCS.disturbances.GG_Disturbance()],
    )
    x_hat = np.concatenate([np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), np.zeros(9)])

    P_hat = block_diag(
        np.eye(3) * (0.01) ** 2,
        np.eye(3),
        np.eye(9) * (0.05) ** 2,
    )
    Q_hat = block_diag(
        np.eye(3) * (1.0e-8) ** 2,
        1.0e-8 * np.eye(3),
        np.eye(3) * (1.0e-10) ** 2 * dt,
        np.eye(3) * (1.0e-10) ** 2 * dt,
        np.eye(3) * (1.0e-6) ** 2 * dt,
    )

    estimator = ADCS.UAKF(
        J2000=0.22,
        est_sat=est_satellite,
        x_hat=x_hat,
        P_hat=P_hat,
        Q_hat=Q_hat,
        dt=dt,
        cross_term=True,
        quat_as_vec=False,
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
        tf=2500.0,
    )

    ADCS.plot(
        results,
        ADCS.plots.AttitudePlot(sources=["real", "estimated"]),
        layout=(1, 1),
        title="UKF Estimator: Gyro Bias Estimation",
    )

    ADCS.plot(
        results,
        ADCS.plots.QuaternionPlot(sources=["real", "estimated"]),
        ADCS.plots.AngularVelocityPlotCombined(sources=["real", "estimated"]),
        ADCS.plots.SensorsPlot(title="Sensor Readings", sources=["real", "clean"]),
        ADCS.plots.BiasPlot(
            sources=["real", "estimated"],
            labels=[
                "MTM x",
                "MTM y",
                "MTM z",
                "Gyro x",
                "Gyro y",
                "Gyro z",
                "Sun x",
                "Sun y",
                "Sun z",
            ],
        ),
        ADCS.plots.IlluminationPlot(),
        layout=(2, 3),
        title="UKF Estimator: Sensor Bias Estimation",
    )

    plt.show()


if __name__ == "__main__":
    main()
