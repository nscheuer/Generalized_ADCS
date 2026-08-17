import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import ADCS as ADCS


def main() -> None:
    dt = 10.0

    mtm_noise = ADCS.Noise(std_noise=5.0e-8)
    gyro_noise = ADCS.Noise(std_noise=5.0e-7)
    sun_noise = ADCS.Noise(std_noise=1.0e-3)
    rw_torque_noise = ADCS.Noise(std_noise=1.0e-5)
    rw_meas_noise = ADCS.Noise(std_noise=1.0e-5)

    actuators = [ADCS.MTQ(axis=axis, max_torque=0.1, noise=ADCS.Noise(std_noise=1.0e-5)) for axis in np.eye(3)]
    actuators += [
        ADCS.RW(
            axis=axis,
            max_torque=0.2,
            J=0.02,
            h=0.8,
            h_max=4.0,
            noise=rw_torque_noise.copy(),
            h_meas_noise=rw_meas_noise.copy(),
        )
        for axis in np.eye(3)
    ]

    real_sensors = [ADCS.MTM(axis, noise=mtm_noise.copy()) for axis in np.eye(3)]
    real_sensors += [ADCS.Gyro(axis, noise=gyro_noise.copy()) for axis in np.eye(3)]
    real_sensors += [ADCS.SunPair(axis, efficiency=0.3, noise=sun_noise.copy()) for axis in np.eye(3)]

    satellite = ADCS.Satellite(
        mass=4.0,
        J_0=np.diag([3.4, 2.9, 1.3]),
        actuators=actuators,
        sensors=real_sensors,
        disturbances=[ADCS.disturbances.GG_Disturbance()],
    )
    x_0 = ADCS.State.from_array(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.8, 0.8, 0.8]))

    est_actuators = [ADCS.MTQ(axis=axis, max_torque=0.1, noise=ADCS.Noise(std_noise=1.0e-5)) for axis in np.eye(3)]
    est_actuators += [
        ADCS.RW(
            axis=axis,
            max_torque=0.2,
            J=0.02,
            h=0.0,
            h_max=4.0,
            noise=rw_torque_noise.copy(),
            h_meas_noise=rw_meas_noise.copy(),
        )
        for axis in np.eye(3)
    ]

    est_sensors = [ADCS.MTM(axis, noise=mtm_noise.copy()) for axis in np.eye(3)]
    est_sensors += [ADCS.Gyro(axis, noise=gyro_noise.copy()) for axis in np.eye(3)]
    est_sensors += [ADCS.SunPair(axis, efficiency=0.3, noise=sun_noise.copy()) for axis in np.eye(3)]

    est_satellite = ADCS.EstimatedSatellite(
        mass=4.0,
        J_0=np.diag([3.4, 2.9, 1.3]),
        actuators=est_actuators,
        sensors=est_sensors,
        disturbances=[ADCS.disturbances.GG_Disturbance()],
    )
    x_hat = ADCS.EstimatorState(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0], h=np.zeros(3))

    P_hat = block_diag(
        np.eye(3) * (0.01) ** 2,
        np.eye(3),
        np.eye(3) * (0.2) ** 2,
    )
    Q_hat = block_diag(
        np.eye(3) * (1.0e-8) ** 2,
        1.0e-8 * np.eye(3),
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
        tf=600.0,
    )

    ADCS.plot(
        results,
        ADCS.plots.AttitudePlot(sources=["real", "estimated"]),
        layout=(1, 1),
        title="UKF Estimator: Reaction Wheel Momentum States",
    )

    ADCS.plot(
        results,
        ADCS.plots.QuaternionPlot(sources=["real", "estimated"]),
        ADCS.plots.AngularVelocityPlotCombined(sources=["real", "estimated"]),
        ADCS.plots.SensorsPlot(title="Sensor and RW Readings", sources=["real", "clean"]),
        ADCS.plots.IlluminationPlot(),
        layout=(2, 2),
        title="UKF Estimator: Reaction Wheel Momentum States",
    )

    plt.show()


if __name__ == "__main__":
    main()
