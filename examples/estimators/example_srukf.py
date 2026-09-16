import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag

import ADCS


def main():
    dt = 20.0
    mtm_noise = ADCS.Noise(std_noise=5.0e-8)
    gyro_noise = ADCS.Noise(std_noise=5.0e-7)
    sun_noise = ADCS.Noise(std_noise=1.0e-3)

    def sensors():
        result = [ADCS.MTM(axis, noise=mtm_noise.copy()) for axis in np.eye(3)]
        result += [ADCS.Gyro(axis, noise=gyro_noise.copy()) for axis in np.eye(3)]
        result += [ADCS.SunPair(axis, efficiency=0.3, noise=sun_noise.copy()) for axis in np.eye(3)]
        return result

    satellite = ADCS.Satellite(
        mass=3000.0, J_0=np.diag([500.0, 1500.0, 1500.0]),
        sensors=sensors(), disturbances=[ADCS.disturbances.GG_Disturbance()],
    )
    estimated = ADCS.EstimatedSatellite(
        mass=3200.0, J_0=np.diag([450.0, 1400.0, 1400.0]),
        sensors=sensors(), disturbances=[ADCS.disturbances.GG_Disturbance()],
    )
    x = ADCS.State.from_array([0.001, 0.001, -0.002, 0.2588, 0.0, 0.9659, 0.0])
    covariance = ADCS.Covariance(
        block_diag(np.eye(3) * 0.01**2, np.eye(3)),
        form="sqrt", coordinates="state_tangent", psd_policy="allow_indefinite",
    )
    x_hat = ADCS.EstimatorState(
        w=[0.0012, 0.0008, -0.0018],
        q=[np.cos(np.deg2rad(72.5)), 0.0, np.sin(np.deg2rad(72.5)), 0.0],
        covariance=covariance,
        process_noise=ADCS.Covariance.zeros(
            6, form="sqrt", coordinates="state_tangent", psd_policy="allow_indefinite",
        ),
    )
    estimator = ADCS.SRUKF(
        estimated, x_hat, dt=dt,
        unmodeled_dynamics_psd=np.array([1.0e-16 / dt] * 3 + [1.0e-8 / dt] * 3),
    )
    os0 = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(), J2000=0.22,
        R=[5000.0, 0.0, 5000.0], V=[0.0, -7.5, 0.0],
    )
    results = ADCS.simulate(
        x=x, satellite=satellite, est_satellite=estimated,
        estimator=estimator, os0=os0, dt=dt, tf=2000.0,
    )
    run = results.first()
    assert abs(np.dot(run.state_hist[-1].q, run.est_state_hist[-1].q)) > 0.98
    assert np.linalg.norm(run.state_hist[-1].w - run.est_state_hist[-1].w) < 1.0e-3
    ADCS.plot(results, ADCS.plots.AttitudePlot(sources=["real", "estimated"]), layout=(1, 1), title="SRUKF")
    ADCS.plot(
        results,
        ADCS.plots.QuaternionPlot(sources=["real", "estimated"]),
        ADCS.plots.AngularVelocityPlotCombined(sources=["real", "estimated"]),
        ADCS.plots.SensorsPlot(title="Sensor Readings", sources=["real", "clean"]),
        ADCS.plots.IlluminationPlot(), layout=(2, 2), title="SRUKF",
    )
    plt.show()


if __name__ == "__main__":
    main()

