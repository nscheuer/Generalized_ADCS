import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag

import ADCS


def main():
    np.random.seed(7)
    dt = 10.0
    gyro_bias = np.array([6.0e-4, -4.0e-4, 5.0e-4])
    mtm_noise = ADCS.Noise(std_noise=5.0e-8)
    gyro_noise = ADCS.Noise(std_noise=5.0e-7)
    sun_noise = ADCS.Noise(std_noise=1.0e-3)

    def sensors(bias, estimate_bias):
        result = [ADCS.MTM(axis, noise=mtm_noise.copy()) for axis in np.eye(3)]
        result += [
            ADCS.Gyro(
                axis,
                noise=gyro_noise.copy(),
                bias=ADCS.Bias(value, 1.0e-10 if estimate_bias else 0.0),
                estimate_bias=estimate_bias,
            )
            for axis, value in zip(np.eye(3), bias)
        ]
        result += [ADCS.SunPair(axis, efficiency=0.3, noise=sun_noise.copy()) for axis in np.eye(3)]
        return result

    faces = [
        ADCS.disturbances.GeometryFace(0.03, axis * 0.05, axis, eta_a=0.7, CD=2.2)
        for axis in np.eye(3)
    ]
    faces += [
        ADCS.disturbances.GeometryFace(0.03, -axis * 0.05, -axis, eta_a=0.7, CD=2.2)
        for axis in np.eye(3)
    ]
    geometry = ADCS.disturbances.GeometryConfig(faces)

    satellite = ADCS.Satellite(
        mass=4.0,
        J_0=np.diag([3.4, 2.9, 1.3]),
        sensors=sensors(gyro_bias, False),
        disturbances=[
            ADCS.disturbances.Dipole_Disturbance([0.4, -0.3, 0.2]),
            ADCS.disturbances.GG_Disturbance(),
            ADCS.disturbances.SRP_Disturbance(geometry),
            ADCS.disturbances.Drag_Disturbance(geometry),
        ],
    )
    lumped_disturbance = ADCS.disturbances.Torque_Disturbance(np.zeros(3), estimate_dist=True)
    lumped_disturbance.parameter_std_rate = np.full(3, 1.0e-5)
    estimated = ADCS.EstimatedSatellite(
        mass=4.2,
        J_0=np.diag([3.5, 3.0, 1.4]),
        sensors=sensors(np.zeros(3), True),
        disturbances=[lumped_disturbance],
    )
    assert estimated.att_sens_bias_len == 3
    assert estimated.dist_param_len == 3

    x = ADCS.State.from_array([0.0012, 0.0008, -0.0018, 0.2588, 0.0, 0.9659, 0.0])
    x_hat = ADCS.EstimatorState(
        w=[0.0012, 0.0008, -0.0018],
        q=[np.cos(np.deg2rad(72.5)), 0.0, np.sin(np.deg2rad(72.5)), 0.0],
        sens_bias=np.zeros(3),
        dist_param=np.zeros(3),
        cov=block_diag(
            np.eye(3) * 0.01**2,
            np.eye(3) * 0.15**2,
            np.eye(3) * 0.05**2,
            np.eye(3) * 0.5**2,
        ),
        int_cov=block_diag(
            np.eye(3) * 1.0e-16,
            np.eye(3) * 1.0e-8,
            np.eye(3) * 1.0e-12,
            np.eye(3) * 1.0e-10,
        ),
    )
    estimator = ADCS.AugmentedMEKF(
        estimated,
        x_hat,
        dt=dt,
        unmodeled_dynamics_psd=np.array([1.0e-16 / dt] * 3 + [1.0e-8 / dt] * 3),
    )
    os0 = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(), J2000=0.22,
        R=[5000.0, 0.0, 5000.0], V=[0.0, -7.5, 0.0],
    )
    results = ADCS.simulate(
        x=x,
        satellite=satellite,
        est_satellite=estimated,
        estimator=estimator,
        os0=os0,
        dt=dt,
        tf=2000.0,
    )
    run = results.first()
    truth = run.state_hist[-1]
    estimate = run.est_state_hist[-1]
    assert np.isfinite(estimate.as_estimator_array()).all()
    assert abs(np.dot(truth.q, estimate.q)) > 0.98
    assert np.linalg.norm(truth.w - estimate.w) < 1.0e-3
    np.testing.assert_allclose(estimate.sens_bias, gyro_bias, atol=1.0e-4)
    print("final gyro bias:", estimate.sens_bias)
    print("final lumped disturbance torque:", estimate.dist_param)

    ADCS.plot(results, ADCS.plots.AttitudePlot(sources=["real", "estimated"]), layout=(1, 1), title="Advanced MEKF")
    ADCS.plot(
        results,
        ADCS.plots.QuaternionPlot(sources=["real", "estimated"]),
        ADCS.plots.AngularVelocityPlotCombined(sources=["real", "estimated"]),
        ADCS.plots.SensorsPlot(title="Sensor Readings", sources=["real", "clean"]),
        ADCS.plots.BiasPlot(kind="sensor", sources=["real", "estimated"], title="Estimated Gyro Biases"),
        ADCS.plots.DisturbanceParameterPlot(
            labels=[r"$\tau_x$", r"$\tau_y$", r"$\tau_z$"], units="N m", plot_torque=True,
        ),
        ADCS.plots.IlluminationPlot(),
        layout=(2, 3),
        title="Advanced MEKF: Gyro Bias and Lumped Disturbance",
    )
    plt.show()


if __name__ == "__main__":
    main()
