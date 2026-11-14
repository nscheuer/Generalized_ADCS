"""Estimate attitude, gyro bias, and residual magnetic dipole with an MEKF."""

import numpy as np

import ADCS


np.random.seed(17)
dt = 10.0
axes = np.vstack((np.eye(3), -np.eye(3)))
gyro_bias = [0.002, -0.0015, 0.001]
residual_dipole = np.array([0.4, -0.3, 0.2])  # A m^2


def make_sensors(bias_values, estimate_bias=False):
    sun = [
        ADCS.SunSensor(axis=axis, efficiency=1.0, sample_time=dt,
                       noise=ADCS.Noise(std_noise=0.03))
        for axis in axes
    ]
    gyros = [
        ADCS.Gyro(axis=axis, sample_time=dt,
                  noise=ADCS.Noise(std_noise=0.001),
                  bias=ADCS.Bias(bias=bias), estimate_bias=estimate_bias)
        for axis, bias in zip(np.eye(3), bias_values)
    ]
    magnetic = [
        ADCS.MTM(axis=axis, sample_time=dt,
                 noise=ADCS.Noise(std_noise=2e-6))
        for axis in np.eye(3)
    ]
    return sun + gyros + magnetic


satellite = ADCS.Satellite(
    mass=4.0, J_0=np.eye(3) * 0.02,
    sensors=make_sensors(gyro_bias),
    disturbances=[ADCS.disturbances.Dipole_Disturbance(residual_dipole)],
)
estimated_dipole = ADCS.disturbances.Dipole_Disturbance(
    np.zeros(3), estimate_dist=True,
)
estimated_dipole.parameter_std_rate = np.full(3, 1e-5)
estimated_satellite = ADCS.EstimatedSatellite(
    mass=4.0, J_0=np.eye(3) * 0.02,
    sensors=make_sensors([0.0, 0.0, 0.0], estimate_bias=True),
    disturbances=[estimated_dipole],
)
orbit = ADCS.Orbital_State(
    ephem=ADCS.Ephemeris(), J2000=0.22,
    R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]),
)
initial = ADCS.EstimatorState(
    w=np.array([0.01, -0.008, 0.012]),
    q=np.array([0.98, 0.1, 0.1, 0.1]),
    sens_bias=np.zeros(3),
    dist_param=np.zeros(3),
    cov=np.diag([0.003**2] * 3 + [0.2**2] * 3
                + [0.004**2] * 3 + [0.5**2] * 3),
    int_cov=np.diag([1e-8] * 3 + [1e-6] * 3
                    + [1e-9] * 3 + [1e-10] * 3),
)
estimator = ADCS.AugmentedMEKF(estimated_satellite, initial, dt=dt)
results = ADCS.simulate(
    x=ADCS.State(w=np.array([0.01, -0.008, 0.012]),
                 q=np.array([1.0, 0.0, 0.0, 0.0])),
    satellite=satellite, est_satellite=estimated_satellite,
    estimator=estimator, os0=orbit, dt=dt, tf=600.0,
)
run = results.first()
run.target_hist = np.asarray([state.q for state in run.state_hist])
run.boresight_hist = np.tile([1.0, 0.0, 0.0], (len(run.time_s), 1))
# Both histories use the 12-sensor layout; the three gyro biases occupy
# sensor slots 6 through 8.
run.sensor_bias = [snapshot[6:9] for snapshot in run.sensor_bias]
run.est_sensor_bias = [snapshot[6:9] for snapshot in run.est_sensor_bias]
ADCS.plot(
    results,
    ADCS.plots.TargetPlot(modes=["est_target"], title="Attitude angle error"),
    ADCS.plots.BiasPlot(kind="sensor", sources=["real", "estimated"],
                        labels=["Gyro X", "Gyro Y", "Gyro Z"],
                        units="rad/s", title="Gyro bias"),
    ADCS.plots.DisturbanceParameterPlot(
        labels=["Dipole X", "Dipole Y", "Dipole Z"],
        units="A m²", title="Residual dipole"),
    layout=(3, 1), figsize=(10, 12),
    title="MEKF with gyro bias and residual dipole estimates",
)
