import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import ADCS as ADCS
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

dt = 20.0

# Real Satellite
gyro_noise = ADCS.Noise(std_noise=3.1623e-7)
sens = [ADCS.Gyro(axis, noise=gyro_noise) for axis in np.eye(3)]

mtm_noise = ADCS.Noise(std_noise=5e-8)
sens += [ADCS.MTM(axis, noise=mtm_noise) for axis in np.eye(3)]

sun_noise = ADCS.Noise(std_noise=2e-3)
sens += [ADCS.SunPair(axis, efficiency=0.3,noise=sun_noise) for axis in np.eye(3)]

gg_dist = [ADCS.disturbances.GG_Disturbance()]
satellite = ADCS.Satellite(mass=3000, J_0=np.diag([500, 1500, 1500]), sensors=sens, disturbances=gg_dist)
x_0 = ADCS.State.from_array(np.array([0.001, 0.001, -0.002] + [0.2588, 0, 0.9659, 0])) # w, q

# Estimated Satellite
est_gyro_noise = ADCS.Noise(std_noise=5e-7)
est_sens = [ADCS.Gyro(axis, noise=est_gyro_noise) for axis in np.eye(3)]

est_mtm_noise = ADCS.Noise(std_noise=5e-8)
est_sens += [ADCS.MTM(axis, noise=est_mtm_noise) for axis in np.eye(3)]

est_sun_noise = ADCS.Noise(std_noise=1e-3)
est_sens += [ADCS.SunPair(axis, efficiency=0.3, noise=est_sun_noise) for axis in np.eye(3)]

est_gg_dist = [ADCS.disturbances.GG_Disturbance()]
est_satellite = ADCS.EstimatedSatellite(mass=3200, J_0=np.diag([450, 1400, 1400]), sensors=est_sens, disturbances=est_gg_dist)
x_hat = ADCS.EstimatorState(w=np.zeros(3), q=[1, 0, 0, 0]) # w, q

# Estimator
P_hat = block_diag(
    np.eye(3)*(0.01)**2, # Angular Velocity Initial State Error
    np.eye(3), # Attitude (MRP) Initial State Error
    )
Q_hat = block_diag(
    np.eye(3)*(1e-8)**2.0, 
    1e-8*np.eye(3)
)
estimator = ADCS.SRUAKF(J2000=0.22, est_sat=est_satellite, x_hat=x_hat, P_hat=P_hat, Q_hat=Q_hat, dt=dt, cross_term=True, quat_as_vec=False)

os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(), J2000=0.22, R=np.array([5000, 0, 5000]), V=np.array([0, -7.5, 0]))

results = ADCS.simulate(
    x=x_0,
    satellite=satellite,
    est_satellite=est_satellite,
    estimator=estimator,
    os0=os0,
    dt=dt,
    tf=5000.0
)

ADCS.plot(
    results,
    ADCS.plots.AttitudePlot(sources=["real", "estimated"]),
    layout=(1,1),
    title="Estimator Convergence",
)

ADCS.plot(
    results,
    ADCS.plots.AngularVelocityPlotCombined(sources=["real", "estimated"]),
    ADCS.plots.TargetPlot(modes=["real_est"]),
    ADCS.plots.SensorsPlot(title="All Sensor Readings", sources=["real", "clean"]),
    ADCS.plots.IlluminationPlot(),
    layout=(2,2),
    title="Estimator Convergence",
)
plt.show()
