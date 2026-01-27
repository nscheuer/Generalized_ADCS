import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
import ADCS as ADCS
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

from ADCS.helpers.math_helpers import random_n_unit_vec
from ADCS.helpers.math_constants import MathConstants

dt = 1

# Sensors: Magnetometers
mtm_noise = ADCS.Noise(noise=0.0, std_noise=1e-8)
mtm_bias_mean = random_n_unit_vec(3)*np.random.uniform(1e-9, 1e-7)
mtm_bsr = 1e-9
mtm_bias = [ADCS.Bias(bias=mtm_bias_mean[j], std_bias=mtm_bsr) for j in range(3)]
mtms = [ADCS.MTM(axis=MathConstants.unitvecs[j], bias=mtm_bias[j], noise=mtm_noise.copy()) for j in range(3)]

# Sensors: Gyroscopes
gyro_noise = ADCS.Noise(noise=0.0, std_noise=0.0001)
gyro_bias_mean = np.array([0.002, 0.002, 0.002])
gyro_bsr = 0.0004*np.pi/180.0
gyro_bias = [ADCS.Bias(bias=gyro_bias_mean[j], std_bias=gyro_bsr) for j in range(3)]
gyros = [ADCS.Gyro(axis=MathConstants.unitvecs[j], bias=gyro_bias[j], noise=gyro_noise.copy()) for j in range(3)]

# Sensors: SunPair
sun_noise = ADCS.Noise(noise=0.0, std_noise=0.0001)
sun_eff = 1.0
sun_bias_mean = np.array([0.05,0.09,-0.03])*sun_eff
sun_bsr = 0.00001*sun_eff
sun_bias = [ADCS.Bias(bias=sun_bias_mean[j], std_bias=sun_bsr) for j in range(3)]
suns = [ADCS.SunPair(axis=MathConstants.unitvecs[j], efficiency=sun_eff, bias=sun_bias[j], noise=sun_noise.copy()) for j in range(3)]

real_sat = ADCS.Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]), sensors=mtms+gyros+suns)
w0 = random_n_unit_vec(3)*np.random.uniform(0, 0.1)*np.pi/180.0
q0 = random_n_unit_vec(4)
x = np.concatenate([w0, q0])

est_mtm_bias = [ADCS.Bias(bias=0.0, std_bias=mtm_bsr) for j in range(3)]
est_mtms = [ADCS.MTM(axis=MathConstants.unitvecs[j], bias=est_mtm_bias[j], noise=mtm_noise.copy(), estimate_bias=True) for j in range(3)]
est_gyro_bias = [ADCS.Bias(bias=0.0, std_bias=gyro_bsr) for j in range(3)]
est_gyros = [ADCS.Gyro(axis=MathConstants.unitvecs[j], bias=est_gyro_bias[j], noise=gyro_noise.copy(), estimate_bias=True) for j in range(3)]
est_sun_bias = [ADCS.Bias(bias=0.0, std_bias=sun_bsr) for j in range(3)]
est_suns = [ADCS.SunPair(axis=MathConstants.unitvecs[j], efficiency=sun_eff, bias=est_sun_bias[j], noise=sun_noise.copy(), estimate_bias=True) for j in range(3)]

est_sat = ADCS.EstimatedSatellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]), sensors=est_mtms+est_gyros+est_suns)

x_hat = np.zeros(16)
x_hat[3] = 1
invJ = np.linalg.inv(est_sat.J_0)
sigma_torque = 1e-4 
Q_torque_continuous = np.eye(3) * sigma_torque**2
Q_alpha = invJ @ Q_torque_continuous @ invJ.T
Q_omega = Q_alpha * dt
Q_att   = Q_alpha * (dt**3 / 3.0)
Q_cross = Q_alpha * (dt**2 / 2.0)
Q_dyn_block = np.block([
    [Q_omega, Q_cross],
    [Q_cross, Q_att]
])
# Biases
mult_mtm = 1
mult_sun = 10.0
Q_mtm  = np.eye(3) * (mtm_bsr * mult_mtm)**2.0 * dt
Q_gyro = np.eye(3) * (gyro_bsr)**2.0 * dt
Q_sun  = np.eye(3) * (sun_bsr * mult_sun)**2.0 * dt
Q_est = block_diag(Q_dyn_block, Q_mtm, Q_gyro, Q_sun)
P_est = block_diag(np.eye(3)*(0.01)**2.0, np.eye(3)*3, 0.001*np.eye(3)*mtm_bsr**2.0, np.eye(3)*1000*gyro_bsr**2.0, np.eye(3)*100*sun_bsr**2.0)
att_estimator = ADCS.SRUAKF(est_sat=est_sat, J2000=0.22, x_hat=x_hat, P_hat=P_est, Q_hat=Q_est, dt=dt, cross_term=True, quat_as_vec=False)

os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(),J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 7.5, 1]))
goal = ADCS.ECI_Goal(np.array([0, 0, 1]))

results = ADCS.simulate(
    x=x,
    satellite=real_sat,
    est_satellite=est_sat,
    estimator=att_estimator,
    os0=os0,
    goal=goal,
    dt=50.0,
    tf=2000.0
)

from ADCS.helpers.plot import plot, AngularVelocityPlot, AngularVelocityPlotSingle, AngularVelocityPlotCombined

plot(
    results,
    AngularVelocityPlot(sources=["real", "estimated"]),
    AngularVelocityPlotSingle(component="m", sources=["real", "estimated"]),
    AngularVelocityPlotCombined(sources=["real", "estimated"]),
    layout=(3,1),
    title="Angular Rates",
)

from ADCS.helpers.plot import QuaternionPlot, QuaternionPlotSingle, QuaternionPlotCombined

plot(
    results,
    QuaternionPlot(sources=["real", "estimated"]),
    QuaternionPlotSingle(component=0, sources=["real", "estimated"]),
    QuaternionPlotCombined(sources=["real", "estimated"]),
    layout=(3,1),
    title="Attitude Quaternions",
)

from ADCS.helpers.plot import TargetPlot

plot(
    results,
    TargetPlot(modes=["real_target", "real_est", "est_target", "directions3d"]),
    layout=(1,1),
    title="Target Tracking Error",
)

from ADCS.helpers.plot import SensorsPlot, SensorsPlotSingle, SensorsPlotCombined

plot(
    results,
    SensorsPlot(sources=["clean", "real"]),
    SensorsPlotSingle(index=0, sources=["clean", "real"]),
    SensorsPlotCombined(sources=["clean", "real"]),
    layout=(3,1),
    title="Sensor Measurements",
)

from ADCS.helpers.plot import BiasPlot, BiasPlotSingle, BiasPlotCombined

plot(
    results,
    BiasPlot(kind="sensor", sources=["real", "estimated"]),
    BiasPlotSingle(index=0, kind="sensor", sources=["real", "estimated"]),
    BiasPlotCombined(kind="sensor", sources=["real", "estimated"]),
    layout=(3,1),
    title="Sensor Biases",
)

from ADCS.helpers.plot import IlluminationPlot

plot(
    results,
    IlluminationPlot(),
    layout=(1,1),
    title="Satellite Illumination",
)

plt.show()