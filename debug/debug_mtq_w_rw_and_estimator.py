import sys
import os
import numpy as np
from typing import List, Tuple
from scipy.linalg import block_diag
from scipy.integrate import solve_ivp
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from ADCS.estimators.attitude_estimators.attitude_SRUAKF import SRUAKF
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import Noise, Bias, MTQ
from ADCS.satellite_hardware.sensors import MTM, Gyro, SunPair
from ADCS.satellite_hardware.disturbances import GeometryFace, GG_Disturbance, Drag_Disturbance, GeometryConfig, DisturbanceMode
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import random_n_unit_vec, normalize

from ADCS.helpers.plotting.plot_estimator import plot_state_comparison, plot_error_and_sun, plot_sensor_data, plot_bias_comparison
from ADCS.helpers.plotting.animate_estimator import animate_attitude
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window

def create_satellite() -> Satellite:
    mtq_noise = Noise(noise=0.0, std_noise=0.0001)
    mtq_max_torque = 1.0
    acts = [MTQ(axis=j, max_torque=mtq_max_torque, noise=mtq_noise.copy()) for j in MathConstants.unitvecs]

    mtm_noise = Noise(noise=0.0, std_noise=1e-8)
    mtm_bias_mean = random_n_unit_vec(3)*np.random.uniform(1e-9, 1e-7)
    mtm_bsr = 1e-9
    mtm_bias = [Bias(bias=mtm_bias_mean[j], std_bias=mtm_bsr) for j in range(3)]
    mtms = [MTM(axis=MathConstants.unitvecs[j], bias=mtm_bias[j], noise=mtm_noise.copy()) for j in range(3)]

    gyro_noise = Noise(noise=0.0, std_noise=0.0001)
    gyro_bias_mean = np.array([0.002, 0.002, 0.002])
    gyro_bsr = 0.0004*np.pi/180.0
    gyro_bias = [Bias(bias=gyro_bias_mean[j], std_bias=gyro_bsr) for j in range(3)]
    gyros = [Gyro(axis=MathConstants.unitvecs[j], bias=gyro_bias[j], noise=gyro_noise.copy()) for j in range(3)]

    sun_noise = Noise(noise=0.0, std_noise=0.0001)
    sun_eff = 1.0
    sun_bias_mean = np.array([0.05,0.09,-0.03])*sun_eff
    sun_bsr = 0.00001*sun_eff
    sun_bias = [Bias(bias=sun_bias_mean[j], std_bias=sun_bsr) for j in range(3)]
    suns = [SunPair(axis=MathConstants.unitvecs[j], efficiency=sun_eff, bias=sun_bias[j], noise=sun_noise.copy()) for j in range(3)]

    faces: List[GeometryFace] = [GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[0]*0.05, normal=MathConstants.unitvecs[0], CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[0]*0.05, normal=-MathConstants.unitvecs[0], CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[1]*0.05, normal=MathConstants.unitvecs[1], CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[1]*0.05, normal=-MathConstants.unitvecs[1], CD=2.2),
                                 GeometryFace(area=0.1*0.1, centroid=MathConstants.unitvecs[2]*0.15, normal=MathConstants.unitvecs[2], CD=2.2),
                                 GeometryFace(area=0.1*0.1, centroid=-MathConstants.unitvecs[2]*0.15, normal=-MathConstants.unitvecs[2], CD=2.2)]
    config = GeometryConfig(geometry_faces=faces)
    drag_dist = Drag_Disturbance(config=config)
    gg_dist = GG_Disturbance()
    dists = [drag_dist, gg_dist]

    real_sat_mass = 4.0
    real_sat_J = np.diagflat([3.4, 2.9, 1.3])
    real_sat = Satellite(mass=real_sat_mass, J_0=real_sat_J, actuators=acts, sensors=mtms+gyros+suns, disturbances=dists)

    return real_sat

def create_estimated_satellite() -> EstimatedSatellite:
    mtq_noise = Noise(noise=0.0, std_noise=0.0001)
    mtq_max_torque = 1.0
    est_acts = [MTQ(axis=j, max_torque=mtq_max_torque, noise=mtq_noise.copy()) for j in MathConstants.unitvecs]

    mtm_noise = Noise(noise=0.0, std_noise=1e-8)
    mtm_bsr = 1e-9
    mtm_bias = [Bias(bias=0.0, std_bias=mtm_bsr) for j in range(3)]
    est_mtms = [MTM(axis=MathConstants.unitvecs[j], bias=mtm_bias[j], noise=mtm_noise.copy(), estimate_bias=True) for j in range(3)]

    gyro_noise = Noise(noise=0.0, std_noise=0.0001)
    gyro_bsr = 0.0004*np.pi/180.0
    gyro_bias = [Bias(bias=0.0, std_bias=gyro_bsr) for j in range(3)]
    est_gyros = [Gyro(axis=MathConstants.unitvecs[j], bias=gyro_bias[j], noise=gyro_noise.copy(), estimate_bias=True) for j in range(3)]

    sun_noise = Noise(noise=0.0, std_noise=0.0001)
    sun_eff = 1.0
    sun_bsr = 0.00001*sun_eff
    sun_bias = [Bias(bias=0.0, std_bias=sun_bsr) for j in range(3)]
    est_suns = [SunPair(axis=MathConstants.unitvecs[j], efficiency=sun_eff, bias=sun_bias[j], noise=sun_noise.copy(), estimate_bias=True) for j in range(3)]

    faces: List[GeometryFace] = [GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[0]*0.05, normal=MathConstants.unitvecs[0], CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[0]*0.05, normal=-MathConstants.unitvecs[0], CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[1]*0.05, normal=MathConstants.unitvecs[1], CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[1]*0.05, normal=-MathConstants.unitvecs[1], CD=2.2),
                                 GeometryFace(area=0.1*0.1, centroid=MathConstants.unitvecs[2]*0.15, normal=MathConstants.unitvecs[2], CD=2.2),
                                 GeometryFace(area=0.1*0.1, centroid=-MathConstants.unitvecs[2]*0.15, normal=-MathConstants.unitvecs[2], CD=2.2)]
    config = GeometryConfig(geometry_faces=faces)
    est_drag_dist = Drag_Disturbance(config=config)
    est_gg_dist = GG_Disturbance()
    est_dists = [est_drag_dist, est_gg_dist]

    # Satellite configuration
    est_sat_mass = 4.0
    est_sat_J = np.diagflat([3.4, 2.9, 1.3])
    est_sat = EstimatedSatellite(mass=est_sat_mass, J_0=est_sat_J, actuators=est_acts, sensors=est_mtms+est_gyros+est_suns, disturbances=est_dists)

    return est_sat

def create_orbit(t0: float, tf: float, dt: float) -> Orbit:
    ephem = Ephemeris()
    start_time = 0.22 - 1*TimeConstants.sec2cent
    end_time = 0.22 + (tf-t0)*TimeConstants.sec2cent
    R = -7000*np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([8, 0, 0])
    if True:
        os = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
        orb = Orbit(os0=os, end_time=end_time, dt=dt, use_J2=True, fast=False)
    else:
        os = Orbital_State(ephem=ephem, J2000=0.22-1*TimeConstants.sec2cent, R=R, V=V, B=np.array([0, 0.1, 0]), S=np.array([1e5+1, 0, 0]), rho=5e-12)
        dur = int((tf-t0)/dt)+10
        orbs = [os]*(dur+10)
        for j in range(dur):
            orbs[j] = os.copy()
            orbs[j].J2000 = os.J2000 + j*dt*TimeConstants.sec2cent
        orb = Orbit(orbs)

    return orb

def create_matrices(est_sat: EstimatedSatellite, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    mtm_bsr = 1e-9
    gyro_bsr = 0.0004*np.pi/180.0
    sun_bsr = 0.00001
    
    invJ = np.linalg.inv(est_sat.J_0)
    # Dynamics
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

    return P_est, Q_est

def plot_results(time_hist, state_hist, est_state_hist, os_hist, sensor_hist, clean_sensor_hist, u_hist, cov_hist) -> None:
    plot_state_comparison(time_hist, state_hist, est_state_hist)
    plot_error_and_sun(time_hist, state_hist, est_state_hist, os_hist)
    plot_sensor_data(time_hist, sensor_hist, clean_sensor_hist)
    plot_bias_comparison(time_hist, state_hist[:,10:13], est_state_hist[:,10:13], 
                        "Real vs Estimated Gyroscope Bias", "rad/s")
    plot_bias_comparison(time_hist, state_hist[:,7:10], est_state_hist[:,7:10], 
                        "Real vs Estimated MTM Bias", "T/s")
    plot_bias_comparison(time_hist, state_hist[:,13:16], est_state_hist[:,13:16], 
                        "Real vs Estimated Sun Bias", "W/s")
    animate_attitude(time_hist, state_hist, est_state_hist, os_hist)
    create_close_all_button_window()


def main():
    np.random.seed(3)
    t0 = 0
    tf = 100
    dt = 3

    N = int((tf-t0)/dt)

    orb = create_orbit(t0, tf, dt)
    os = orb.get_os(0.22)
    real_sat = create_satellite()
    est_sat = create_estimated_satellite()
    P_est, Q_est = create_matrices(est_sat, dt)

    w0 = np.array([0, 0, 0])
    q0 = np.array([1, 0, 0, 0])
    x = np.concatenate([w0, q0])

    x_hat = np.zeros(16)
    x_hat[3] = 1

    J2000 = 0.22 + t0*TimeConstants.sec2cent
    ukf = SRUAKF(est_sat=est_sat, J2000=J2000, x_hat=x_hat, P_hat=P_est, Q_hat=Q_est, dt=dt, cross_term=True, quat_as_vec=False)

    time_hist = np.nan*np.zeros(N)
    state_hist = np.nan*np.zeros((N, ukf.state_len))
    est_state_hist = np.nan*np.zeros((N, ukf.state_len))
    os_hist: List[Orbital_State] = list()
    sensor_hist: List[np.ndarray] = np.nan*np.zeros((N, 9))
    clean_sensor_hist: List[np.ndarray] = np.nan*np.zeros((N, 9))
    u_hist = np.nan*np.zeros((N, 3))
    cov_hist: List[np.ndarray] = list()
    
    t = t0
    ind = 0
    
    steps = int((tf - t0)/dt)

    for step in tqdm(range(steps), desc="Simulating ukf"):
        u = np.zeros(3)

        dmode = DisturbanceMode(add_bias=True, add_noise=True, update_bias=True, update_noise=True)
        noisy_sensor_readings = real_sat.sensor_readings(x=x, os=os, dmode=dmode)
        clean_sensor_readings = real_sat.noiseless_sensor_readings(x=x, os=os)
        x_hat = ukf.update(u=u, sensors=noisy_sensor_readings, os=os)

        time_hist[ind] = t
        real_gyro_biases = np.concatenate([gyro.bias.bias for gyro in real_sat.sensors if isinstance(gyro, Gyro)])
        real_mtm_biases = np.concatenate([mtm.bias.bias for mtm in real_sat.sensors if isinstance(mtm, MTM)])
        real_sun_biases = np.concatenate([sun.bias.bias for sun in real_sat.sensors if isinstance(sun, SunPair)])
        state_hist[ind,:] = np.concatenate([x, real_mtm_biases, real_gyro_biases, real_sun_biases])
        est_state_hist[ind,:] = x_hat
        os_hist += [os]
        sensor_hist[ind,:] = noisy_sensor_readings
        clean_sensor_hist[ind,:] = clean_sensor_readings
        u_hist[ind,:] = u
        cov_hist += [ukf.x_hat.cov]

        # Propagate
        ind += 1
        t += dt
        prev_os = os.copy()
        os = orb.get_os(0.22+(t-t0)*TimeConstants.sec2cent)

        out = solve_ivp(fun=real_sat.dynamics_for_solver, t_span=(0, dt), y0=x, method="RK45", args=(u, prev_os, os), rtol=1e-7, atol=1e-7)

        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])

    plot_results(time_hist, state_hist, est_state_hist, os_hist, sensor_hist, clean_sensor_hist, u_hist, cov_hist)


if __name__ == "__main__":
    main()