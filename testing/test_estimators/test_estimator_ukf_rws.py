import sys
import os
import numpy as np
from typing import List, Tuple
from scipy.linalg import block_diag
from scipy.integrate import solve_ivp
from tqdm import tqdm
import time

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from ADCS.controller import MTQ_w_RW, BDot
from ADCS.estimators.attitude_estimators.attitude_SRUAKF import UAKF, SRUAKF
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import Noise, Bias, MTQ, RW
from ADCS.satellite_hardware.sensors import MTM, Gyro, SunPair
from ADCS.satellite_hardware.disturbances import GeometryFace, GG_Disturbance, Drag_Disturbance, GeometryConfig, DisturbanceMode
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import random_n_unit_vec, normalize, norm
from ADCS.flight_software.single_core.ttc_single_core import TTC_Single_Core
from ADCS.flight_software.tasks.task import Task
from ADCS.helpers.plotting.plot_estimator import plot_state_comparison, plot_error_and_sun, plot_sensor_data, plot_bias_comparison
from ADCS.helpers.plotting.animate_estimator import animate_attitude
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window


def create_satellite() -> Satellite:
    mtq_noise = Noise(noise=0.0, std_noise=0.0001)
    mtq_max_torque = 1.0
    mtqs = [MTQ(axis=j, max_torque=mtq_max_torque, noise=mtq_noise) for j in MathConstants.unitvecs]

    rw_max_torque = 4.51
    rw_J = 0.22
    rw_h0 = 1
    rw_hmax = 3.8
    rw_noise = Noise(noise=0.0, std_noise=0.0001)
    rw_h_noise = Noise(noise=0.0, std_noise=0.0001)
    rws = [RW(axis=j, max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax, noise=rw_noise, h_meas_noise=rw_h_noise) for j in MathConstants.unitvecs]

    mtm_noise = Noise(noise=0.0, std_noise=1e-8)
    mtms = [MTM(axis=MathConstants.unitvecs[j], noise=mtm_noise) for j in range(3)]

    gyro_noise = Noise(noise=0.0, std_noise=0.0001)
    gyros = [Gyro(axis=MathConstants.unitvecs[j], noise=gyro_noise) for j in range(3)]

    sun_noise = Noise(noise=0.0, std_noise=0.0001)
    sun_eff = 0.3
    suns = [SunPair(axis=MathConstants.unitvecs[j], efficiency=sun_eff, noise=sun_noise) for j in range(3)]

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
    real_sat = Satellite(mass=real_sat_mass, J_0=real_sat_J, actuators=mtqs+rws, sensors=mtms+gyros+suns, disturbances=dists)

    return real_sat
    
def create_estimated_satellite() -> EstimatedSatellite:
    mtq_noise = Noise(noise=0.0, std_noise=0.0001)
    mtq_max_torque = 1.0
    est_mtqs = [MTQ(axis=j, max_torque=mtq_max_torque, noise=mtq_noise) for j in MathConstants.unitvecs]

    rw_max_torque = 4.51
    rw_J = 0.22
    rw_h0 = 1
    rw_hmax = 3.8
    rw_noise = Noise(noise=0.0, std_noise=0.0001)
    rw_h_noise = Noise(noise=0.0, std_noise=0.0001)
    est_rws = [RW(axis=j, max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax, noise=rw_noise, h_meas_noise=rw_h_noise) for j in MathConstants.unitvecs]

    mtm_noise = Noise(noise=0.0, std_noise=1e-8)
    est_mtms = [MTM(axis=MathConstants.unitvecs[j], noise=mtm_noise) for j in range(3)]

    gyro_noise = Noise(noise=0.0, std_noise=0.0001)
    est_gyros = [Gyro(axis=MathConstants.unitvecs[j], noise=gyro_noise) for j in range(3)]

    sun_noise = Noise(noise=0.0, std_noise=0.0001)
    sun_eff = 0.3
    est_suns = [SunPair(axis=MathConstants.unitvecs[j], efficiency=sun_eff, noise=sun_noise) for j in range(3)]

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
    est_sat = EstimatedSatellite(mass=est_sat_mass, J_0=est_sat_J, actuators=est_mtqs+est_rws, sensors=est_mtms+est_gyros+est_suns, disturbances=est_dists)

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
    # RW
    Q_rw  = np.eye(3) * (0.0001)**2.0 * dt
    Q_est = block_diag(Q_dyn_block, Q_rw)

    P_est = block_diag(np.eye(3)*(0.01)**2.0, np.eye(3)*0.1, np.eye(3)*0.2)

    return P_est, Q_est

def main(verbose: bool = False):
    np.random.seed(1)
    tf = 300
    t0 = 0
    dt = 5
    N = int((tf-t0)/dt)

    real_sat = create_satellite()
    est_sat = create_estimated_satellite()

    w0 = random_n_unit_vec(3)*np.random.uniform(0, 0.1)*np.pi/180.0
    q0 = random_n_unit_vec(4)
    h0 = np.array([1, 1, 1])
    x = np.concatenate([w0, q0, h0])
    x_hat = np.array([0, 0, 0, 1, 0, 0, 0, 0.1, 0.1, 0.1])

    orb = create_orbit(t0, tf, dt)

    J2000 = 0.22+t0*TimeConstants.sec2cent
    P_est, Q_est = create_matrices(est_sat=est_sat, dt=dt)
    ukf = SRUAKF(est_sat=est_sat, J2000=J2000, x_hat=x_hat, P_hat=P_est, Q_hat=Q_est, dt=dt, cross_term=True)

    time_hist = np.nan*np.zeros(N)
    state_hist = np.nan*np.zeros((N, ukf.state_len))
    est_state_hist = np.nan*np.zeros((N, ukf.state_len))
    os_hist: List[Orbital_State] = list()
    sensor_hist: List[np.ndarray] = np.nan*np.zeros((N, 12))
    clean_sensor_hist: List[np.ndarray] = np.nan*np.zeros((N, 12))
    u_hist = np.nan*np.zeros((N, 6))
    cov_hist: List[np.ndarray] = list()
    
    t = t0
    ind = 0
    
    steps = int((tf - t0)/dt)

    for step in tqdm(range(steps), desc="Simulating ukf"):
        # One Step Propagation
        J2000 = 0.22 + t*TimeConstants.sec2cent
        os = orb.get_os(J2000=J2000)

        # Determine control
        u = np.zeros(6)

        noisy_sensor_readings = real_sat.sensor_readings(x=x, os=os)
        clean_sensor_readings = real_sat.noiseless_sensor_readings(x=x, os=os)
        x_hat = ukf.update(u=u, sensors=noisy_sensor_readings, os=os)

        if verbose:
            # Full State Debug
            print("Real State ", x)
            print("Estimated State ", x_hat)

            # Attitude Debug
            quaternion_error_deg = (180.0/np.pi)*np.acos(-1 + 2*np.clip(np.dot(x_hat[3:7], x[3:7]), -1, 1)**2.0)
            print("Attitude Error (Degrees) ", quaternion_error_deg)
            angular_velocity_error = norm(x_hat[0:3] - x[0:3])*180.0/np.pi
            print("Angular Velocity Error ", angular_velocity_error)
            diagonal_covariances = np.diagonal(ukf.x_hat.cov)
            print("Attitude Covariance ", diagonal_covariances[3:6])
            print("Angular Velocity Covariance ", diagonal_covariances[0:3])
            print("")

        # Save Information for Plotting
        time_hist[ind] = t
        state_hist[ind,:] = x
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


    plot_state_comparison(time_hist, state_hist, est_state_hist)
    plot_error_and_sun(time_hist, state_hist, est_state_hist, os_hist)
    plot_sensor_data(time_hist, sensor_hist, clean_sensor_hist)
    animate_attitude(time_hist, state_hist, est_state_hist, os_hist)
    create_close_all_button_window()


if __name__ == "__main__":
    main(verbose=False)