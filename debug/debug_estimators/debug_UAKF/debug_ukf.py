import sys
import os
import numpy as np
import numdifftools as nd
import pytest
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Union
from scipy.stats import kstest, ks_2samp
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag
from asciichartpy import plot
import time
from tqdm import tqdm

# === Import project modules ===
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import Actuator, RW, MTQ
from ADCS.satellite_hardware.sensors import MTM, Gyro, SunPair
from ADCS.satellite_hardware.sensors import GPS
from ADCS.satellite_hardware.errors import Noise, Bias, ErrorMode
from ADCS.satellite_hardware.disturbances import SRP_Disturbance, Drag_Disturbance, Prop_Disturbance, Dipole_Disturbance, GG_Disturbance, GeometryConfig, GeometryFace
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import EarthConstants
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_helpers import random_n_unit_vec, rot_mat, norm, normalize, limit
from ADCS.helpers.math_constants import MathConstants
from ADCS.estimators.attitude_estimators import UAKF, SRUAKF
from ADCS.state import State, EstimatedState

from ADCS.helpers.plotting.plot_estimator import plot_state_comparison, plot_error_and_sun, plot_sensor_data, plot_bias_comparison
from ADCS.helpers.plotting.animate_estimator import animate_attitude
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window

def run_ukf(verbose: bool = False, tf: float = 1000, dt: float = 10, real_orbit: bool = False):
    t0 = 0

    N = int((tf-t0)/dt)
    np.set_printoptions(precision=6)

    ## REAL SATELLITE
    # Actuators: Magnetorquers
    mtq_noise = Noise(noise=0.0, std_noise=0.0001)
    mtq_max_torque = 1.0
    acts = [MTQ(axis=j, max_torque=mtq_max_torque, noise=mtq_noise.copy()) for j in MathConstants.unitvecs]

    # Sensors: Magnetometers
    mtm_noise = Noise(noise=0.0, std_noise=1e-8)
    mtm_bias_mean = random_n_unit_vec(3)*np.random.uniform(1e-9, 1e-7)
    mtm_bsr = 1e-9
    mtm_bias = [Bias(bias=mtm_bias_mean[j], std_bias=mtm_bsr) for j in range(3)]
    mtms = [MTM(axis=MathConstants.unitvecs[j], bias=mtm_bias[j], noise=mtm_noise.copy()) for j in range(3)]

    # Sensors: Gyroscopes
    gyro_noise = Noise(noise=0.0, std_noise=0.0001)
    gyro_bias_mean = np.array([0.002, 0.002, 0.002])
    gyro_bsr = 0.0004*np.pi/180.0
    gyro_bias = [Bias(bias=gyro_bias_mean[j], std_bias=gyro_bsr) for j in range(3)]
    gyros = [Gyro(axis=MathConstants.unitvecs[j], bias=gyro_bias[j], noise=gyro_noise.copy()) for j in range(3)]

    # Sensors: SunPair
    sun_noise = Noise(noise=0.0, std_noise=0.0001)
    sun_eff = 1.0
    sun_bias_mean = np.array([0.05,0.09,-0.03])*sun_eff
    sun_bsr = 0.00001*sun_eff
    sun_bias = [Bias(bias=sun_bias_mean[j], std_bias=sun_bsr) for j in range(3)]
    suns = [SunPair(axis=MathConstants.unitvecs[j], efficiency=sun_eff, bias=sun_bias[j], noise=sun_noise.copy()) for j in range(3)]

    # Disturbance: Drag
    faces: List[GeometryFace] = [GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[0]*0.05, normal=MathConstants.unitvecs[0], CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[0]*0.05, normal=-MathConstants.unitvecs[0], CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[1]*0.05, normal=MathConstants.unitvecs[1], CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[1]*0.05, normal=-MathConstants.unitvecs[1], CD=2.2),
                                 GeometryFace(area=0.1*0.1, centroid=MathConstants.unitvecs[2]*0.15, normal=MathConstants.unitvecs[2], CD=2.2),
                                 GeometryFace(area=0.1*0.1, centroid=-MathConstants.unitvecs[2]*0.15, normal=-MathConstants.unitvecs[2], CD=2.2)]
    config = GeometryConfig(geometry_faces=faces)
    drag_dist = Drag_Disturbance(config=config)

    # Disturbance: Gravity Gradient
    gg_dist = GG_Disturbance()

    dists = [drag_dist, gg_dist]

    # Satellite configuration
    real_sat_mass = 4.0
    real_sat_J = np.diagflat([3.4, 2.9, 1.3])
    real_sat = Satellite(mass=real_sat_mass, J_0=real_sat_J, actuators=acts, sensors=mtms+gyros+suns, disturbances=dists)

    # Initial State
    w0 = random_n_unit_vec(3)*np.random.uniform(0, 0.1)*np.pi/180.0
    print(w0)

    q0 = random_n_unit_vec(4)
    print(q0)

    x = State(w=w0, q=q0)
    ephem = Ephemeris()
    start_time = 0.22 - 1*TimeConstants.sec2cent
    end_time = 0.22 + (tf-t0)*TimeConstants.sec2cent
    R = 7000*np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([8, 0, 0])
    if real_orbit:
        # Real Orbit Generation
        os = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
        orb = Orbit(os0=os, end_time=end_time, dt=dt, zonal_J=2, fast=False)
    else:
        os = Orbital_State(ephem=ephem, J2000=0.22-1*TimeConstants.sec2cent, R=R, V=V, B=np.array([0, 0.1, 0]), S=np.array([1e5+1, 0, 0]), rho=5e-12)
        dur = int((tf-t0)/dt)+10
        orbs = [os]*(dur+10)
        for j in range(dur):
            orbs[j] = os.copy()
            orbs[j].J2000 = os.J2000 + j*dt*TimeConstants.sec2cent
        orb = Orbit(orbs)

    ## ESTIMATED SATELLITE
    # Actuators
    est_acts = [MTQ(axis=j, max_torque=mtq_max_torque, noise=mtq_noise.copy()) for j in MathConstants.unitvecs]
    # Sensors
    est_mtm_bias = [Bias(bias=0.0, std_bias=mtm_bsr) for j in range(3)]
    est_mtms = [MTM(axis=MathConstants.unitvecs[j], bias=est_mtm_bias[j], noise=mtm_noise.copy(), estimate_bias=True) for j in range(3)]
    est_gyro_bias = [Bias(bias=0.0, std_bias=gyro_bsr) for j in range(3)]
    est_gyros = [Gyro(axis=MathConstants.unitvecs[j], bias=est_gyro_bias[j], noise=gyro_noise.copy(), estimate_bias=True) for j in range(3)]
    est_sun_bias = [Bias(bias=0.0, std_bias=sun_bsr) for j in range(3)]
    est_suns = [SunPair(axis=MathConstants.unitvecs[j], efficiency=sun_eff, bias=est_sun_bias[j], noise=sun_noise.copy(), estimate_bias=True) for j in range(3)]
    # Disturbances
    est_drag_dist = Drag_Disturbance(config=config)
    est_gg_dist = GG_Disturbance()
    est_dists = [est_drag_dist, est_gg_dist]

    # Satellite configuration
    est_sat_mass = 4.0
    est_sat_J = np.diagflat([3.4, 2.9, 1.3])
    est_sat = EstimatedSatellite(mass=est_sat_mass, J_0=est_sat_J, actuators=est_acts, sensors=est_mtms+est_gyros+est_suns, disturbances=est_dists)

    # Initial Estimated State
    x_hat = EstimatedState(w=np.zeros(3), q=[1, 0, 0, 0], sens_bias=np.zeros(9))

    # Create Covariance Matrices
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
    #Q_est = block_diag(np.eye(3)*(1e-4)**2.0, 1e-4*np.eye(3), 0.1*np.eye(3)*mtm_bsr**2.0, np.eye(3)*gyro_bsr**2.0, 0.1*np.eye(3)*sun_bsr**2.0)

    ## Build Estimator
    J2000 = 0.22 + t0*TimeConstants.sec2cent
    ukf = UAKF(est_sat=est_sat, J2000=J2000, x_hat=x_hat, P_hat=P_est, Q_hat=Q_est, dt=dt, cross_term=True, quat_as_vec=False)

    # Create history vectors
    time_hist = np.nan*np.zeros(N)
    state_hist: List[State] = []
    est_state_hist: List[EstimatedState] = []
    os_hist: List[Orbital_State] = list()
    sensor_hist: List[np.ndarray] = np.nan*np.zeros((N, 9))
    clean_sensor_hist: List[np.ndarray] = np.nan*np.zeros((N, 9))
    u_hist = np.nan*np.zeros((N, len(acts)))
    real_mtm_bias_hist = np.nan*np.zeros((N, 3))
    est_mtm_bias_hist = np.nan*np.zeros((N, 3))
    real_gyro_bias_hist = np.nan*np.zeros((N, 3))
    est_gyro_bias_hist = np.nan*np.zeros((N, 3))
    real_sun_bias_hist = np.nan*np.zeros((N, 3))
    est_sun_bias_hist = np.nan*np.zeros((N, 3))
    cov_hist: List[np.ndarray] = list()
    
    t = t0
    ind = 0
    
    steps = int((tf - t0)/dt)
    
    for step in tqdm(range(steps), desc="Simulating ukf"):
        # Determine control
        u = np.zeros(len(acts))

        dmode = ErrorMode(add_bias=True, add_noise=True, update_bias=True, update_noise=True)
        noisy_sensor_readings = real_sat.sensor_readings(x=x, os=os, dmode=dmode)
        clean_sensor_readings = real_sat.noiseless_sensor_readings(x=x, os=os)
        x_hat = ukf.update(u=u, sensors=noisy_sensor_readings, os=os)

        if verbose:
            # Full State Debug
            print("Real State ", x.as_array())
            print("Estimated State ", x_hat.as_array())
            print("Real Gyro Bias ", [gyro.bias.bias for gyro in real_sat.attitude_sensors if isinstance(gyro, Gyro)])
            print("Estimated Gyro Bias ", x_hat.sens_bias[3:6])
            print("Real MTM Bias ", [mtm.bias.bias for mtm in real_sat.attitude_sensors if isinstance(mtm, MTM)])
            print("Estimated MTM Bias ", x_hat.sens_bias[0:3])

            # Attitude Debug
            quaternion_error_deg = (180.0/np.pi)*np.acos(-1 + 2*np.clip(np.dot(x_hat.q, x.q), -1, 1)**2.0)
            print("Attitude Error (Degrees) ", quaternion_error_deg)
            angular_velocity_error = norm(x_hat.w - x.w)*180.0/np.pi
            print("Angular Velocity Error ", angular_velocity_error)
            diagonal_covariances = np.diagonal(ukf.x_hat.cov)
            print("Attitude Covariance ", diagonal_covariances[3:6])
            print("Angular Velocity Covariance ", diagonal_covariances[0:3])
            print("Bias Covariance: ", diagonal_covariances[6:9])
            print("")

        # Save Information for Plotting
        time_hist[ind] = t
        real_gyro_biases = np.concatenate([gyro.bias.bias for gyro in real_sat.sensors if isinstance(gyro, Gyro)])
        real_mtm_biases = np.concatenate([mtm.bias.bias for mtm in real_sat.sensors if isinstance(mtm, MTM)])
        real_sun_biases = np.concatenate([sun.bias.bias for sun in real_sat.sensors if isinstance(sun, SunPair)])
        state_hist.append(x.copy())
        est_state_hist.append(x_hat.copy())
        real_mtm_bias_hist[ind,:] = real_mtm_biases
        est_mtm_bias_hist[ind,:] = x_hat.sens_bias[0:3]
        real_gyro_bias_hist[ind,:] = real_gyro_biases
        est_gyro_bias_hist[ind,:] = x_hat.sens_bias[3:6]
        real_sun_bias_hist[ind,:] = real_sun_biases
        est_sun_bias_hist[ind,:] = x_hat.sens_bias[6:9]
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

        out = solve_ivp(fun=real_sat.dynamics_for_solver, t_span=(0, dt), y0=x.as_array(), method="RK45", args=(u, prev_os, os), rtol=1e-7, atol=1e-7)

        x = State.from_array(out.y[:, -1])
        x = x.normalized()

    return (
        time_hist,
        state_hist,
        est_state_hist,
        os_hist,
        sensor_hist,
        clean_sensor_hist,
        u_hist,
        real_mtm_bias_hist,
        est_mtm_bias_hist,
        real_gyro_bias_hist,
        est_gyro_bias_hist,
        real_sun_bias_hist,
        est_sun_bias_hist,
        cov_hist,
    )


def plot_ukf(verbose: bool = False, tf: float = 60, dt: float = 1, real_orbit: bool = False) -> None:
    (
        time_hist,
        state_hist,
        est_state_hist,
        os_hist,
        sensor_hist,
        clean_sensor_hist,
        u_hist,
        real_mtm_bias_hist,
        est_mtm_bias_hist,
        real_gyro_bias_hist,
        est_gyro_bias_hist,
        real_sun_bias_hist,
        est_sun_bias_hist,
        cov_hist,
    ) = run_ukf(verbose=verbose, tf=tf, dt=dt, real_orbit=real_orbit)

    plot_state_comparison(time_hist, state_hist, est_state_hist)
    plot_error_and_sun(time_hist, state_hist, est_state_hist, os_hist)
    plot_sensor_data(time_hist, sensor_hist, clean_sensor_hist)
    plot_bias_comparison(time_hist, real_gyro_bias_hist, est_gyro_bias_hist,
                        "Real vs Estimated Gyroscope Bias", "rad/s")
    plot_bias_comparison(time_hist, real_mtm_bias_hist, est_mtm_bias_hist,
                        "Real vs Estimated MTM Bias", "T/s")
    plot_bias_comparison(time_hist, real_sun_bias_hist, est_sun_bias_hist,
                        "Real vs Estimated Sun Bias", "W/s")
    animate_attitude(time_hist, state_hist, est_state_hist, os_hist)
    create_close_all_button_window()
    
if __name__ == "__main__":
    plot_ukf(verbose=False, tf=2000, dt=100, real_orbit=True)
