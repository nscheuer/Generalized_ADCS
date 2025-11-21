import sys
import os
import numpy as np
import numdifftools as nd
import pytest
from typing import List
from scipy.stats import kstest, ks_2samp
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag
from asciichartpy import plot
import time

# === Import project modules ===
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import Actuator, RW, MTQ, Noise, Bias
from ADCS.satellite_hardware.sensors import MTM, Gyro, SunPair
from ADCS.satellite_hardware.sensors import GPS
from ADCS.satellite_hardware.disturbances import SRP_Disturbance, Drag_Disturbance, Prop_Disturbance, Dipole_Disturbance, GG_Disturbance, GeometryConfig, GeometryFace
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import EarthConstants
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_helpers import random_n_unit_vec, rot_mat, norm, normalize
from ADCS.helpers.math_constants import MathConstants
from ADCS.estimators.attitude_SRUKF import SRUKF
from ADCS.estimators.attitude_UKF import UKF

def test_ukf():
    np.random.seed(1)

    t0 = 0
    tf = 60*10
    tlim00 = 5
    tlim0 = 0.5*60
    tlim1 = 2*60
    tlim2 = 4*60

    dt = 1
    N = int((tf-t0)/dt)
    np.set_printoptions(precision=6)

    ## REAL SATELLITE
    # Actuators: Magnetorquers
    mtq_noise = Noise(noise=0.0, std_noise=0.0001)
    mtq_noise = Noise(noise=0.0, std_noise=0.0)
    mtq_max_torque = 1.0
    acts = [MTQ(axis=j, max_torque=mtq_max_torque, noise=mtq_noise) for j in MathConstants.unitvecs]

    # Sensors: Magnetometers
    mtm_noise = Noise(noise=0.0, std_noise=1e-8)
    mtm_noise = Noise(noise=0.0, std_noise=0.0)
    mtms = [MTM(axis=j, noise=mtm_noise, scale=5e2) for j in MathConstants.unitvecs]

    # Sensors: Gyroscopes
    gyro_noise = Noise(noise=0.0, std_noise=0.0001)
    gyro_noise = Noise(noise=0.0, std_noise=0.0)
    gyros = [Gyro(axis=j, noise=gyro_noise) for j in MathConstants.unitvecs]

    # Sensors: SunPair
    sun_noise = Noise(noise=0.0, std_noise=0.0001)
    sun_noise = Noise(noise=0.0, std_noise=0.0)
    sun_eff = 1.0
    suns = [SunPair(axis=j, efficiency=sun_eff, noise=sun_noise) for j in MathConstants.unitvecs]

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
    w0 = random_n_unit_vec(3)*np.random.uniform(0, 1.0)*np.pi/180.0
    w0 = np.array([0, 0, 0])
    q0 = random_n_unit_vec(4)
    q0 = np.array([np.sqrt(2)/2,np.sqrt(2)/2,0,0])

    x = np.concatenate([w0, q0])
    ephem = Ephemeris()
    start_time = 0.22 - 1*TimeConstants.sec2cent
    end_time = 0.22 + (tf-t0)*TimeConstants.sec2cent
    R = 7000*np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([8, 0, 0])
    # Real Orbit Generation
    #os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
    #orb = Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=True, fast=False)

    # Quick Orbit Generation
    os0 = Orbital_State(ephem=ephem, J2000=0.22-1*TimeConstants.sec2cent, R=np.array([0, 1e5, 0]), V=np.array([1, 0, 0]), B=np.array([0.01, 0, 0]), S=np.array([0, 1e5+1, 0]), rho=1e-7)
    dur = int((tf-t0)/dt)+10
    orbs = [os0]*(dur+10)
    for j in range(dur):
        orbs[j] = os0.copy()
        orbs[j].J2000 = os0.J2000 + j*dt*TimeConstants.sec2cent
    orb = Orbit(orbs)

    ## ESTIMATED SATELLITE
    # Actuators
    est_acts = [MTQ(axis=j, max_torque=mtq_max_torque, noise=mtq_noise) for j in MathConstants.unitvecs]
    # Sensors
    est_mtms = [MTM(axis=j, noise=mtm_noise, scale=5e2) for j in MathConstants.unitvecs]
    est_gyros = [Gyro(axis=j, noise=gyro_noise) for j in MathConstants.unitvecs]
    est_suns = [SunPair(axis=j, efficiency=sun_eff, noise=sun_noise) for j in MathConstants.unitvecs]
    # Disturbances
    est_drag_dist = Drag_Disturbance(config=config)
    est_gg_dist = GG_Disturbance()
    est_dists = [est_drag_dist, est_gg_dist]

    # Satellite configuration
    est_sat_mass = 4.0
    est_sat_J = np.diagflat([3.4, 2.9, 1.3])
    est_sat = EstimatedSatellite(mass=est_sat_mass, J_0=est_sat_J, actuators=est_acts, sensors=est_mtms+est_gyros+est_suns, disturbances=est_dists)

    # Initial Estimated State
    x_hat = np.zeros(7)
    x_hat[3] = 1
    P_est = block_diag(np.eye(3)*(0.01)**2.0, np.eye(3)*3)
    Q_est = block_diag(np.eye(3)*(1e-4)**2.0, 1e-4*np.eye(3))

    ## Build Estimator
    J2000 = 0.22 + t0*TimeConstants.sec2cent
    srukf = UKF(est_sat=est_sat, J2000=J2000, x_hat=x_hat, P_hat=P_est, Q_hat=Q_est, dt=dt, cross_term=True, quat_as_vec=False)

    # Create history vectors
    time_hist = np.nan*np.zeros(N)
    state_hist = np.nan*np.zeros((N, srukf.state_len))
    est_state_hist = np.nan*np.zeros((N, srukf.state_len))
    os_hist: List[Orbital_State] = list()
    u_hist = np.nan*np.zeros((N, len(acts)))
    cov_hist: List[np.ndarray] = list()
    
    t = t0
    ind = 0
    
    while t < tf:
        # One Step Propagation
        J2000 = 0.22 + t*TimeConstants.sec2cent
        os = orb.get_os(J2000=J2000)

        # Determine control
        u = np.zeros(len(acts))

        noisy_sensor_readings = real_sat.sensor_readings(x=x, os=os)
        clean_sensor_readings = real_sat.noiseless_sensor_readings(x=x, os=os)
        x_hat = srukf.update(u=u, sensors=noisy_sensor_readings, os=os)

        # Full State Debug
        print("Real State ", x)
        print("Estimated State ", x_hat)

        # Attitude Debug
        quaternion_error_deg = (180.0/np.pi)*np.acos(-1 + 2*np.clip(np.dot(x_hat[3:7], x[3:7]), -1, 1)**2.0)
        print("Attitude Error (Degrees) ", quaternion_error_deg)
        angular_velocity_error = norm(x_hat[0:3] - x[0:3])*180.0/np.pi
        print("Angular Velocity Error ", angular_velocity_error)
        diagonal_covariances = np.diagonal(srukf.x_hat.cov)
        print("Attitude Covariance ", diagonal_covariances[3:6])
        print("Angular Velocity Covariance ", diagonal_covariances[0:3])
        print("")

        # Save Information for Plotting
        time_hist[ind] = t
        state_hist[ind,:] = x
        est_state_hist[ind,:] = x_hat
        os_hist += [os]
        u_hist[ind,:] = u
        cov_hist += [srukf.x_hat.cov]

        # Propagate
        ind += 1
        t += dt
        prev_os = os.copy()
        os = orb.get_os(0.22+(t-t0)*TimeConstants.sec2cent)

        out = solve_ivp(fun=real_sat.dynamics_for_solver, t_span=(0, dt), y0=x, method="RK45", args=(u, prev_os, os), rtol=1e-7, atol=1e-7)

        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])



    
if __name__ == "__main__":
    test_ukf()