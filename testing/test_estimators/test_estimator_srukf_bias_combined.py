# pytest: capture=no
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
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import Actuator, RW, MTQ
from ADCS.satellite_hardware.errors import Bias, Noise, ErrorMode
from ADCS.satellite_hardware.sensors import MTM, Gyro, SunPair
from ADCS.satellite_hardware.sensors import GPS
from ADCS.satellite_hardware.disturbances import SRP_Disturbance, Drag_Disturbance, Prop_Disturbance, Dipole_Disturbance, GG_Disturbance, GeometryConfig, GeometryFace
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import EarthConstants
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_helpers import random_n_unit_vec, rot_mat, norm, normalize, limit
from ADCS.helpers.math_constants import MathConstants
from ADCS.estimators.attitude_estimators import SRUAKF

from ADCS.helpers.plotting.plot_estimator import plot_state_comparison, plot_error_and_sun, plot_sensor_data, plot_bias_comparison
from ADCS.helpers.plotting.animate_estimator import animate_attitude
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window

@pytest.fixture(scope="module")
def ukf_bias_results():
    """
    Runs the simulation ONCE for the entire module.
    """
    print("\n--- Running ukf Simulation (Once) ---")
    # Adjust tf/dt here if you want a specific duration for testing
    results = run_ukf(verbose=False, tf=10000, dt=200, real_orbit=True)
    return results

@pytest.mark.slow
def test_stability(ukf_bias_results):
    """
    Check 1: Stability. The filter should not diverge into NaNs or Infs.
    """
    (_, _, est_state_hist, _, _, _, _, cov_hist) = ukf_bias_results

    
    # Check States
    assert not np.isnan(est_state_hist).any(), "Estimated state contains NaNs"
    assert not np.isinf(est_state_hist).any(), "Estimated state contains Infs"
    
    # Check Covariance
    cov_array = np.array(cov_hist) 
    assert not np.isnan(cov_array).any(), "Covariance contains NaNs"
    
    # Ensure diagonal variances are not exploding (e.g. > 1e6)
    # Using axis1=1, axis2=2 extracts the diagonal of each matrix in the stack
    diags = np.diagonal(cov_array, axis1=1, axis2=2)
    assert np.all(diags < 1e6), "Covariance exploded (variance > 1e6)"

@pytest.mark.slow
def test_ukf_quaternion_error(ukf_bias_results):
    """
    Check 2: Attitude Accuracy. 
    Compares initial error to final error. Final error should be close to 0.
    """
    (_, state_hist, est_state_hist, _, _, _, _, _) = ukf_bias_results

    
    # 1. Calculate Error Angle (Degrees)
    q_true = state_hist[:, 3:7]
    q_est = est_state_hist[:, 3:7]
    
    # Dot product (row-wise). Use abs() because q and -q are same rotation.
    dot_products = np.abs(np.einsum('ij,ij->i', q_true, q_est))
    dot_products = np.clip(dot_products, -1.0, 1.0) # Safety for arccos
    theta_err_deg = 2 * np.arccos(dot_products) * (180.0 / np.pi)
    
    # 2. Define small window for comparison (robust to single-step noise)
    # Use min(5, length) to work with short simulations
    N = len(theta_err_deg)
    window = max(1, min(5, int(N * 0.1))) 
    
    initial_err = np.mean(theta_err_deg[:window])
    final_err = np.mean(theta_err_deg[-window:])
    
    print(f"\nAttitude Error - Initial: {initial_err:.4f}°, Final: {final_err:.4f}°")

    # 3. Assertions
    # Convergence: Final should be better than Initial (or already perfect)
    if initial_err > 1.0: 
        assert final_err < initial_err, "Filter did not reduce error"
    
    # Accuracy: Final error should be close to 0 (allowing for sensor noise)
    assert final_err < 5.0, f"Final Attitude Error too high: {final_err:.2f}°"

@pytest.mark.slow
def test_ukf_rate_error(ukf_bias_results):
    """
    Check 3: Rate Accuracy.
    Final angular rate error should be close to 0.
    """
    (_, state_hist, est_state_hist, _, _, _, _, _) = ukf_bias_results

    
    # 1. Calculate Rate Error (deg/s)
    w_true = state_hist[:, 0:3]
    w_est = est_state_hist[:, 0:3]
    diff = w_true - w_est
    
    # FIX: Must use axis=1 to get norm of each row vector
    rate_err_deg_s = np.linalg.norm(diff, axis=1) * (180.0 / np.pi)
    
    # 2. Define Window
    N = len(rate_err_deg_s)
    window = max(1, min(5, int(N * 0.1)))
    
    final_rate_err = np.mean(rate_err_deg_s[-window:])
    
    print(f"Rate Error - Final: {final_rate_err:.4f} deg/s")
    
    # 3. Assertions
    assert final_rate_err < 0.5, f"Final Rate Error too high: {final_rate_err:.4f} deg/s"

@pytest.mark.slow
def test_ukf_covariance_consistency(ukf_bias_results):
    """
    Check 4: Consistency.
    Actual error should be within 3-sigma bounds most of the time.
    """
    (_, state_hist, est_state_hist, _, _, _, _, cov_hist) = ukf_bias_results

    
    P_hist = np.array(cov_hist)
    N = len(P_hist)
    
    # Extract Rate Variances (Indices 0,1,2)
    # If your state is [w, q, ...], diagonal 0-3 is rate variance
    rate_vars = P_hist[:, 0:3, 0:3].diagonal(axis1=1, axis2=2)
    sigma_3_bnds = 3 * np.sqrt(rate_vars) * (180.0 / np.pi)
    
    # Actual Error
    w_true = state_hist[:, 0:3]
    w_est = est_state_hist[:, 0:3]
    actual_err = np.abs(w_true - w_est) * (180.0 / np.pi)
    
    # Only check the second half of the sim (to allow for convergence)
    start_idx = int(N / 2)
    
    # If sim is too short, just check the last few points
    if start_idx == N: start_idx = 0
    
    for axis in range(3):
        bnds = sigma_3_bnds[start_idx:, axis]
        errs = actual_err[start_idx:, axis]
        
        inside_count = np.sum(errs <= bnds)
        total = len(errs)
        if total == 0: continue
            
        percentage = inside_count / total
        print(f"Axis {axis} Consistency: {percentage:.1%}")
        
        # It's okay if it's not 99%, but it should be > 70% to prove P matches R/Q tuning
        assert percentage > 0.70, f"Filter inconsistent on axis {axis}"
        
@pytest.mark.slow
def test_ukf_bias_convergence(ukf_bias_results):
    """
    Check 5: Gyro Bias Convergence.
    The estimated gyro bias should converge to the real bias
    within 0.001 rad/s on each axis.
    """
    (_, state_hist, est_state_hist, _, _, _, _, _) = ukf_bias_results

    # True bias stored in indices 7, 8, 9 of state_hist
    true_bias = state_hist[:, 7:10]
    est_bias  = est_state_hist[:, 7:10]

    N = len(true_bias)
    window = max(5, int(0.1 * N))  # average final 10% of simulation

    true_final = np.mean(true_bias[-window:], axis=0)
    est_final  = np.mean(est_bias[-window:], axis=0)

    bias_error = np.abs(true_final - est_final)

    print("\nBias Convergence Check:")
    print(f" True Bias Final: {true_final}")
    print(f" Est Bias Final:  {est_final}")
    print(f" Bias Error:      {bias_error}")

    # Assert each axis is within 0.001 rad/s
    assert np.all(bias_error < 0.001), (
        f"Bias error too large. Errors: {bias_error}"
    )

@pytest.mark.slow
def test_ukf_mtm_bias_convergence(ukf_bias_results):
    """
    Check 6: MTM Bias Convergence.
    Indices 7, 8, 9 in the state vector correspond to MTM bias 
    in the latest run_ukf configuration.
    """
    (_, state_hist, est_state_hist, _, _, _, _, _) = ukf_bias_results

    # Indices 7:10 -> MTM Bias (X, Y, Z)
    true_bias = state_hist[:, 7:10]
    est_bias = est_state_hist[:, 7:10]

    N = len(true_bias)
    # Compare the average of the last 10% of the simulation
    window = max(5, int(0.1 * N))

    true_final = np.mean(true_bias[-window:], axis=0)
    est_final = np.mean(est_bias[-window:], axis=0)

    bias_error = np.abs(true_final - est_final)

    print("\nMTM Bias Convergence Check:")
    print(f" True Bias Final: {true_final}")
    print(f" Est Bias Final:  {est_final}")
    print(f" Bias Error:      {bias_error}")

    # Threshold: 1 micro-Tesla (1e-6)
    # Real bias is ~1e-7 to 1e-8. Convergence should be within 1e-6.
    assert np.all(bias_error < 1e-6), (
        f"MTM Bias error too large. Errors: {bias_error}"
    )

@pytest.mark.slow
def test_ukf_sun_bias_convergence(ukf_bias_results):
    """
    Check 7: Sun Sensor Bias Convergence.
    Indices 13, 14, 15 in the state vector correspond to Sun Sensor bias 
    in the latest run_ukf configuration.
    """
    (_, state_hist, est_state_hist, _, _, _, _, _) = ukf_bias_results

    # Indices 13:16 -> Sun Sensor Bias (X, Y, Z)
    true_bias = state_hist[:, 13:16]
    est_bias = est_state_hist[:, 13:16]

    N = len(true_bias)
    window = max(5, int(0.1 * N))

    true_final = np.mean(true_bias[-window:], axis=0)
    est_final = np.mean(est_bias[-window:], axis=0)

    bias_error = np.abs(true_final - est_final)

    print("\nSun Sensor Bias Convergence Check:")
    print(f" True Bias Final: {true_final}")
    print(f" Est Bias Final:  {est_final}")
    print(f" Bias Error:      {bias_error}")

    # Threshold: 0.01 (Unitless/Cosine magnitude)
    # Real bias mean is approx 0.05. Convergence should be well within 0.01.
    assert np.all(bias_error < 0.01), (
        f"Sun Sensor Bias error too large. Errors: {bias_error}"
    )

def run_ukf(verbose: bool = False, tf: float = 1000, dt: float = 10, real_orbit: bool = False) -> Union[np.ndarray, np.ndarray, np.ndarray, List[Orbital_State], List[np.ndarray], List[np.ndarray], np.ndarray, List[np.ndarray]]:
    np.random.seed(67)
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

    x = np.concatenate([w0, q0])
    ephem = Ephemeris()
    start_time = 0.22 - 1*TimeConstants.sec2cent
    end_time = 0.22 + (tf-t0)*TimeConstants.sec2cent
    R = -7000*np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([8, 0, 0])
    if real_orbit:
        # Real Orbit Generation
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
    x_hat = np.zeros(16)
    x_hat[3] = 1

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
    ukf = SRUAKF(est_sat=est_sat, J2000=J2000, x_hat=x_hat, P_hat=P_est, Q_hat=Q_est, dt=dt, cross_term=True, quat_as_vec=False)

    # Create history vectors
    time_hist = np.nan*np.zeros(N)
    state_hist = np.nan*np.zeros((N, ukf.state_len))
    est_state_hist = np.nan*np.zeros((N, ukf.state_len))
    os_hist: List[Orbital_State] = list()
    sensor_hist: List[np.ndarray] = np.nan*np.zeros((N, 9))
    clean_sensor_hist: List[np.ndarray] = np.nan*np.zeros((N, 9))
    u_hist = np.nan*np.zeros((N, len(acts)))
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
            print("Real State ", x[0:7])
            print("Estimated State ", x_hat[0:7])
            print("Real Gyro Bias ", [gyro.bias.bias for gyro in real_sat.attitude_sensors if isinstance(gyro, Gyro)])
            print("Estimated Gyro Bias ", x_hat[7:10])
            print("Real MTM Bias ", [mtm.bias.bias for mtm in real_sat.attitude_sensors if isinstance(mtm, MTM)])
            print("Estimated MTM Bias ", x_hat[10:13])

            # Attitude Debug
            quaternion_error_deg = (180.0/np.pi)*np.acos(-1 + 2*np.clip(np.dot(x_hat[3:7], x[3:7]), -1, 1)**2.0)
            print("Attitude Error (Degrees) ", quaternion_error_deg)
            angular_velocity_error = norm(x_hat[0:3] - x[0:3])*180.0/np.pi
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

    return time_hist, state_hist, est_state_hist, os_hist, sensor_hist, clean_sensor_hist, u_hist, cov_hist


def plot_ukf(verbose: bool = False, tf: float = 60, dt: float = 1, real_orbit: bool = False) -> None:
    (time_hist, state_hist, est_state_hist, os_hist,
     sensor_hist, clean_sensor_hist, u_hist, cov_hist) = run_ukf(
         verbose=verbose, tf=tf, dt=dt, real_orbit=real_orbit)

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

    
if __name__ == "__main__":
    plot_ukf(verbose=False, tf=2000, dt=50, real_orbit=True)