import sys
import os
import numpy as np
import pytest
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag
from tqdm import tqdm

# === Import project modules ===
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import Actuator, RW, MTQ
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.sensors import MTM, Gyro, SunPair
from ADCS.satellite_hardware.disturbances import Drag_Disturbance, GG_Disturbance, GeometryConfig, GeometryFace
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_helpers import random_n_unit_vec, norm, normalize
from ADCS.helpers.math_constants import MathConstants
from ADCS.estimators.attitude_estimators.attitude_UAKF import UAKF

from ADCS.helpers.plotting.plot_estimator import plot_state_comparison, plot_error_and_sun, plot_sensor_data
from ADCS.helpers.plotting.animate_estimator import animate_attitude
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window

# ==========================================
#      HELPER FUNCTIONS (Satellite Setup)
# ==========================================

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

    faces: List[GeometryFace] = [
        GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[0]*0.05, normal=MathConstants.unitvecs[0], CD=2.2),
        GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[0]*0.05, normal=-MathConstants.unitvecs[0], CD=2.2),
        GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[1]*0.05, normal=MathConstants.unitvecs[1], CD=2.2),
        GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[1]*0.05, normal=-MathConstants.unitvecs[1], CD=2.2),
        GeometryFace(area=0.1*0.1, centroid=MathConstants.unitvecs[2]*0.15, normal=MathConstants.unitvecs[2], CD=2.2),
        GeometryFace(area=0.1*0.1, centroid=-MathConstants.unitvecs[2]*0.15, normal=-MathConstants.unitvecs[2], CD=2.2)
    ]
    config = GeometryConfig(geometry_faces=faces)
    dists = [Drag_Disturbance(config=config), GG_Disturbance()]

    real_sat = Satellite(
        mass=4.0, 
        J_0=np.diagflat([3.4, 2.9, 1.3]), 
        actuators=mtqs+rws, 
        sensors=mtms+gyros+suns, 
        disturbances=dists
    )
    return real_sat

def create_estimated_satellite() -> EstimatedSatellite:
    # Similar setup but for the estimator model
    mtq_noise = Noise(noise=0.0, std_noise=0.0001)
    est_mtqs = [MTQ(axis=j, max_torque=1.0, noise=mtq_noise) for j in MathConstants.unitvecs]

    rw_noise = Noise(noise=0.0, std_noise=0.0001)
    rw_h_noise = Noise(noise=0.0, std_noise=0.0001)
    est_rws = [RW(axis=j, max_torque=4.51, J=0.22, h=1, h_max=3.8, noise=rw_noise, h_meas_noise=rw_h_noise) for j in MathConstants.unitvecs]

    mtm_noise = Noise(noise=0.0, std_noise=1e-8)
    est_mtms = [MTM(axis=MathConstants.unitvecs[j], noise=mtm_noise) for j in range(3)]

    gyro_noise = Noise(noise=0.0, std_noise=0.0001)
    est_gyros = [Gyro(axis=MathConstants.unitvecs[j], noise=gyro_noise) for j in range(3)]

    sun_noise = Noise(noise=0.0, std_noise=0.0001)
    est_suns = [SunPair(axis=MathConstants.unitvecs[j], efficiency=0.3, noise=sun_noise) for j in range(3)]

    faces = [
        GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[0]*0.05, normal=MathConstants.unitvecs[0], CD=2.2),
        GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[0]*0.05, normal=-MathConstants.unitvecs[0], CD=2.2),
        GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[1]*0.05, normal=MathConstants.unitvecs[1], CD=2.2),
        GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[1]*0.05, normal=-MathConstants.unitvecs[1], CD=2.2),
        GeometryFace(area=0.1*0.1, centroid=MathConstants.unitvecs[2]*0.15, normal=MathConstants.unitvecs[2], CD=2.2),
        GeometryFace(area=0.1*0.1, centroid=-MathConstants.unitvecs[2]*0.15, normal=-MathConstants.unitvecs[2], CD=2.2)
    ]
    config = GeometryConfig(geometry_faces=faces)
    
    est_sat = EstimatedSatellite(
        mass=4.0, 
        J_0=np.diagflat([3.4, 2.9, 1.3]), 
        actuators=est_mtqs+est_rws, 
        sensors=est_mtms+est_gyros+est_suns, 
        disturbances=[Drag_Disturbance(config=config), GG_Disturbance()]
    )
    return est_sat

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
    
    # RW Noise Covariance
    Q_rw  = np.eye(3) * (0.0001)**2.0 * dt
    Q_est = block_diag(Q_dyn_block, Q_rw)

    # Initial Covariance Guess
    # 0:3 (Omega), 3:6 (Attitude Error), 6:9 (RW Momentum)
    P_est = block_diag(np.eye(3)*(0.01)**2.0, np.eye(3)*0.1, np.eye(3)*0.2)

    return P_est, Q_est

def create_orbit(t0: float, tf: float, dt: float, real_orbit: bool = True) -> Orbit:
    ephem = Ephemeris()
    start_time = 0.22 - 1*TimeConstants.sec2cent
    end_time = 0.22 + (tf-t0)*TimeConstants.sec2cent
    R = -7000*np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([8, 0, 0])
    
    if real_orbit:
        os = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
        orb = Orbit(os0=os, end_time=end_time, dt=dt, use_J2=True, fast=False)
    else:
        # Simplified static orbit for faster testing if needed
        os = Orbital_State(ephem=ephem, J2000=0.22-1*TimeConstants.sec2cent, R=R, V=V, B=np.array([0, 0.1, 0]), S=np.array([1e5+1, 0, 0]), rho=5e-12)
        dur = int((tf-t0)/dt)+10
        orbs = [os]*(dur+10)
        for j in range(dur):
            orbs[j] = os.copy()
            orbs[j].J2000 = os.J2000 + j*dt*TimeConstants.sec2cent
        orb = Orbit(orbs)
    return orb

# ==========================================
#           SIMULATION RUNNER
# ==========================================

def run_rw_ukf(verbose: bool = False, tf: float = 300, dt: float = 5, real_orbit: bool = True):
    np.random.seed(1)
    t0 = 0
    N = int((tf-t0)/dt)
    
    # 1. Setup Hardware & Estimator
    real_sat = create_satellite()
    est_sat = create_estimated_satellite()
    orb = create_orbit(t0, tf, dt, real_orbit)

    # 2. Initial State: [w(3), q(4), h_rw(3)]
    w0 = random_n_unit_vec(3)*np.random.uniform(0, 0.1)*np.pi/180.0
    q0 = random_n_unit_vec(4)
    h0 = np.array([1, 1, 1]) 
    x = np.concatenate([w0, q0, h0])

    # 3. Initial Estimation Guess
    # [w(3), q(4), h_rw(3)]
    x_hat = np.array([0, 0, 0, 1, 0, 0, 0, 0.1, 0.1, 0.1]) 

    # 4. Build Estimator (SRUAKF)
    J2000 = 0.22 + t0*TimeConstants.sec2cent
    P_est, Q_est = create_matrices(est_sat=est_sat, dt=dt)
    
    # Note: State len is 10, Error State len is 9
    ukf = UAKF(est_sat=est_sat, J2000=J2000, x_hat=x_hat, P_hat=P_est, Q_hat=Q_est, dt=dt, cross_term=True)

    # 5. History Arrays
    time_hist = np.nan*np.zeros(N)
    state_hist = np.nan*np.zeros((N, ukf.state_len)) # Size 10
    est_state_hist = np.nan*np.zeros((N, ukf.state_len)) # Size 10
    os_hist: List[Orbital_State] = list()
    # Sensors: MTM(3) + Gyro(3) + Sun(3*2??) -> 12 based on source file 2
    sensor_hist = np.nan*np.zeros((N, 12)) 
    clean_sensor_hist = np.nan*np.zeros((N, 12))
    u_hist = np.nan*np.zeros((N, 6)) # 3 MTQ + 3 RW
    cov_hist: List[np.ndarray] = list()
    
    t = t0
    ind = 0
    
    steps = int((tf - t0)/dt)

    for _ in tqdm(range(steps), desc="Simulating RW SRUAKF"):
        # --- Propagate Environment ---
        J2000 = 0.22 + t*TimeConstants.sec2cent
        os = orb.get_os(J2000=J2000)

        # --- Control (None for this test) ---
        u = np.zeros(6) 

        # --- Readings & Update ---
        noisy_sensor_readings = real_sat.sensor_readings(x=x, os=os)
        clean_sensor_readings = real_sat.noiseless_sensor_readings(x=x, os=os)
        
        x_hat = ukf.update(u=u, sensors=noisy_sensor_readings, os=os)

        # --- Debug Printing ---
        if verbose:
            print(f"Time: {t}")
            print("Real State: ", x)
            print("Est State:  ", x_hat)
            quaternion_error_deg = (180.0/np.pi)*np.acos(-1 + 2*np.clip(np.dot(x_hat[3:7], x[3:7]), -1, 1)**2.0)
            print(f"Att Err: {quaternion_error_deg:.4f} deg")
            print("")

        # --- Save History ---
        time_hist[ind] = t
        state_hist[ind,:] = x
        est_state_hist[ind,:] = x_hat
        os_hist.append(os)
        sensor_hist[ind,:] = noisy_sensor_readings
        clean_sensor_hist[ind,:] = clean_sensor_readings
        u_hist[ind,:] = u
        cov_hist.append(ukf.x_hat.cov)

        # --- Propagate Dynamics ---
        ind += 1
        t += dt
        prev_os = os.copy()
        os = orb.get_os(0.22+(t-t0)*TimeConstants.sec2cent)

        # Solve dynamics
        out = solve_ivp(fun=real_sat.dynamics_for_solver, t_span=(0, dt), y0=x, method="RK45", 
                        args=(u, prev_os, os), rtol=1e-7, atol=1e-7)

        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7]) # Re-normalize quaternion

    return time_hist, state_hist, est_state_hist, os_hist, sensor_hist, clean_sensor_hist, u_hist, cov_hist

# ==========================================
#               PYTEST FIXTURES
# ==========================================

@pytest.fixture(scope="module")
def rw_ukf_results():
    """
    Runs the RW simulation ONCE for the entire module.
    """
    print("\n--- Running RW SRUAKF Simulation (Once) ---")
    # Using 1000s duration to allow convergence of momentum bias
    results = run_rw_ukf(verbose=False, tf=200, dt=10, real_orbit=True)
    return results

# ==========================================
#                 TESTS
# ==========================================

@pytest.mark.slow
def test_rw_stability(rw_ukf_results):
    """
    Check 1: Stability. 
    """
    (_, _, est_state_hist, _, _, _, _, cov_hist) = rw_ukf_results
    
    # Check States
    assert not np.isnan(est_state_hist).any(), "Estimated state contains NaNs"
    assert not np.isinf(est_state_hist).any(), "Estimated state contains Infs"
    
    # Check Covariance
    cov_array = np.array(cov_hist) 
    assert not np.isnan(cov_array).any(), "Covariance contains NaNs"
    
    diags = np.diagonal(cov_array, axis1=1, axis2=2)
    assert np.all(diags < 1e7), "Covariance exploded"

@pytest.mark.slow
def test_rw_attitude_error(rw_ukf_results):
    """
    Check 2: Attitude Accuracy (Quaternions).
    """
    (_, state_hist, est_state_hist, _, _, _, _, _) = rw_ukf_results
    
    # Quaternions are indices 3:7
    q_true = state_hist[:, 3:7]
    q_est = est_state_hist[:, 3:7]
    
    dot_products = np.abs(np.einsum('ij,ij->i', q_true, q_est))
    dot_products = np.clip(dot_products, -1.0, 1.0)
    theta_err_deg = 2 * np.arccos(dot_products) * (180.0 / np.pi)
    
    N = len(theta_err_deg)
    window = max(1, min(10, int(N * 0.1))) 
    
    initial_err = np.mean(theta_err_deg[:window])
    final_err = np.mean(theta_err_deg[-window:])
    
    print(f"\n[RW] Attitude Error - Initial: {initial_err:.4f}°, Final: {final_err:.4f}°")

    if initial_err > 1.0: 
        assert final_err < initial_err, "Filter did not reduce attitude error"
    
    assert final_err < 5.0, f"Final Attitude Error too high: {final_err:.2f}°"

@pytest.mark.slow
def test_rw_momentum_estimation(rw_ukf_results):
    """
    Check 3: Reaction Wheel Momentum Estimation.
    This is specific to the RW test. Indices 7:10.
    """
    (_, state_hist, est_state_hist, _, _, _, _, _) = rw_ukf_results

    # RW Momentum indices are 7:10
    h_true = state_hist[:, 7:10]
    h_est = est_state_hist[:, 7:10]

    diff = h_true - h_est
    h_err_norm = np.linalg.norm(diff, axis=1)

    N = len(h_err_norm)
    window = max(1, min(10, int(N * 0.1))) 

    initial_err = np.mean(h_err_norm[:window])
    final_err = np.mean(h_err_norm[-window:])

    print(f"[RW] Momentum Error - Initial: {initial_err:.4f} Nms, Final: {final_err:.4f} Nms")

    # The filter initializes with [0.1, 0.1, 0.1] but truth is [1, 1, 1].
    # Error should decrease significantly.
    assert final_err < initial_err, "Momentum estimation did not converge"
    assert final_err < 0.2, f"Final Momentum Error too high: {final_err:.4f} Nms"

@pytest.mark.slow
def test_rw_covariance_consistency(rw_ukf_results):
    """
    Check 4: Consistency.
    Actual error should be within 3-sigma bounds.
    """
    (_, state_hist, est_state_hist, _, _, _, _, cov_hist) = rw_ukf_results
    
    P_hist = np.array(cov_hist)
    N = len(P_hist)
    
    # --- Check Rate Consistency (Indices 0:3) ---
    # Rate Variance in P is usually the top-left 3x3 block (indices 0:3)
    rate_vars = P_hist[:, 0:3, 0:3].diagonal(axis1=1, axis2=2)
    sigma_3_bnds = 3 * np.sqrt(rate_vars) * (180.0 / np.pi)
    
    w_true = state_hist[:, 0:3]
    w_est = est_state_hist[:, 0:3]
    actual_err = np.abs(w_true - w_est) * (180.0 / np.pi)
    
    start_idx = int(N / 2) # Check second half
    
    for axis in range(3):
        bnds = sigma_3_bnds[start_idx:, axis]
        errs = actual_err[start_idx:, axis]
        
        inside_count = np.sum(errs <= bnds)
        total = len(errs)
        if total == 0: continue
            
        percentage = inside_count / total
        print(f"[RW] Rate Axis {axis} Consistency: {percentage:.1%}")
        
        assert percentage > 0.70, f"Filter inconsistent on rate axis {axis}"

# ==========================================
#           MANUAL EXECUTION
# ==========================================

def plot_ukf_rw(verbose: bool = False, tf: float = 300, dt: float = 5, real_orbit: bool = True) -> None:
    results = run_rw_ukf(verbose=verbose, tf=tf, dt=dt, real_orbit=real_orbit)
    
    (time_hist, state_hist, est_state_hist, os_hist,
     sensor_hist, clean_sensor_hist, u_hist, cov_hist) = results

    plot_state_comparison(time_hist, state_hist, est_state_hist)
    plot_error_and_sun(time_hist, state_hist, est_state_hist, os_hist)
    plot_sensor_data(time_hist, sensor_hist, clean_sensor_hist)
    
    # We can also animate
    animate_attitude(time_hist, state_hist, est_state_hist, os_hist)
    
    create_close_all_button_window()

if __name__ == "__main__":
    # You can run this file directly to visualize the results
    plot_ukf_rw(verbose=False, tf=200, dt=10, real_orbit=True)