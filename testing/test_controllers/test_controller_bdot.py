import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Union, Tuple, Optional
from tqdm import tqdm
import pytest
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from ADCS.controller import BDot
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import random_n_unit_vec, normalize

from ADCS.helpers.plotting.animate_estimator import animate_attitude
from ADCS.helpers.plotting.plot_estimator import plot_state_comparison
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window
from ADCS.helpers.plotting.plot_controller import plot_control
from ADCS.helpers.plotting.animate_orbit import animate_orbit

def run_bdot_simulation(
    verbose: bool = False, 
    tf: float = 1000, 
    dt: float = 1, 
    real_orbit: bool = False,
    include_rw: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[Orbital_State], np.ndarray, np.ndarray]:
    """
    Runs the B-Dot simulation.
    """
    np.random.seed(1)
    t0 = 0
    N = int((tf-t0)/dt)

    # --- Hardware Setup ---
    # Magnetorquers (always present for B-Dot)
    mtm_max_torque = 0.1 
    if include_rw:
        mtm_max_torque = 0.01 
        
    mtqs = [MTQ(axis=j, max_torque=mtm_max_torque) for j in MathConstants.unitvecs]
    acts = list(mtqs)
    
    # Reaction Wheels (Optional)
    if include_rw:
        rw_max_torque = 4.51
        rw_J = 0.22
        rw_h0 = 0
        rw_hmax = 3.8
        rws = [RW(axis=j, max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax) for j in MathConstants.unitvecs]
        acts += rws

    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    real_sat = Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]), actuators=acts, sensors=mtms)

    # --- Initial Conditions ---
    w0 = random_n_unit_vec(3) * np.random.uniform(1, 2) * np.pi / 180.0
    q0 = random_n_unit_vec(4)
    
    if include_rw:
        h0 = np.array([rw_h0, rw_h0, rw_h0])
        x = np.concatenate([w0, q0, h0])
        state_dim = 10
    else:
        x = np.concatenate([w0, q0])
        state_dim = 7

    # --- Orbit Generation ---
    ephem = Ephemeris()
    start_time_j2000 = 0.22
    
    if real_orbit:
        start_date = start_time_j2000 - 1 * TimeConstants.sec2cent
        end_date = start_time_j2000 + (tf - t0) * TimeConstants.sec2cent
        R = 7000 * np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
        V = np.array([8, 0, 0])
        os0 = Orbital_State(ephem=ephem, J2000=start_date, R=R, V=V)
        orb = Orbit(os0=os0, end_time=end_date, dt=dt, use_J2=True, fast=False)
    else:
        # Static magnetic field case for easier debugging
        R = 7000 * np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
        V = np.array([8, 0, 0])
        # Constant B-field simplifies "rotation about B-field" checks
        os0 = Orbital_State(ephem=ephem, J2000=start_time_j2000, R=R, V=V, 
                            B=np.array([0, 0.1, 0]), S=np.array([1e5+1, 0, 0]), rho=5e-12)
        
        dur = int((tf-t0)/dt) + 10
        orbs = [os0] * (dur + 10)
        for j in range(dur):
            temp_os = os0.copy()
            temp_os.J2000 = os0.J2000 + j * dt * TimeConstants.sec2cent
            orbs[j] = temp_os
        orb = Orbit(orbs)

    # --- Controller ---
    controller = BDot(est_sat=real_sat, gain=100)

    # --- Data Logging ---
    time_hist = np.nan * np.zeros(N)
    state_hist = np.nan * np.zeros((N, state_dim))
    os_hist: List[Orbital_State] = list()
    sensor_hist: np.ndarray = np.nan * np.zeros((N, len(real_sat.sensors + real_sat.rw_actuators)))
    u_hist = np.nan * np.zeros((N, len(acts)))

    # --- Simulation Loop ---
    t = t0
    ind = 0
    steps = int((tf - t0) / dt)
    
    for step in tqdm(range(steps), desc=f"Simulating BDot (RW={include_rw})"):
        current_j2000 = start_time_j2000 + t * TimeConstants.sec2cent
        os = orb.get_os(J2000=current_j2000)

        sens = real_sat.sensor_readings(x=x, os=os)
        u = controller.find_u(x_hat=x, sens=sens, est_sat=real_sat, os_hat=os)

        if verbose:
            print("u: ", u)

        time_hist[ind] = t
        state_hist[ind, :] = x
        os_hist.append(os)
        sensor_hist[ind, :] = sens
        u_hist[ind, :] = u

        # Integration
        ind += 1
        t += dt
        prev_os = os.copy()
        next_os = orb.get_os(start_time_j2000 + t * TimeConstants.sec2cent)

        out = solve_ivp(
            fun=real_sat.dynamics_for_solver, 
            t_span=(0, dt), 
            y0=x, 
            method="RK45", 
            args=(u, prev_os, next_os), 
            rtol=1e-7, 
            atol=1e-7
        )
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7]) 

    return time_hist, state_hist, os_hist, sensor_hist, u_hist


@pytest.fixture(scope="module")
def bdot_results():
    """
    Runs the B-Dot simulation ONCE for the entire module.
    Returns: (time_hist, state_hist, os_hist, sensor_hist, u_hist)
    """
    print("\n--- Running B-Dot Simulation (Once) ---")
    results = run_bdot_simulation(
        verbose=False, 
        tf=500, 
        dt=1.0, 
        real_orbit=False, 
        include_rw=True
    )
    return results


def test_bdot_control_effort(bdot_results):
    """
    Test 1: Check that control effort for all MTQ channels is close to zero at the end.
    """
    (_, _, _, _, u_hist) = bdot_results

    # Analyze last 10 steps to account for minor noise
    final_u = np.mean(np.abs(u_hist[-10:]), axis=0)
    
    # Extract MTQ indices (first 3 actuators)
    mtq_effort = final_u[0:3] 
    
    print(f"\nFinal MTQ Effort (Mean Abs): {mtq_effort}")
    assert np.all(mtq_effort < 1e-3), f"Control effort did not settle. Final MTQ u: {mtq_effort}"

def test_bdot_rotational_dynamics(bdot_results):
    """
    Test 2: Check that only rotation about the axis of the magnetic field line exists.
    All rotation perpendicular to B-field should be close to zero.
    """
    (_, state_hist, os_hist, _, _) = bdot_results

    final_w = state_hist[-1, 0:3] # Body angular velocity
    final_B_eci = os_hist[-1].B   # B-field in ECI
    q_final = state_hist[-1, 3:7] # Quaternion (ECI -> Body)

    # Transform B-field from ECI to Body frame
    r = R.from_quat(q_final)
    final_B_body = r.apply(final_B_eci, inverse=True)
    
    # Calculate component of w perpendicular to B
    # w_perp = w - proj_B(w)
    B_unit = final_B_body / np.linalg.norm(final_B_body)
    w_parallel_mag = np.dot(final_w, B_unit)
    w_parallel = w_parallel_mag * B_unit
    w_perp = final_w - w_parallel
    
    w_perp_mag = np.linalg.norm(w_perp)
    
    print(f"\nResidual Perpendicular Rotation: {w_perp_mag:.2e} rad/s")
    assert w_perp_mag < 1e-2, f"Residual rotation perpendicular to B-field detected: {w_perp_mag} rad/s"


def debug_plots(time_hist, state_hist, os_hist, u_hist):
    """Helper to generate plots for debugging."""
    animate_attitude(time=time_hist, state_hist=state_hist, os_hist=os_hist)
    plot_control(time=time_hist, u_hist=u_hist)
    plot_state_comparison(time=time_hist, state_hist=state_hist)
    # animate_orbit(time_hist=time_hist, state_hist=state_hist, os_hist=os_hist)
    create_close_all_button_window()

if __name__ == "__main__":
    print("Running B-Dot Simulation for visual debugging...")
    
    # Configuration
    TF = 100
    DT = 1.0
    REAL_ORBIT = False 
    INCLUDE_RW = True
    
    t, x, os_h, sens, u = run_bdot_simulation(
        verbose=False, 
        tf=TF, 
        dt=DT, 
        real_orbit=REAL_ORBIT, 
        include_rw=INCLUDE_RW
    )
    
    debug_plots(t, x, os_h, u)