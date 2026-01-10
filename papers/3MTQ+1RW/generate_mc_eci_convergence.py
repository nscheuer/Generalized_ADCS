import sys
import os
import numpy as np
import concurrent.futures
import multiprocessing
from scipy.integrate import solve_ivp
from typing import List, Union, Dict, Any
from tqdm import tqdm

# --- Your Existing Imports ---
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from ADCS.CONOPS.goals import Goal, ECI_Goal, Coordinate_Goal
from ADCS.controller import MTQ_w_1RW
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import random_n_unit_vec, normalize
from ADCS.helpers.save_and_load.save_and_load import save_data, load_data
from ADCS.helpers.mc.monte_carlo_runner import MonteCarloRunner
from ADCS.helpers.plotting_mc.plot_controller_mc import plot_target_tracking_mc
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window

def run_single_sim(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function that runs one simulation instance based on the provided config.
    Returns full state histories.
    """
    # 1. Setup Randomness for this process
    np.random.seed(config["seed"])

    # 2. Unpack Config
    tf = config.get("tf", 500)
    dt = config.get("dt", 2)
    t0 = 0
    N = int((tf-t0)/dt)
    
    # 3. Hardware Setup (Preserving your logic)
    mtq_max_torque = 0.4
    mtqs = [MTQ(axis=j, max_torque=mtq_max_torque) for j in MathConstants.unitvecs]

    rw_max_torque = 7*0.001
    rw_J = 0.001
    rw_hmax = 16.2*0.001
    
    # Setup RWs and pop one as per your original code
    rws = [RW(axis=j, max_torque=rw_max_torque, J=rw_J, h=0, h_max=rw_hmax) for j in MathConstants.unitvecs]
    rws.pop()
    rws.pop()
    
    acts = mtqs + rws
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    real_sat = Satellite(mass=1.2, J_0=np.diagflat([0.022, 0.022, 0.004]), actuators=acts, sensors=mtms, boresight=np.array([0, 0, 1]))

    # 4. Initial Conditions (From Config)
    w0 = config["w0"]
    q0 = config["q0"]
    h0 = config["h0"] # Random initial momentum
    x = np.concatenate([w0, q0, h0])

    # Update RW objects with initial h (for consistency, though x vector drives dynamics)
    for i, rw in enumerate(rws):
        rw.h = h0[i]

    # 5. Orbit Setup
    ephem = Ephemeris()
    start_time = 0.22 - 1*TimeConstants.sec2cent
    # For MC speed, using simplified orbit propagator unless "real_orbit" is strictly required
    # Assuming J2/Basic propagator is sufficient for attitude control testing
    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=config["orbit_R"], V=config["orbit_V"])
    orb = Orbit(os0=os0, end_time=0.22 + (tf)*TimeConstants.sec2cent, dt=dt, use_J2=True, fast=False)

    # 6. Controller & Goal
    controller = MTQ_w_1RW(est_sat=real_sat, p_gain=0.00005, d_gain=0.001, c_gain=0.001, h_target=0.004)
    goal = ECI_Goal(config["goal_eci_vec"])

    # 7. Pre-allocate Arrays
    time_hist = np.zeros(N)
    state_hist = np.zeros((N, len(x)))
    u_hist = np.zeros((N, len(acts)))
    boresight_hist = np.zeros((N, 3))
    # Note: We skip os_hist/sensor_hist to save a bit of RAM, unless strictly needed.
    # If needed, uncomment below:
    # sensor_hist = np.zeros((N, len(real_sat.sensors + real_sat.rw_actuators)))

    t = t0
    ind = 0
    steps = int((tf - t0)/dt)

    # 8. Main Loop
    for _ in range(steps):
        J2000 = 0.22 + t*TimeConstants.sec2cent
        os = orb.get_os(J2000=J2000)

        sens = real_sat.sensor_readings(x=x, os=os)
        u = controller.find_u(x_hat=x, sens=sens, est_sat=real_sat, os_hat=os, goal=goal)

        # Record History
        time_hist[ind] = t
        state_hist[ind,:] = x
        u_hist[ind,:] = u
        
        # Calculate Boresight Error for recording
        eci_goal_ref, _ = goal.to_ref(os0=os)
        boresight_hist[ind, :] = eci_goal_ref

        ind += 1
        t += dt
        
        # Propagate
        prev_os = os.copy()
        os_next = orb.get_os(0.22+(t-t0)*TimeConstants.sec2cent)
        
        out = solve_ivp(fun=real_sat.dynamics_for_solver, t_span=(0, dt), y0=x, method="RK45", args=(u, prev_os, os_next), rtol=1e-6, atol=1e-6)
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])

    # 9. Return Packet
    return {
        "run_id": config["run_id"],
        "config": config, # Optional: pass back config if you want to correlate inputs/outputs later
        "time": time_hist,
        "state": state_hist,
        "u": u_hist,
        "boresight_goal": boresight_hist
    }

# --- Config Generator ---

def generate_mc_config(run_id: int) -> Dict[str, Any]:
    """
    Generates randomized start states and goals.
    """
    # Create a local RNG for reproducibility based on run_id
    rng = np.random.default_rng(seed=run_id)
    
    # Randomize Initial Attitude (Quaternion)
    q0 = normalize(rng.standard_normal(4))
    
    # Randomize Initial Omega (0.1 to 2.0 deg/s)
    w_dir = normalize(rng.standard_normal(3))
    w_mag = rng.uniform(0.1, 2.0) * np.pi / 180.0
    w0 = w_dir * w_mag
    
    # Randomize RW Momentum (2 wheels, random +/- 5mNms)
    # The user code had 2 RWs (rws.pop()), so h0 is length 2
    h0 = rng.uniform(-0.005, 0.005, size=1)
    
    # Randomize ECI Goal Vector
    goal_eci = normalize(rng.standard_normal(3))
    
    return {
        "run_id": run_id,
        "seed": run_id,
        "tf": 1000,
        "dt": 2,
        "w0": w0,
        "q0": q0,
        "h0": h0,
        "goal_eci_vec": goal_eci,
        "orbit_R": 7000*np.array([0, np.sqrt(2)/2, np.sqrt(2)/2]),
        "orbit_V": np.array([8, 0, 0])
    }

if __name__ == "__main__":
    RUN_MC: bool = True

    if RUN_MC:
        runner = MonteCarloRunner(
            sim_func=run_single_sim,
            config_generator=generate_mc_config,
            num_runs=5
        )
        
        full_results = runner.run()
        
        print("\n--- Monte Carlo Complete ---")
        print(f"Generated {len(full_results)} histories.")
        
        save_data("3MTQ+1RW_mc_eci_convergence", full_results, out_dir="papers/3MTQ+1RW/output_data")

        plot_target_tracking_mc(full_results=full_results)
        create_close_all_button_window()

    else:
        results = load_data("papers/3MTQ+1RW/output_data/3MTQ+1RW_mc_eci_convergence_20260110_145232")
        full_results = results[0]
        plot_target_tracking_mc(full_results=full_results)
        create_close_all_button_window()
