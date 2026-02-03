import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Tuple, Optional, Dict, Union
from tqdm import tqdm
import pytest
from functools import lru_cache

# Path setup
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

# ADCS Imports
from ADCS.CONOPS.goals import Goal, ECI_Goal, No_Goal
from ADCS.controller import MTQ_Wisniewski
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import random_n_unit_vec, normalize
from ADCS.controller.helpers.quaternion_math import vector_alignment_error

# Plotting Imports
from ADCS.helpers.plotting.animate_estimator import animate_attitude
from ADCS.helpers.plotting.plot_estimator import plot_state_comparison
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window
from ADCS.helpers.plotting.plot_controller import plot_control, plot_rw_momentum, plot_target_tracking

# ----------------------------
# Configuration Constants
# ----------------------------

# Tuned for ~4kg satellite with magnetic control
GAIN_CFG: Dict[str, float] = dict(
    lambda_s=np.diag([0.003, 0.003, 0.003]),
    lambda_q=np.diag([0.002, 0.002, 0.002])
)

# Simulation Physics
DT_PHYSICS = 50  # Magnetic control requires small time steps!
TF = 10000   # Alignment takes time with weak magnetorquers

# ----------------------------
# Shared Setup + Utilities
# ----------------------------

def _make_satellite(initial_rw_h: float = 0.0) -> Tuple[Satellite, np.ndarray, List]:
    mtq_max_torque = 1.0
    mtqs = [MTQ(axis=j, max_torque=mtq_max_torque) for j in MathConstants.unitvecs]

    if isinstance(initial_rw_h, (float, int)):
        h0_vec = np.array([float(initial_rw_h)] * 3)
    else:
        h0_vec = np.array(initial_rw_h, dtype=float)

    rw_max_torque = 7*0.001
    rw_J = 0.001
    rw_h0 = 5*0.001
    rw_hmax = 16.2*0.001
    rws = [RW(axis=j, max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax) for j in MathConstants.unitvecs]
    acts = mtqs + rws

    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    sat = Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]), actuators=acts, sensors=mtms, boresight=np.array([0, 0, 1]))

    w0 = np.zeros(3)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    h0 = np.array([initial_rw_h] * 3)
    x0 = np.concatenate([w0, q0, h0])

    return sat, x0, acts

def _make_controller(est_sat: Satellite, goal: Goal) -> MTQ_Wisniewski:
    """
    Factory for MTQ_Wisniewski controller with appropriate gains.
    """
    cfg = GAIN_CFG
        
    return MTQ_Wisniewski(
        est_sat=est_sat,
        lambda_s=cfg["lambda_s"],
        lambda_q=cfg["lambda_q"]
    )

@lru_cache(maxsize=4)
def _get_orbit_cached(tf: float, dt: float) -> Orbit:
    """
    Cached orbit propagation to speed up multiple tests.
    """
    ephem = Ephemeris()
    start_time = 0.22 - 1 * TimeConstants.sec2cent
    end_time = 0.22 + tf * TimeConstants.sec2cent

    # 7000km Orbit
    R = 7000 * np.array([0.0, -np.sqrt(2) / 2, np.sqrt(2) / 2])
    V = np.array([8.0, 0.0, 0.0])

    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
    # Using real J2 orbit for accurate magnetic field variance
    return Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=True, fast=False)


def get_scenario_init(name: str) -> Tuple[np.ndarray, np.ndarray, float, Goal, float]:
    """
    Parses scenario string to return: w0, q0, h_wheel, Goal, TF
    """
    name = name.lower()
    
    # 1. Defaults
    w0 = np.zeros(3)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    h_val = 0.0
    goal = No_Goal()
    tf = TF

    # 2. Angular Velocity
    if "moving" in name or "spin" in name:
        # random_n_unit_vec(3) * uniform(1, 2) deg/s -> rad/s
        # Using smaller spin for alignment tests, larger for detumble
        scale = 1.5 if "stop" in name else 0.2 
        w0 = random_n_unit_vec(3) * np.random.uniform(scale*0.5, scale*0.8) * np.pi / 180.0

    # 3. Quaternion
    if "random_q" in name or "moving" in name or "spin" in name:
        q0 = normalize(random_n_unit_vec(4))

    # 4. Momentum
    if "wh" in name or "h=0.001" in name:
        h_val = 0.001

    # 5. Goal & Time Horizon
    if "align_xyz" in name:
        goal = ECI_Goal(normalize(np.array([1, 1, 1])))
        tf = TF
    elif "align_x" in name:
        goal = ECI_Goal(np.array([1, 0, 0]))
        tf = TF
    
    return w0, q0, h_val, goal, tf


def simulate_scenario(
    scenario_name: str,
    verbose: bool = False,
    override_tf: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Orbital_State], np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(37)

    # 1. Init Conditions
    w0, q0, h_val, goal, tf_default = get_scenario_init(scenario_name)
    tf = override_tf if override_tf else tf_default
    dt = DT_PHYSICS

    # 2. Setup
    sat, _, acts = _make_satellite(initial_rw_h=h_val)
    # Overwrite state with specific scenario conditions
    x = np.concatenate([w0, q0, np.array([h_val]*3)])
    
    controller = _make_controller(est_sat=sat, goal=goal)
    orbit = _get_orbit_cached(tf=tf, dt=dt)

    # 3. Logging
    steps = int(tf / dt)
    N = steps
    
    time_hist = np.nan * np.zeros(N)
    state_hist = np.nan * np.zeros((N, len(x)))
    os_hist: List[Orbital_State] = []
    sensor_hist = np.nan * np.zeros((N, len(sat.sensors + sat.rw_actuators)))
    u_hist = np.nan * np.zeros((N, len(acts)))
    boresight_hist = np.nan * np.zeros((N, 4))

    t = 0.0
    ind = 0

    desc = f"Simulating: {scenario_name}"
    for _ in tqdm(range(steps), desc=desc, disable=not verbose):
        J2000 = 0.22 + t * TimeConstants.sec2cent
        os_now = orbit.get_os(J2000=J2000)

        # Control Step
        sens = sat.sensor_readings(x=x, os=os_now)
        u = controller.find_u(x_hat=x, sens=sens, est_sat=sat, os_hat=os_now, goal=goal)
        
        if verbose and ind % 1000 == 0:
            # print(f"t={t:.0f}, |w|={np.linalg.norm(x[0:3]):.4f}, u_max={np.max(np.abs(u)):.4f}")
            pass

        # Log
        time_hist[ind] = t
        state_hist[ind, :] = x
        os_hist.append(os_now)
        sensor_hist[ind, :] = sens
        u_hist[ind, :] = u
        
        if goal is not None:
            eci_goal, _ = goal.to_ref(os0=os_now)
            boresight_hist[ind, :] = eci_goal

        # Propagation
        t_next = t + dt
        # For solver, we need next state of environment. 
        # Approximation: pass current OS or propagate slightly.
        # Ideally, we query orbit again for t_next.
        os_next = orbit.get_os(J2000=0.22 + t_next * TimeConstants.sec2cent)

        out = solve_ivp(
            fun=sat.dynamics_for_solver,
            t_span=(0, dt),
            y0=x,
            method="RK45",
            args=(u, os_now, os_next),
            rtol=1e-7, atol=1e-7,
        )
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])

        t = t_next
        ind += 1

    return time_hist, state_hist, os_hist, sensor_hist, u_hist, boresight_hist

# ----------------------------
# Pytest Scenarios
# ----------------------------

SCENARIO_LIST = [
    # Stop Rotation
    "stop_rot_zero",
    "stop_rot_moving",
    "stop_rot_moving_wh",
    
    # Align X
    "align_x_zero",
    "align_x_zero_wh",
    "align_x_moving",
    "align_x_moving_wh",

    # Align XYZ
    "align_xyz_zero",
    "align_xyz_zero_wh",
    "align_xyz_moving",
    "align_xyz_moving_wh",
]

@pytest.mark.slow
@pytest.mark.parametrize("scenario", SCENARIO_LIST)
def test_mtq_wisniewski_scenarios(scenario: str) -> None:
    # Run sim
    _, state_hist, _, _, u_hist, boresight_hist = simulate_scenario(scenario)
    
    # Analyze Final State
    valid = np.where(~np.isnan(state_hist[:, 0]))[0]
    k = valid[-1]
    w_final = state_hist[k, 0:3]
    q_final = state_hist[k, 3:7]
    
    # 1. Stability Check (Omega)
    w_norm = np.linalg.norm(w_final)
    assert w_norm < 2e-3, f"Final rate too high: {w_norm:.5f} rad/s"

    # 2. Pointing Check (If Alignment)
    if "align" in scenario:
        goal_vec = boresight_hist[k, 1:4]
        # Check for NaN in logs
        if np.isnan(goal_vec).any(): goal_vec = np.array([1,0,0]) # Fallback if logging issue

        sat_boresight = np.array([0, 0, 1])
        err_vec = vector_alignment_error(q_final, goal_vec, sat_boresight)
        err_deg = np.degrees(np.linalg.norm(err_vec))
        
        # Magnetic control with disturbances (wheels) is imprecise
        tol = 10.0 if "wh" in scenario else 5.0
        assert err_deg < tol, f"Pointing error {err_deg:.2f} > tolerance {tol}"


def plot_scenario_manual(scenario: str):
    print(f"\n--- Running Manual Scenario: {scenario} ---")
    
    res = simulate_scenario(scenario, verbose=True)
    (time_hist, state_hist, os_hist, _, u_hist, boresight_hist) = res

    # Plotting
    animate_attitude(time=time_hist, state_hist=state_hist, os_hist=os_hist, boresight_goal_hist=boresight_hist)
    plot_control(time=time_hist, u_hist=u_hist)
    plot_state_comparison(time=time_hist, state_hist=state_hist)
    plot_rw_momentum(time=time_hist, state_hist=state_hist)
    plot_target_tracking(
        state_hist=state_hist,
        boresight_hist=boresight_hist,
        body_boresight=np.array([0.0, 0.0, 1.0]),
    )
    create_close_all_button_window()

if __name__ == "__main__":
    np.random.seed(37)
    if len(sys.argv) > 1:
        chosen = sys.argv[1]
        if chosen in SCENARIO_LIST:
            plot_scenario_manual(chosen)
        else:
            print(f"Unknown scenario '{chosen}'.")
            print(f"Available: {SCENARIO_LIST}")
    else:
        # Default run if no args
        plot_scenario_manual("align_xyz_zero")