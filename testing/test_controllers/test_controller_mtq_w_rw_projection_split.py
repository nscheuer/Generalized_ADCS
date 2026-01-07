import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Union, Tuple
from tqdm import tqdm
import pytest
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from ADCS.CONOPS.goals import Goal, ECI_Goal, Coordinate_Goal
from ADCS.controller import MTQ_w_RW_Projection_Split
from ADCS.controller.helpers.quaternion_math import vector_alignment_error
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
from ADCS.helpers.plotting.plot_controller import plot_control, plot_rw_momentum
from ADCS.helpers.plotting.animate_orbit import animate_orbit

# ==========================================
# UNIFIED SIMULATION FUNCTION
# ==========================================

def run_simulation(
    verbose: bool, 
    tf: float, 
    dt: float, 
    real_orbit: bool,
    p_gain: float, 
    d_gain: float, 
    c_gain: float,
    initial_w: np.ndarray,
    initial_q: np.ndarray,
    initial_rw_h: Union[float, np.ndarray],
    goal: Union[Goal, None] = None,
    boresight: np.ndarray = np.array([0, 0, 1])
) -> Tuple[np.ndarray, np.ndarray, List[Orbital_State], np.ndarray, np.ndarray, np.ndarray]:
    """
    Unified simulation engine to replace repetitive test code.
    """
    np.random.seed(1)
    t0 = 0
    N = int((tf-t0)/dt)

    # Hardware Setup
    mtm_max_torque = 0.1
    mtqs = [MTQ(axis=j, max_torque=mtm_max_torque) for j in MathConstants.unitvecs]

    # Handle Momentum Initialization (Scalar or Vector)
    if isinstance(initial_rw_h, (float, int)):
        h0_vec = np.array([float(initial_rw_h)] * 3)
    else:
        h0_vec = np.array(initial_rw_h, dtype=float)

    rw_max_torque = 4.51
    rw_J = 0.22
    rw_hmax = 3.8
    rws = []
    for idx, axis in enumerate(MathConstants.unitvecs):
        rws.append(RW(axis=axis, max_torque=rw_max_torque, J=rw_J, h=h0_vec[idx], h_max=rw_hmax))

    acts = mtqs + rws
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]

    # 
    real_sat = Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]), actuators=acts, sensors=mtms, boresight=boresight)

    # Initial State
    x = np.concatenate([initial_w, initial_q, h0_vec])

    # Orbit Setup
    ephem = Ephemeris()
    start_time = 0.22 - 1*TimeConstants.sec2cent
    end_time = 0.22 + (tf-t0)*TimeConstants.sec2cent
    pos_R = 7000*np.array([0, -np.sqrt(2)/2, np.sqrt(2)/2])
    vel_V = np.array([8, 0, 0])

    if real_orbit:
        os0 = Orbital_State(ephem=ephem, J2000=start_time, R=pos_R, V=vel_V)
        orb = Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=True, fast=False)
    else:
        os0 = Orbital_State(ephem=ephem, J2000=0.22-1*TimeConstants.sec2cent, R=pos_R, V=vel_V, 
                            B=np.array([0, 0.1, 0]), S=np.array([1e5+1, 0, 0]), rho=5e-12)
        dur = int((tf-t0)/dt)+10
        orbs = [os0]*(dur+10)
        for j in range(dur):
            temp_os = os0.copy()
            temp_os.J2000 = os0.J2000 + j*dt*TimeConstants.sec2cent
            orbs[j] = temp_os
        orb = Orbit(orbs)

    # Controller
    controller = MTQ_w_RW(est_sat=real_sat, p_gain=p_gain, d_gain=d_gain, c_gain=c_gain, h_target=np.array([0, 0, 0]))

    # Logging
    time_hist = np.nan*np.zeros(N)
    state_hist = np.nan*np.zeros((N, 10))
    os_hist: List[Orbital_State] = list()
    sensor_hist: np.ndarray = np.nan*np.zeros((N, len(real_sat.sensors + real_sat.rw_actuators)))
    u_hist = np.nan*np.zeros((N, len(acts)))
    boresight_hist = np.nan*np.zeros((N, 3))

    t = t0
    ind = 0
    steps = int((tf - t0)/dt)
    
    desc_str = f"Simulating (Goal={'Yes' if goal else 'No'}, Orbit={'Real' if real_orbit else 'Static'})"
    for step in tqdm(range(steps), desc=desc_str):
        J2000 = 0.22 + t*TimeConstants.sec2cent
        os = orb.get_os(J2000=J2000)

        sens = real_sat.sensor_readings(x=x, os=os)
        u = controller.find_u(x_hat=x, sens=sens, est_sat=real_sat, os_hat=os, goal=goal)

        if verbose:
            print("u: ", u)

        time_hist[ind] = t
        state_hist[ind,:] = x
        os_hist.append(os)
        sensor_hist[ind,:] = sens
        u_hist[ind,:] = u
        
        if goal is not None:
            eci_goal, _ = goal.to_ref(os0=os)
            boresight_hist[ind, :] = eci_goal

        ind += 1
        t += dt
        prev_os = os.copy()
        next_os = orb.get_os(0.22+(t-t0)*TimeConstants.sec2cent)

        out = solve_ivp(fun=real_sat.dynamics_for_solver, t_span=(0, dt), y0=x, method="RK45", args=(u, prev_os, next_os), rtol=1e-7, atol=1e-7)
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])

    return time_hist, state_hist, os_hist, sensor_hist, u_hist, boresight_hist

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture(scope="module")
def stop_rotation_results():
    w0 = random_n_unit_vec(3)*np.random.uniform(1, 2)*np.pi/180.0
    q0 = random_n_unit_vec(4)
    return run_simulation(
        verbose=False, tf=100, dt=1, real_orbit=False,
        p_gain=0, d_gain=1, c_gain=0,
        initial_w=w0, initial_q=q0, initial_rw_h=0, goal=None
    )

@pytest.fixture(scope="module")
def align_results():
    w0 = np.array([0, 0, 0])
    q0 = normalize(np.array([1, 0, 0, 0]))
    goal = ECI_Goal(np.array([1, 0, 0]))
    return run_simulation(
        verbose=False, tf=100, dt=1, real_orbit=False,
        p_gain=0.1, d_gain=0.7, c_gain=0,
        initial_w=w0, initial_q=q0, initial_rw_h=1, goal=goal
    )

@pytest.fixture(scope="module")
def complex_align_results():
    w0 = np.array([0, 0, 0])
    q0 = normalize(np.array([1, 0, 0, 0]))
    goal = ECI_Goal(normalize(np.array([1, 1, 1])))
    return run_simulation(
        verbose=False, tf=100, dt=1, real_orbit=False,
        p_gain=0.1, d_gain=0.7, c_gain=0,
        initial_w=w0, initial_q=q0, initial_rw_h=1, goal=goal
    )

@pytest.fixture(scope="module")
def desaturate_results():
    w0 = np.array([0, 0, 0])
    q0 = normalize(np.array([1, 0, 0, 0]))
    goal = ECI_Goal(np.array([1, 0, 0]))
    # Wheel 0 has 0.5 momentum, others 0
    h_init = np.array([0.5, 0.0, 0.0])
    # Increased time for desaturation
    return run_simulation(
        verbose=False, tf=100, dt=1, real_orbit=False,
        p_gain=0.1, d_gain=0.7, c_gain=0.1,
        initial_w=w0, initial_q=q0, initial_rw_h=h_init, goal=goal
    )

@pytest.fixture(scope="module")
def full_task_results():
    w0 = np.array([0, 0, 0])
    q0 = normalize(np.array([1, 0, 0, 0]))
    goal = ECI_Goal(normalize(np.array([1, 1, 1])))
    # All wheels have momentum
    h_init = 0.5
    # Real orbit + Desaturation + Pointing
    return run_simulation(
        verbose=False, tf=100, dt=1, real_orbit=True,
        p_gain=0.1, d_gain=0.7, c_gain=0.1,
        initial_w=w0, initial_q=q0, initial_rw_h=h_init, goal=goal
    )

# ==========================================
# TESTS
# ==========================================

@pytest.mark.slow
def test_mtq_w_rw_stop_rotation(stop_rotation_results):
    _, state_hist, _, _, u_hist, _ = stop_rotation_results
    
    # Check Omega
    final_w = state_hist[-1, 0:3]
    w_norm = np.linalg.norm(final_w)
    assert w_norm < 1e-3, f"Final omega too high: {w_norm}"

    # Check Control
    final_u_avg = np.mean(np.abs(u_hist[-5:]), axis=0)
    u_max = np.max(final_u_avg)
    assert u_max < 1e-4, f"Control effort not settled: {u_max}"

@pytest.mark.slow
def test_mtq_w_rw_align(align_results):
    _, state_hist, _, _, u_hist, _ = align_results
    
    # Check Omega
    final_w = state_hist[-1, 0:3]
    assert np.linalg.norm(final_w) < 1e-3

    # Check Control
    final_u_avg = np.mean(np.abs(u_hist[-5:]), axis=0)
    assert np.max(final_u_avg) < 1e-3

    # Check Alignment using vector_alignment_error
    final_q_adcs = state_hist[-1, 3:7]
    target_vec_eci = np.array([1, 0, 0])
    sat_boresight = np.array([0, 0, 1])
    
    q_err_vec = vector_alignment_error(q=final_q_adcs, eci_goal=target_vec_eci, body_boresight=sat_boresight)
    err_rad = np.linalg.norm(q_err_vec)
    err_deg = np.degrees(err_rad)

    assert err_deg < 1.0, f"Alignment error too high: {err_deg}"

@pytest.mark.slow
def test_mtq_w_rw_complex_align(complex_align_results):
    _, state_hist, _, _, u_hist, _ = complex_align_results
    
    # Check Omega
    final_w = state_hist[-1, 0:3]
    assert np.linalg.norm(final_w) < 1e-3

    # Check Control
    final_u_avg = np.mean(np.abs(u_hist[-5:]), axis=0)
    assert np.max(final_u_avg) < 1e-3

    # Check Alignment using vector_alignment_error
    final_q_adcs = state_hist[-1, 3:7]
    target_vec_eci = normalize(np.array([1, 1, 1]))
    sat_boresight = np.array([0, 0, 1])
    
    q_err_vec = vector_alignment_error(q=final_q_adcs, eci_goal=target_vec_eci, body_boresight=sat_boresight)
    err_rad = np.linalg.norm(q_err_vec)
    err_deg = np.degrees(err_rad)

    assert err_deg < 1.0, f"Complex Alignment error too high: {err_deg}"

@pytest.mark.slow
def test_mtq_w_rw_desaturate(desaturate_results):
    _, state_hist, _, _, _, _ = desaturate_results
    
    final_h = state_hist[-1, 7:10]
    # Wheel 0 started at 0.5, expect it to drop
    assert abs(final_h[0]) < 0.1, f"Wheel 0 did not desaturate: {final_h[0]}"
    # Others started at 0
    assert abs(final_h[1]) < 0.1
    assert abs(final_h[2]) < 0.1

@pytest.mark.slow
def test_mtq_w_rw_full(full_task_results):
    _, state_hist, _, _, _, _ = full_task_results
    
    # Check Omega
    final_w = state_hist[-1, 0:3]
    assert np.linalg.norm(final_w) < 1e-3, "Full task instability"

    # Check Alignment using vector_alignment_error
    final_q_adcs = state_hist[-1, 3:7]
    target_vec_eci = normalize(np.array([1, 1, 1]))
    sat_boresight = np.array([0, 0, 1])
    
    q_err_vec = vector_alignment_error(q=final_q_adcs, eci_goal=target_vec_eci, body_boresight=sat_boresight)
    err_rad = np.linalg.norm(q_err_vec)
    err_deg = np.degrees(err_rad)

    assert err_deg < 1.0, f"Full task alignment error: {err_deg}"

# ==========================================
# PLOTTING FUNCTIONS (Wrappers for manual use)
# ==========================================

def plot_mtq_w_rw_stop_rotation(verbose: bool = False, tf: float = 100, dt: float = 1, real_orbit: bool = False):
    w0 = random_n_unit_vec(3)*np.random.uniform(1, 2)*np.pi/180.0
    q0 = random_n_unit_vec(4)
    res = run_simulation(verbose, tf, dt, real_orbit, 0, 1, 0, w0, q0, 0, None)
    _plot_results(res)

def plot_mtq_w_rw_align(verbose: bool = False, tf: float = 100, dt: float = 1, real_orbit: bool = False):
    w0 = np.array([0, 0, 0])
    q0 = normalize(np.array([1, 0, 0, 0]))
    goal = ECI_Goal(np.array([1, 0, 0]))
    res = run_simulation(verbose, tf, dt, real_orbit, 0.1, 0.7, 0, w0, q0, 1, goal)
    _plot_results(res)

def plot_mtq_w_rw_complex_align(verbose: bool = False, tf: float = 100, dt: float = 1, real_orbit: bool = False):
    w0 = random_n_unit_vec(3)*np.random.uniform(1, 2)*np.pi/180.0
    q0 = random_n_unit_vec(4)
    goal = ECI_Goal(normalize(np.array([1, 1, 1])))
    res = run_simulation(verbose, tf, dt, real_orbit, 0.1, 0.7, 0, w0, q0, 1, goal)
    _plot_results(res)

def plot_mtq_w_rw_desaturate(verbose: bool = False, tf: float = 300, dt: float = 1, real_orbit: bool = False):
    w0 = np.array([0, 0, 0])
    q0 = normalize(np.array([1, 0, 0, 0]))
    goal = ECI_Goal(np.array([0, 0, 1]))
    h_init = np.array([0.5, 0.0, 0.0])
    res = run_simulation(verbose, tf, dt, real_orbit, 0.1, 0.7, 0.1, w0, q0, h_init, goal)
    _plot_results(res)

def plot_mtq_w_rw_full(verbose: bool = False, tf: float = 500, dt: float = 1, real_orbit: bool = True):
    w0 = random_n_unit_vec(3)*np.random.uniform(1, 2)*np.pi/180.0
    q0 = random_n_unit_vec(4)
    goal = ECI_Goal(normalize(np.array([1, 1, 1])))
    h_init = 0.5
    res = run_simulation(verbose, tf, dt, real_orbit, 0.1, 0.7, 0.1, w0, q0, h_init, goal)
    _plot_results(res)

def _plot_results(results):
    (time_hist, state_hist, os_hist, sensor_hist, u_hist, boresight_hist) = results
    animate_attitude(time=time_hist, state_hist=state_hist, os_hist=os_hist, boresight_goal_hist=boresight_hist)
    plot_control(time=time_hist, u_hist=u_hist)
    plot_state_comparison(time=time_hist, state_hist=state_hist)
    plot_rw_momentum(time=time_hist, state_hist=state_hist)
    create_close_all_button_window()

if __name__ == "__main__":
    plot_mtq_w_rw_full(verbose=False, tf = 100, dt = 1, real_orbit= True)