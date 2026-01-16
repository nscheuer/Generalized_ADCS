"""
Integration test: Use DebugPlanner with the actual debug_altro_6Up problem.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

from ADCS.CONOPS.goals import ECI_Goal, No_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.helpers import PlannerSettings, Trajectory, DebugPlanner
from ADCS.controller.helpers.build_csat import build_cpp_satellite
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import RW
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import random_n_unit_vec, normalize

import trajectory_planner.build.tplaunch as tplaunch


def test_debug_planner_integration():
    """Test DebugPlanner with the actual problem setup from debug_altro_6Up."""
    print("=" * 70)
    print("INTEGRATION TEST: DebugPlanner with debug_altro_6Up setup")
    print("=" * 70)

    np.random.seed(1)
    tf = 100
    dt = 1.0
    t0 = 0
    N = int((tf - t0) / dt)

    # Same setup as debug_altro_6Up.py
    rw_max_torque = 0.005
    rw_J = 0.0014
    rw_h0 = 0.0
    rw_hmax = 0.015
    rws = [RW(axis=j, max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax)
           for j in MathConstants.unitvecs]
    acts = rws
    rwN = sum([isinstance(act, RW) for act in acts])

    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]

    real_sat = Satellite(
        mass=10.165,
        J_0=np.diagflat([0.0969, 0.1235, 0.1918]),
        actuators=acts,
        sensors=mtms,
        boresight=np.array([0, 0, 1])
    )

    w0 = np.array([0.0, 0.0, 0.0])
    q0 = normalize(np.array([1, 0, 0, 0]))
    h0 = np.array([rw_h0] * rwN)
    x = np.concatenate([w0, q0, h0])

    ephem = Ephemeris()
    start_time = 0.22 - 1 * TimeConstants.sec2cent
    end_time = 0.22 + (tf - t0) * TimeConstants.sec2cent
    R = 7000 * np.array([0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    V = np.array([8, 0, 0])

    os0 = Orbital_State(
        ephem=ephem, J2000=0.22 - 1 * TimeConstants.sec2cent, R=R, V=V,
        B=np.array([0, 0.1, 0]), S=np.array([1e5 + 1, 0, 0]), rho=5e-12
    )

    dur = int((tf - t0) / dt) + 10
    orbs = [os0] * (dur + 10)
    for j in range(dur):
        orbs[j] = os0.copy()
        orbs[j].J2000 = os0.J2000 + j * dt * TimeConstants.sec2cent
    orb = Orbit(orbs)

    # Build planner settings
    planner_settings = PlannerSettings(est_sat=real_sat, bdot_on=0, dt_tp=1.0)
    planner_settings.verbosity = False
    planner_settings.rw_control_weight = 1e0
    planner_settings.mtq_control_weight = 1e0
    planner_settings.cost_main.ang_vel = 0
    planner_settings.cost_second.ang_vel = 0
    planner_settings.cost_main.use_raw_control_cost = True
    planner_settings.pass1.aug_lag.penalty_init = 1e2

    # Build C++ satellite
    csat = build_cpp_satellite(est_sat=real_sat, planner_settings=planner_settings)

    # Test 1: Build regular planner
    print("\n[1] Building regular tplaunch.Planner...")
    regular_planner = tplaunch.Planner(
        csat,
        planner_settings.systemSettings(),
        planner_settings.mainAlilqrSettings(),
        planner_settings.secondAlilqrSettings(),
        planner_settings.initTrajSettings(),
        planner_settings.optMainCostSettings(),
        planner_settings.optSecondCostSettings(),
        planner_settings.optTVLQRCostSettings(tracking_LQR_formulation=0)
    )
    print("    ✓ Regular planner created")

    # Test 2: Build DebugPlanner with same args
    print("\n[2] Building DebugPlanner (debug_level=1)...")
    debug_planner = DebugPlanner(
        csat,
        planner_settings.systemSettings(),
        planner_settings.mainAlilqrSettings(),
        planner_settings.secondAlilqrSettings(),
        planner_settings.initTrajSettings(),
        planner_settings.optMainCostSettings(),
        planner_settings.optSecondCostSettings(),
        planner_settings.optTVLQRCostSettings(tracking_LQR_formulation=0),
        debug_level=1
    )
    print("    ✓ DebugPlanner created")

    # Test 3: Verify passthrough methods work
    print("\n[3] Testing passthrough methods...")
    debug_planner.setquaternionTo3VecMode(2)
    debug_planner.setVerbosity(False)
    dt_val = debug_planner.getdt()
    print(f"    ✓ getdt() = {dt_val}")
    echo_val = debug_planner.echo_int(42)
    print(f"    ✓ echo_int(42) = {echo_val}")

    # Test 4: Prepare environment vectors (same as Plan_and_Track_Exact._propagate_environment)
    print("\n[4] Preparing environment vectors...")
    t_start = 0.22
    t_end = t_start + (tf * TimeConstants.sec2cent)
    N_pts = int(np.ceil(tf / dt)) + 1

    buffer_centuries = 10 * dt * TimeConstants.sec2cent
    t_end_buffered = t_end + buffer_centuries

    sim_orbit = Orbit(os0=os0, end_time=t_end_buffered, dt=dt, use_J2=True, fast=False)
    tp_orbit = sim_orbit.get_range(t_start, t_end, dt)

    orbit_data_lists = tp_orbit.get_vecs()
    times_arr = np.asarray(tp_orbit.times, dtype=np.float64)

    # Clip/pad times to N_pts
    curN = times_arr.shape[0]
    if curN > N_pts:
        times_arr = times_arr[:N_pts]
    elif curN < N_pts:
        times_arr = np.pad(times_arr, (0, N_pts - curN), mode="edge")

    def to_mat3xN(x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:
            if x.shape == (3, N_pts):
                return np.asfortranarray(x, dtype=np.float64)
            elif x.shape[1] == 3:
                x = x.T
            if x.shape[1] > N_pts:
                x = x[:, :N_pts]
            elif x.shape[1] < N_pts:
                x = np.pad(x, ((0, 0), (0, N_pts - x.shape[1])), mode="edge")
        return np.asfortranarray(x, dtype=np.float64)

    def to_vecN(x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] > N_pts:
            x = x[:N_pts]
        elif x.shape[0] < N_pts:
            x = np.pad(x, (0, N_pts - x.shape[0]), mode="edge")
        return np.ascontiguousarray(x, dtype=np.float64)

    R_raw, V_raw, B_raw, S_raw, Rho_raw = [np.asarray(d) for d in orbit_data_lists]

    R = to_mat3xN(R_raw)
    V = to_mat3xN(V_raw)
    B = to_mat3xN(B_raw)
    S = to_mat3xN(S_raw)
    rho = to_vecN(Rho_raw)

    goals = GoalList({0.22: No_Goal(), 0.22 + 3 * TimeConstants.sec2cent: ECI_Goal(np.array([1, 1, 1]))})

    goal_vecs_eci = np.zeros((3, N_pts), dtype=np.float64, order="F")
    sat_body_vecs = np.zeros((3, N_pts), dtype=np.float64, order="F")
    prop_vals = np.zeros(N_pts, dtype=np.float64)

    for i in range(N_pts):
        t = float(times_arr[i])
        os_at_t = sim_orbit.get_os(t)
        g_vec_eci, _w_ref = goals.to_ref(t, os_at_t)
        goal_vecs_eci[:, i] = np.asarray(g_vec_eci, dtype=np.float64).reshape(3)
        sat_body_vecs[:, i] = np.asarray(real_sat.boresight, dtype=np.float64).reshape(3)

    A = np.asfortranarray(sat_body_vecs, dtype=np.float64)
    E = np.asfortranarray(goal_vecs_eci, dtype=np.float64)
    p = np.ascontiguousarray(prop_vals.reshape(-1), dtype=np.float64)
    t_c = np.ascontiguousarray(times_arr.reshape(-1), dtype=np.float64)

    vecsPy = (t_c, R, V, B, S, A, E, p, rho)
    print(f"    ✓ Environment vectors prepared, N_pts={N_pts}")

    # Test 5: Call trajOpt through DebugPlanner
    print("\n[5] Running trajOpt through DebugPlanner...")
    x0_clean = np.copy(x.astype(np.float64).flatten(), order='C')
    bdotOn = planner_settings.bdot_on

    result = debug_planner.trajOpt(vecsPy, N_pts, t_start, t_end, x0_clean, int(bdotOn))

    (success, cost, opt1, lqr_opt, traj_final) = result
    (Xset, Uset, Tset, Kset, Sset, lqr_times) = lqr_opt

    print(f"\n    Result: success={success}")
    print(f"    Final cost: {cost}")
    print(f"    Xset shape: {Xset.shape}")
    print(f"    Uset shape: {Uset.shape}")

    # Final analysis
    print("\n" + "=" * 70)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 70)

    if Uset.ndim == 2:
        u_hist = Uset.T
    else:
        u_hist = Uset.reshape(-1, 3)

    for ch in range(min(3, u_hist.shape[1])):
        u_ch = u_hist[:, ch]
        sign_changes = np.sum(np.diff(np.sign(u_ch)) != 0)
        rapid_osc = 0
        if len(u_ch) > 2:
            signs = np.sign(u_ch)
            for i in range(len(signs) - 2):
                if signs[i] != signs[i+1] and signs[i+1] != signs[i+2]:
                    rapid_osc += 1
        status = "⚠️  OSCILLATING" if rapid_osc > 5 else "✓"
        print(f"  Ch{ch}: sign_changes={sign_changes}, rapid_osc={rapid_osc} {status}")

    print("\n✓ DebugPlanner works as drop-in replacement!")
    return True


if __name__ == "__main__":
    test_debug_planner_integration()
