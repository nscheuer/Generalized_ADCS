"""
Debug the first iteration in detail to understand why controls explode.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

from ADCS.CONOPS.goals import ECI_Goal, No_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track import PlannerSettings, DebugPlanner
from ADCS.controller.plan_and_track.build_csat import build_cpp_satellite
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import RW
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize


def setup_and_prepare():
    """Setup and return prepared planner and data."""
    np.random.seed(1)
    tf, dt = 100, 1.0

    rw_max_torque = 0.005
    rws = [RW(axis=j, max_torque=rw_max_torque, J=0.0014, h=0.0, h_max=0.015)
           for j in MathConstants.unitvecs]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]

    real_sat = Satellite(
        mass=10.165, J_0=np.diagflat([0.0969, 0.1235, 0.1918]),
        actuators=rws, sensors=mtms, boresight=np.array([0, 0, 1])
    )

    x0 = np.concatenate([np.zeros(3), normalize(np.array([1,0,0,0])), np.zeros(3)])

    ephem = Ephemeris()
    os0 = Orbital_State(
        ephem=ephem, J2000=0.22 - TimeConstants.sec2cent,
        R=7000*np.array([0, np.sqrt(2)/2, np.sqrt(2)/2]), V=np.array([8,0,0]),
        B=np.array([0, 0.1, 0]), S=np.array([1e5+1, 0, 0]), rho=5e-12
    )

    planner_settings = PlannerSettings(est_sat=real_sat, bdot_on=0, dt_tp=1.0)
    planner_settings.verbosity = False
    planner_settings.rw_control_weight = 1e0
    planner_settings.cost_main.ang_vel = 0
    planner_settings.cost_main.use_raw_control_cost = True
    planner_settings.pass1.aug_lag.penalty_init = 1e2

    csat = build_cpp_satellite(est_sat=real_sat, planner_settings=planner_settings)

    planner = DebugPlanner(
        csat,
        planner_settings.systemSettings(),
        planner_settings.mainAlilqrSettings(),
        planner_settings.secondAlilqrSettings(),
        planner_settings.initTrajSettings(),
        planner_settings.optMainCostSettings(),
        planner_settings.optSecondCostSettings(),
        planner_settings.optTVLQRCostSettings(0),
        debug_level=0
    )
    planner.setquaternionTo3VecMode(2)

    # Environment
    t_start, t_end = 0.22, 0.22 + tf * TimeConstants.sec2cent
    N = int(np.ceil(tf / dt)) + 1

    sim_orbit = Orbit(os0=os0, end_time=t_end + 10*dt*TimeConstants.sec2cent, dt=dt, zonal_J=2, fast=False)
    tp_orbit = sim_orbit.get_range(t_start, t_end, dt)
    orbit_data = tp_orbit.get_vecs()
    times = np.asarray(tp_orbit.times, dtype=np.float64)[:N]
    if len(times) < N:
        times = np.pad(times, (0, N-len(times)), mode="edge")

    def to_mat(x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2 and x.shape[1] == 3: x = x.T
        if x.shape[1] > N: x = x[:, :N]
        elif x.shape[1] < N: x = np.pad(x, ((0,0),(0,N-x.shape[1])), mode="edge")
        return np.asfortranarray(x)

    def to_vec(x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)[:N]
        if len(x) < N: x = np.pad(x, (0, N-len(x)), mode="edge")
        return np.ascontiguousarray(x)

    R, V, B, S, Rho = [np.asarray(d) for d in orbit_data]
    # Simplified: ECI_Goal from start, no transition
    goals = GoalList({0.22: ECI_Goal(np.array([1,1,1]))})

    E = np.zeros((4, N), dtype=np.float64, order="F")
    A = np.zeros((3, N), dtype=np.float64, order="F")
    for i in range(N):
        g, _ = goals.to_ref(float(times[i]), sim_orbit.get_os(float(times[i])))
        E[:, i] = np.asarray(g, dtype=np.float64).reshape(4)
        A[:, i] = real_sat.get_boresight()

    vecsPy = (np.ascontiguousarray(times), to_mat(R), to_mat(V), to_mat(B), to_mat(S),
              A, E, np.zeros(N, dtype=np.float64), to_vec(Rho))

    return planner, planner_settings, vecsPy, x0, N, t_start, t_end, dt, rw_max_torque


def debug_first_iteration():
    """Debug what happens in the first iteration."""
    print("=" * 70)
    print("FIRST ITERATION DETAILED ANALYSIS")
    print("=" * 70)

    planner, settings, vecsPy, x0, N, t_start, t_end, dt, rw_max = setup_and_prepare()
    x0_clean = np.copy(x0.astype(np.float64).flatten(), order='C')
    u_limit = 0.75 * rw_max

    print(f"\nControl limit: {u_limit} Nm = {u_limit*1000} mNm")

    # Get initial trajectory
    print("\nGenerating initial trajectory...")
    (traj, vecs_dt, costSettings) = planner.prepareForAlilqr(
        vecsPy, settings.dt_tp, t_start, t_end, x0_clean, 0
    )

    # Get settings
    auglagSettings = settings.pass1.aug_lag.to_tuple()
    regSettings = settings.pass1.regularization.to_tuple()
    lineSearchSettings = settings.pass1.line_search.to_tuple()

    # Initialize auglag
    (Xset, Uset, _, _) = traj
    num_timesteps = Uset.shape[1]

    test_auglag = (np.zeros((20, num_timesteps), order='F'), 1.0,
                   np.ones((20, num_timesteps), order='F'))
    (clist_test, _) = planner.maxViol(traj, vecs_dt, test_auglag)
    num_c = clist_test.shape[0]

    lam_init, lam_max, mu_init, mu_max, mu_scale = auglagSettings
    lambdas = np.zeros((num_c, num_timesteps), dtype=np.float64, order='F')
    mu = mu_init
    muk = mu * np.ones((num_c, num_timesteps), dtype=np.float64, order='F')
    auglag_vals = (lambdas, mu, muk)
    regs = (regSettings[0], regSettings[0])

    # Initial state
    print("\n" + "=" * 60)
    print("INITIAL STATE")
    print("=" * 60)
    print(f"Xset shape: {Xset.shape}")
    print(f"Uset shape: {Uset.shape}")
    print(f"Initial regularization: {regs}")
    print(f"Initial penalty mu: {mu}")

    print("\nInitial controls (first 10 timesteps, all channels):")
    print(f"  u[0]: {Uset[0, :10]}")
    print(f"  u[1]: {Uset[1, :10]}")
    print(f"  u[2]: {Uset[2, :10]}")

    print(f"\nControl statistics:")
    for ch in range(3):
        u = Uset[ch, :]
        print(f"  u[{ch}]: min={u.min():.4e}, max={u.max():.4e}, mean={u.mean():.4e}, std={u.std():.4e}")

    # Compute initial cost
    cost0 = planner.cost2Func(traj, vecs_dt, auglag_vals, costSettings)
    print(f"\nInitial cost: {cost0:.4e}")

    # Run backward pass
    print("\n" + "=" * 60)
    print("BACKWARD PASS")
    print("=" * 60)

    (bp_results, new_regs) = planner.backwardPass(
        dt, traj, vecs_dt, auglag_vals, regs, costSettings, regSettings, False
    )

    (Kset, dset, Sset) = bp_results

    print(f"Kset shape: {Kset.shape if hasattr(Kset, 'shape') else type(Kset)}")
    print(f"dset shape: {dset.shape}")
    print(f"Sset shape: {Sset.shape if hasattr(Sset, 'shape') else type(Sset)}")
    print(f"New regularization: {new_regs}")

    # Analyze feedforward term d
    print("\n--- Feedforward term d (control update direction) ---")
    print(f"dset statistics:")
    for ch in range(min(3, dset.shape[0])):
        d = dset[ch, :]
        print(f"  d[{ch}]: min={d.min():.4e}, max={d.max():.4e}, mean={d.mean():.4e}, std={d.std():.4e}")

    print(f"\nFirst 10 timesteps of d:")
    print(f"  d[0]: {dset[0, :10]}")
    print(f"  d[1]: {dset[1, :10]}")
    print(f"  d[2]: {dset[2, :10]}")

    # Compute what the update would be with alpha=1
    print("\n--- Expected control update with alpha=1 ---")
    print("New u = old u + K * dx + alpha * d")
    print("At initial state with x = x_ref, the update is approximately: new u ≈ old u + d")

    n_ctrl_steps = min(Uset.shape[1], dset.shape[1])
    u_base = Uset[:3, :n_ctrl_steps]
    d_base = dset[:3, :n_ctrl_steps] if dset.shape[0] >= 3 else dset[:, :n_ctrl_steps]
    expected_new_u = u_base + d_base
    print(f"\nExpected new u[0] (first 10): {expected_new_u[0, :10]}")
    print(f"Expected new u[1] (first 10): {expected_new_u[1, :10]}")
    print(f"Expected new u[2] (first 10): {expected_new_u[2, :10]}")

    # Check if d alone would cause the explosion
    print(f"\n--- Magnitude analysis ---")
    print(f"||d|| / ||u||:")
    for ch in range(3):
        d_norm = np.linalg.norm(dset[ch, :])
        u_norm = np.linalg.norm(Uset[ch, :])
        ratio = d_norm / u_norm if u_norm > 0 else float('inf')
        print(f"  Channel {ch}: ||d||={d_norm:.4e}, ||u||={u_norm:.4e}, ratio={ratio:.4e}")

    # Run forward pass with different alpha values to see step size effect
    print("\n" + "=" * 60)
    print("FORWARD PASS (Line Search)")
    print("=" * 60)
    print(f"Line search settings: {lineSearchSettings}")

    (traj_new, newLA, new_regs) = planner.forwardPass(
        dt, traj, vecs_dt, auglag_vals, bp_results, regs,
        costSettings, regSettings, lineSearchSettings, False
    )

    print(f"\nNew cost after forward pass: {newLA:.4e}")
    print(f"Cost change: {newLA - cost0:.4e} ({(newLA - cost0)/cost0*100:.2f}%)")

    (Xset_new, Uset_new, _, _) = traj_new

    print("\n--- New controls ---")
    print(f"New u[0] (first 10): {Uset_new[0, :10]}")
    print(f"New u[1] (first 10): {Uset_new[1, :10]}")
    print(f"New u[2] (first 10): {Uset_new[2, :10]}")

    print(f"\n--- Control change ---")
    du = Uset_new - Uset
    print(f"du[0] (first 10): {du[0, :10]}")
    print(f"du[1] (first 10): {du[1, :10]}")
    print(f"du[2] (first 10): {du[2, :10]}")

    # Compare du to d to estimate effective alpha
    print(f"\n--- Effective step size (alpha) ---")
    for ch in range(3):
        n_ctrl_steps = min(du.shape[1], dset.shape[1])
        d_ch = dset[ch, :n_ctrl_steps]
        du_ch = du[ch, :n_ctrl_steps]
        # Find alpha such that du ≈ alpha * d (ignoring K*dx term)
        alpha_est = np.dot(du_ch, d_ch) / (np.dot(d_ch, d_ch) + 1e-10)
        print(f"  Channel {ch}: estimated alpha ≈ {alpha_est:.4f}")

    # Final comparison
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nControl magnitude change:")
    for ch in range(3):
        old_max = np.max(np.abs(Uset[ch, :]))
        new_max = np.max(np.abs(Uset_new[ch, :]))
        print(f"  u[{ch}]: {old_max:.4e} -> {new_max:.4e} ({new_max/old_max:.1f}x increase)")

    print(f"\nLimit comparison:")
    print(f"  Limit: {u_limit:.4e}")
    for ch in range(3):
        new_max = np.max(np.abs(Uset_new[ch, :]))
        exceeds = "EXCEEDS!" if new_max > u_limit else "OK"
        print(f"  u[{ch}] max: {new_max:.4e} ({exceeds})")


if __name__ == "__main__":
    debug_first_iteration()
