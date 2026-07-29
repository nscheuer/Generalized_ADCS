"""
Debug the backward pass to understand why the costate pk oscillates.
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
    planner_settings.verbosity = False  # Less C++ noise
    # Only disable stiction cost (non-convex Hessian) - use all other defaults
    planner_settings.rw_stic_weight = 0

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

    # Test both goal configurations
    return planner, planner_settings, real_sat, sim_orbit, times, to_mat, to_vec, R, V, B, S, Rho, x0, N, t_start, t_end, dt, rw_max_torque


def run_with_goals(planner, planner_settings, real_sat, sim_orbit, times, to_mat, to_vec,
                   R, V, B, S, Rho, x0, N, t_start, t_end, dt, rw_max, goals, label):
    """Run backward pass with given goals and analyze."""
    print("\n" + "=" * 70)
    print(f"TESTING: {label}")
    print("=" * 70)

    E = np.zeros((3, N), dtype=np.float64, order="F")
    A = np.zeros((3, N), dtype=np.float64, order="F")
    for i in range(N):
        g, _ = goals.to_ref(float(times[i]), sim_orbit.get_os(float(times[i])))
        E[:, i] = np.asarray(g).reshape(3)
        A[:, i] = real_sat.boresight

    vecsPy = (np.ascontiguousarray(times), to_mat(R), to_mat(V), to_mat(B), to_mat(S),
              A, E, np.zeros(N, dtype=np.float64), to_vec(Rho))

    x0_clean = np.copy(x0.astype(np.float64).flatten(), order='C')
    u_limit = 0.75 * rw_max

    # Check the E vector
    print(f"\nGoal vector E (first 15 timesteps):")
    print(f"  E[0,:15] = {E[0,:15]}")
    print(f"  E[1,:15] = {E[1,:15]}")
    print(f"  E[2,:15] = {E[2,:15]}")

    # Check for transitions
    e_norm = np.linalg.norm(E, axis=0)
    transitions = np.where(np.abs(np.diff(e_norm)) > 0.1)[0]
    if len(transitions) > 0:
        print(f"  Goal transitions at timesteps: {transitions}")
    else:
        print(f"  No goal transitions detected")

    # Get initial trajectory
    print("\nPreparing trajectory...")
    (traj, vecs_dt, costSettings) = planner.prepareForAlilqr(
        vecsPy, planner_settings.dt_tp, t_start, t_end, x0_clean, 0
    )

    # Get settings
    auglagSettings = planner_settings.pass1.aug_lag.to_tuple()
    regSettings = planner_settings.pass1.regularization.to_tuple()

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

    print(f"\nInitial state:")
    print(f"  Uset shape: {Uset.shape}")
    print(f"  Control range: [{Uset.min():.2e}, {Uset.max():.2e}]")
    print(f"  Regularization: {regs}")
    print(f"  Penalty mu: {mu}")

    # Run backward pass with verbose output
    print("\n" + "-" * 60)
    print("BACKWARD PASS (C++ verbose output follows)")
    print("-" * 60)

    (bp_results, new_regs) = planner.backwardPass(
        dt, traj, vecs_dt, auglag_vals, regs, costSettings, regSettings, False
    )

    (Kset, dset, Sset) = bp_results

    print("\n" + "-" * 60)
    print("FEEDFORWARD TERM d ANALYSIS")
    print("-" * 60)

    # Analyze d
    print(f"\ndset shape: {dset.shape}")
    print(f"New regularization: {new_regs}")

    for ch in range(min(3, dset.shape[0])):
        d = dset[ch, :]
        sign_changes = np.sum(np.diff(np.sign(d)) != 0)
        print(f"\n  d[{ch}]: min={d.min():.4e}, max={d.max():.4e}, sign_changes={sign_changes}/{len(d)-1}")

    # Show first 20 d values
    print(f"\nFirst 20 timesteps of d:")
    print(f"  k  |    d[0]    |    d[1]    |    d[2]    | signs")
    print("-" * 60)
    for k in range(min(20, dset.shape[1])):
        d0, d1, d2 = dset[0,k], dset[1,k], dset[2,k]
        signs = f"{'+'if d0>0 else '-'}{'+'if d1>0 else '-'}{'+'if d2>0 else '-'}"
        print(f"  {k:2d} | {d0:10.4e} | {d1:10.4e} | {d2:10.4e} | {signs}")

    # Count rapid oscillations in d
    total_rapid_osc = 0
    for ch in range(3):
        signs = np.sign(dset[ch, :])
        for i in range(len(signs) - 2):
            if signs[i] != signs[i+1] and signs[i+1] != signs[i+2]:
                total_rapid_osc += 1

    print(f"\nTotal rapid oscillations in d: {total_rapid_osc}")

    return dset


def main():
    print("=" * 70)
    print("BACKWARD PASS OSCILLATION ANALYSIS")
    print("=" * 70)

    setup = setup_and_prepare()
    planner, planner_settings, real_sat, sim_orbit, times, to_mat, to_vec, R, V, B, S, Rho, x0, N, t_start, t_end, dt, rw_max = setup

    # Test 1: Original goal setup (No_Goal -> ECI_Goal transition)
    goals1 = GoalList({0.22: No_Goal(), 0.22+3*TimeConstants.sec2cent: ECI_Goal(np.array([1,1,1]))})
    d1 = run_with_goals(planner, planner_settings, real_sat, sim_orbit, times, to_mat, to_vec,
                        R, V, B, S, Rho, x0, N, t_start, t_end, dt, rw_max,
                        goals1, "No_Goal -> ECI_Goal transition")

    # Test 2: Simple ECI_Goal from start
    goals2 = GoalList({0.22: ECI_Goal(np.array([1,1,1]))})
    d2 = run_with_goals(planner, planner_settings, real_sat, sim_orbit, times, to_mat, to_vec,
                        R, V, B, S, Rho, x0, N, t_start, t_end, dt, rw_max,
                        goals2, "ECI_Goal from start (no transition)")

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    for ch in range(3):
        sc1 = np.sum(np.diff(np.sign(d1[ch, :])) != 0)
        sc2 = np.sum(np.diff(np.sign(d2[ch, :])) != 0)
        print(f"d[{ch}] sign changes: with_transition={sc1}, no_transition={sc2}")


if __name__ == "__main__":
    main()
