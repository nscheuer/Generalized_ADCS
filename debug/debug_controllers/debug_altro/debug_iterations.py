"""
Debug multiple iterations to see how the solution evolves.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

from ADCS.CONOPS.goals import ECI_Goal
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


def main():
    print("=" * 70)
    print("ITERATION-BY-ITERATION ANALYSIS")
    print("=" * 70)

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

    # Use defaults, only disable stiction (non-convex)
    planner_settings = PlannerSettings(est_sat=real_sat, bdot_on=0, dt_tp=1.0)
    planner_settings.verbosity = False
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

    # Simple goal - ECI from start
    goals = GoalList({0.22: ECI_Goal(np.array([1,1,1]))})

    E = np.zeros((4, N), dtype=np.float64, order="F")
    A = np.zeros((3, N), dtype=np.float64, order="F")
    for i in range(N):
        g, _ = goals.to_ref(float(times[i]), sim_orbit.get_os(float(times[i])))
        E[:, i] = np.asarray(g, dtype=np.float64).reshape(4)
        A[:, i] = real_sat.get_boresight()

    vecsPy = (np.ascontiguousarray(times), to_mat(R), to_mat(V), to_mat(B), to_mat(S),
              A, E, np.zeros(N, dtype=np.float64), to_vec(Rho))

    x0_clean = np.copy(x0.astype(np.float64).flatten(), order='C')
    u_limit = 0.75 * rw_max_torque

    print(f"\nControl limit: {u_limit} Nm")
    print(f"Default rw_control_weight: {planner_settings.rw_control_weight}")
    print(f"Default ang_vel cost: {planner_settings.cost_main.ang_vel}")
    print(f"Default penalty_init: {planner_settings.pass1.aug_lag.penalty_init}")

    # Prepare
    (traj, vecs_dt, costSettings) = planner.prepareForAlilqr(
        vecsPy, planner_settings.dt_tp, t_start, t_end, x0_clean, 0
    )

    auglagSettings = planner_settings.pass1.aug_lag.to_tuple()
    regSettings = planner_settings.pass1.regularization.to_tuple()
    lineSearchSettings = planner_settings.pass1.line_search.to_tuple()

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

    def analyze_controls(Uset):
        """Count sign changes and rapid oscillations."""
        total_sc = 0
        total_ro = 0
        for ch in range(min(3, Uset.shape[0])):
            u = Uset[ch, :]
            sc = np.sum(np.diff(np.sign(u)) != 0)
            total_sc += sc
            signs = np.sign(u)
            for i in range(len(signs) - 2):
                if signs[i] != signs[i+1] and signs[i+1] != signs[i+2]:
                    total_ro += 1
        return total_sc, total_ro

    # Run iterations
    print("\n" + "=" * 70)
    print("INNER ITERATIONS (fixed mu)")
    print("=" * 70)
    print(f"{'Iter':>4} | {'Cost':>12} | {'cmax':>10} | {'SC':>4} | {'RO':>4} | {'|u|_max':>10} | {'rho':>10}")
    print("-" * 70)

    (Xset, Uset, _, _) = traj
    sc, ro = analyze_controls(Uset)
    cost = planner.cost2Func(traj, vecs_dt, auglag_vals, costSettings)
    (_, cmax) = planner.maxViol(traj, vecs_dt, auglag_vals)
    u_max = np.max(np.abs(Uset))
    print(f"{'init':>4} | {cost:12.4e} | {cmax:10.4e} | {sc:4d} | {ro:4d} | {u_max:10.4e} | {regs[0]:10.4e}")

    for it in range(30):
        # Backward pass
        (bp_results, regs) = planner.backwardPass(
            dt, traj, vecs_dt, auglag_vals, regs, costSettings, regSettings, False
        )

        # Forward pass
        (traj, cost, regs) = planner.forwardPass(
            dt, traj, vecs_dt, auglag_vals, bp_results, regs,
            costSettings, regSettings, lineSearchSettings, False
        )

        (Xset, Uset, _, _) = traj
        sc, ro = analyze_controls(Uset)
        (_, cmax) = planner.maxViol(traj, vecs_dt, auglag_vals)
        u_max = np.max(np.abs(Uset))

        print(f"{it+1:4d} | {cost:12.4e} | {cmax:10.4e} | {sc:4d} | {ro:4d} | {u_max:10.4e} | {regs[0]:10.4e}")

        if it >= 5 and ro > 50:
            print("  --> Still oscillating after 5 iterations!")

    # Final analysis
    print("\n" + "=" * 70)
    print("FINAL CONTROLS AND STATES")
    print("=" * 70)
    print("First 15 timesteps (w = angular velocity, q = quaternion 1-3):")
    print(f"{'k':>3} | {'u[0]':>10} | {'u[1]':>10} | {'u[2]':>10} | {'w[0]':>10} | {'w[1]':>10} | {'w[2]':>10}")
    print("-" * 80)
    for k in range(15):
        print(f"{k:3d} | {Uset[0,k]:10.6f} | {Uset[1,k]:10.6f} | {Uset[2,k]:10.6f} | {Xset[0,k]:10.6f} | {Xset[1,k]:10.6f} | {Xset[2,k]:10.6f}")

    # Check angular velocity oscillation
    w_sc = [np.sum(np.diff(np.sign(Xset[i,:])) != 0) for i in range(3)]
    print(f"\nAngular velocity sign changes: w[0]={w_sc[0]}, w[1]={w_sc[1]}, w[2]={w_sc[2]}")


if __name__ == "__main__":
    main()
