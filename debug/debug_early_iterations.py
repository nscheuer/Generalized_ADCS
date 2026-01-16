"""
Debug the critical early iterations where oscillation emerges.

Focus on iterations 0-15 to understand why the algorithm converges TO oscillation.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

from ADCS.CONOPS.goals import ECI_Goal, No_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.helpers import PlannerSettings, DebugPlanner
from ADCS.controller.helpers.build_csat import build_cpp_satellite
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

    sim_orbit = Orbit(os0=os0, end_time=t_end + 10*dt*TimeConstants.sec2cent, dt=dt, use_J2=True, fast=False)
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
    goals = GoalList({0.22: No_Goal(), 0.22+3*TimeConstants.sec2cent: ECI_Goal(np.array([1,1,1]))})

    E = np.zeros((3, N), dtype=np.float64, order="F")
    A = np.zeros((3, N), dtype=np.float64, order="F")
    for i in range(N):
        g, _ = goals.to_ref(float(times[i]), sim_orbit.get_os(float(times[i])))
        E[:, i] = np.asarray(g).reshape(3)
        A[:, i] = real_sat.boresight

    vecsPy = (np.ascontiguousarray(times), to_mat(R), to_mat(V), to_mat(B), to_mat(S),
              A, E, np.zeros(N, dtype=np.float64), to_vec(Rho))

    return planner, planner_settings, vecsPy, x0, N, t_start, t_end, dt, rw_max_torque


def analyze_early_iterations():
    """Analyze iterations 0-15 in detail."""
    print("=" * 70)
    print("EARLY ITERATION ANALYSIS")
    print("=" * 70)

    planner, settings, vecsPy, x0, N, t_start, t_end, dt, rw_max = setup_and_prepare()
    x0_clean = np.copy(x0.astype(np.float64).flatten(), order='C')
    u_limit = 0.75 * rw_max

    print(f"\nControl limit: {u_limit} Nm")

    # Get initial trajectory
    print("\nGenerating initial trajectory...")
    (traj, vecs_dt, costSettings) = planner.prepareForAlilqr(
        vecsPy, settings.dt_tp, t_start, t_end, x0_clean, 0
    )

    # Get settings
    auglagSettings = settings.pass1.aug_lag.to_tuple()
    regSettings = settings.pass1.regularization.to_tuple()
    lineSearchSettings = settings.pass1.line_search.to_tuple()

    # Initialize
    (Xset, Uset, _, _) = traj
    num_timesteps = Uset.shape[1]

    # Probe constraint count
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

    def analyze_controls(Uset, label=""):
        """Analyze control pattern."""
        results = []
        for ch in range(min(3, Uset.shape[0])):
            u = Uset[ch, :]
            sc = np.sum(np.diff(np.sign(u)) != 0)
            signs = np.sign(u)
            ro = sum(1 for i in range(len(signs)-2) if signs[i] != signs[i+1] and signs[i+1] != signs[i+2])
            at_limit = np.sum(np.abs(u) > 0.95 * u_limit)
            results.append({'ch': ch, 'sc': sc, 'ro': ro, 'at_limit': at_limit,
                           'min': u.min(), 'max': u.max()})
        return results

    # Print initial state
    print("\n" + "-" * 60)
    print("ITERATION 0 (Initial)")
    print("-" * 60)
    (Xset, Uset, _, _) = traj
    cost = planner.cost2Func(traj, vecs_dt, auglag_vals, costSettings)
    (clist, cmax) = planner.maxViol(traj, vecs_dt, auglag_vals)
    print(f"Cost: {cost:.4e}, cmax: {cmax:.4e}")

    analysis = analyze_controls(Uset)
    for r in analysis:
        print(f"  u[{r['ch']}]: range=[{r['min']:.4e}, {r['max']:.4e}], "
              f"sign_changes={r['sc']}, rapid_osc={r['ro']}, at_limit={r['at_limit']}")

    # Show first 10 control values for channel 0
    print(f"\n  First 10 u[0] values:")
    print(f"    {Uset[0, :10]}")

    # Run iterations and track evolution
    print("\n" + "=" * 70)
    print("ITERATION-BY-ITERATION TRACKING")
    print("=" * 70)

    for iter_num in range(20):
        # Backward pass
        (bp_results, regs) = planner.backwardPass(
            dt, traj, vecs_dt, auglag_vals, regs, costSettings, regSettings, False
        )

        # Forward pass
        (traj_new, newLA, regs) = planner.forwardPass(
            dt, traj, vecs_dt, auglag_vals, bp_results, regs,
            costSettings, regSettings, lineSearchSettings, False
        )

        traj = traj_new
        (Xset, Uset, _, _) = traj
        (clist, cmax) = planner.maxViol(traj, vecs_dt, auglag_vals)

        analysis = analyze_controls(Uset)

        # Print summary
        sc_total = sum(r['sc'] for r in analysis)
        ro_total = sum(r['ro'] for r in analysis)
        at_limit_total = sum(r['at_limit'] for r in analysis)
        u_max_all = max(abs(r['max']) for r in analysis)
        u_min_all = min(r['min'] for r in analysis)

        exceeded = "EXCEEDS!" if u_max_all > u_limit or abs(u_min_all) > u_limit else ""

        print(f"Iter {iter_num+1:2d}: cost={newLA:.4e}, cmax={cmax:.4e}, "
              f"sc={sc_total}, ro={ro_total}, at_lim={at_limit_total}, "
              f"u_range=[{u_min_all:.4e}, {u_max_all:.4e}] {exceeded}")

        # Detailed output for key iterations
        if iter_num in [0, 4, 9, 14, 19]:
            print(f"         First 10 u[0]: {Uset[0, :10]}")

        # Check for constraint activity changes
        active = clist > 0
        active_count = [np.sum(active[c, :]) for c in range(min(6, num_c))]
        if any(ac > 0 for ac in active_count):
            print(f"         Active constraints: {active_count[:6]}")

    # Final state
    print("\n" + "=" * 70)
    print("AFTER 20 INNER ITERATIONS")
    print("=" * 70)

    (Xset, Uset, _, _) = traj
    for ch in range(3):
        u = Uset[ch, :]
        print(f"\nu[{ch}] full trajectory:")
        print(f"  {u}")


if __name__ == "__main__":
    analyze_early_iterations()
