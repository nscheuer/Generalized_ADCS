"""
Debug script to analyze constraint handling in the ALTRO planner.

Focuses on:
1. Constraint violations at each timestep
2. Active/inactive constraint sets
3. Augmented Lagrangian multiplier evolution
4. How constraints toggle between active/inactive
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

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

import trajectory_planner.build.tplaunch as tplaunch


def setup_problem():
    """Setup the trajectory optimization problem."""
    np.random.seed(1)
    tf = 100
    dt = 1.0

    rw_max_torque = 0.005
    rw_J = 0.0014
    rw_h0 = 0.0
    rw_hmax = 0.015
    rws = [RW(axis=j, max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax)
           for j in MathConstants.unitvecs]

    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]

    real_sat = Satellite(
        mass=10.165,
        J_0=np.diagflat([0.0969, 0.1235, 0.1918]),
        actuators=rws,
        sensors=mtms,
        boresight=np.array([0, 0, 1])
    )

    w0 = np.array([0.0, 0.0, 0.0])
    q0 = normalize(np.array([1, 0, 0, 0]))
    h0 = np.array([rw_h0] * 3)
    x0 = np.concatenate([w0, q0, h0])

    ephem = Ephemeris()
    R = 7000 * np.array([0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    V = np.array([8, 0, 0])

    os0 = Orbital_State(
        ephem=ephem, J2000=0.22 - 1 * TimeConstants.sec2cent, R=R, V=V,
        B=np.array([0, 0.1, 0]), S=np.array([1e5 + 1, 0, 0]), rho=5e-12
    )

    planner_settings = PlannerSettings(est_sat=real_sat, bdot_on=0, dt_tp=1.0)
    planner_settings.verbosity = False
    planner_settings.rw_control_weight = 1e0
    planner_settings.mtq_control_weight = 1e0
    planner_settings.cost_main.ang_vel = 0
    planner_settings.cost_second.ang_vel = 0
    planner_settings.cost_main.use_raw_control_cost = True
    planner_settings.pass1.aug_lag.penalty_init = 1e2

    return {
        'real_sat': real_sat,
        'planner_settings': planner_settings,
        'x0': x0,
        'os0': os0,
        'tf': tf,
        'dt': dt,
        'rw_max_torque': rw_max_torque,
    }


def prepare_environment(setup):
    """Prepare environment vectors."""
    os0 = setup['os0']
    real_sat = setup['real_sat']
    tf = setup['tf']
    dt = setup['dt']

    t_start = 0.22
    t_end = t_start + tf * TimeConstants.sec2cent
    N = int(np.ceil(tf / dt)) + 1

    buffer = 10 * dt * TimeConstants.sec2cent
    sim_orbit = Orbit(os0=os0, end_time=t_end + buffer, dt=dt, use_J2=True, fast=False)
    tp_orbit = sim_orbit.get_range(t_start, t_end, dt)

    orbit_data = tp_orbit.get_vecs()
    times = np.asarray(tp_orbit.times, dtype=np.float64)

    if len(times) > N:
        times = times[:N]
    elif len(times) < N:
        times = np.pad(times, (0, N - len(times)), mode="edge")

    def to_mat(x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2 and x.shape[1] == 3:
            x = x.T
        if x.shape[1] > N:
            x = x[:, :N]
        elif x.shape[1] < N:
            x = np.pad(x, ((0, 0), (0, N - x.shape[1])), mode="edge")
        return np.asfortranarray(x, dtype=np.float64)

    def to_vec(x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if len(x) > N:
            x = x[:N]
        elif len(x) < N:
            x = np.pad(x, (0, N - len(x)), mode="edge")
        return np.ascontiguousarray(x, dtype=np.float64)

    R, V, B, S, Rho = [np.asarray(d) for d in orbit_data]
    R, V, B, S = to_mat(R), to_mat(V), to_mat(B), to_mat(S)
    rho = to_vec(Rho)

    goals = GoalList({0.22: No_Goal(), 0.22 + 3 * TimeConstants.sec2cent: ECI_Goal(np.array([1, 1, 1]))})

    E = np.zeros((3, N), dtype=np.float64, order="F")
    A = np.zeros((3, N), dtype=np.float64, order="F")
    p = np.zeros(N, dtype=np.float64)

    for i in range(N):
        t = float(times[i])
        os_at_t = sim_orbit.get_os(t)
        g_vec, _ = goals.to_ref(t, os_at_t)
        E[:, i] = np.asarray(g_vec, dtype=np.float64).reshape(3)
        A[:, i] = real_sat.boresight

    t_c = np.ascontiguousarray(times, dtype=np.float64)
    p = np.ascontiguousarray(p, dtype=np.float64)

    return (t_c, R, V, B, S, A, E, p, rho), N, t_start, t_end, sim_orbit


def analyze_constraint_handling():
    """
    Main analysis: step through the optimization and analyze constraint handling.
    """
    print("=" * 70)
    print("CONSTRAINT HANDLING ANALYSIS")
    print("=" * 70)

    # Setup
    setup = setup_problem()
    planner_settings = setup['planner_settings']
    x0 = setup['x0']
    rw_max_torque = setup['rw_max_torque']

    # Build planner
    csat = build_cpp_satellite(est_sat=setup['real_sat'], planner_settings=planner_settings)

    planner = DebugPlanner(
        csat,
        planner_settings.systemSettings(),
        planner_settings.mainAlilqrSettings(),
        planner_settings.secondAlilqrSettings(),
        planner_settings.initTrajSettings(),
        planner_settings.optMainCostSettings(),
        planner_settings.optSecondCostSettings(),
        planner_settings.optTVLQRCostSettings(tracking_LQR_formulation=0),
        debug_level=0  # We'll do our own detailed analysis
    )
    planner.setquaternionTo3VecMode(2)
    planner.setVerbosity(False)

    # Prepare environment
    print("\nPreparing environment...")
    vecsPy, N, t_start, t_end, sim_orbit = prepare_environment(setup)
    x0_clean = np.copy(x0.astype(np.float64).flatten(), order='C')
    dt = setup['dt']

    # Get initial trajectory
    print("Generating initial trajectory...")
    (traj, vecs_dt, costSettings) = planner.prepareForAlilqr(
        vecsPy, planner_settings.dt_tp, t_start, t_end, x0_clean, 0
    )

    (Xset, Uset, Tset, _) = traj
    num_timesteps = Uset.shape[1]
    print(f"Trajectory: {num_timesteps} timesteps")

    # Determine constraint dimensions from maxViol
    test_lam = np.zeros((20, num_timesteps), dtype=np.float64, order='F')
    test_muk = np.ones((20, num_timesteps), dtype=np.float64, order='F')
    test_auglag = (test_lam, 1.0, test_muk)
    (clist_test, _) = planner.maxViol(traj, vecs_dt, test_auglag)
    num_constraints = clist_test.shape[0]
    print(f"Number of constraints: {num_constraints}")

    # Constraint labels (based on typical satellite setup)
    # Usually: RW torque upper (3), RW torque lower (3), angular velocity (1)
    constraint_labels = []
    for i in range(min(3, num_constraints)):
        constraint_labels.append(f"RW{i} upper")
    for i in range(min(3, num_constraints - 3)):
        constraint_labels.append(f"RW{i} lower")
    if num_constraints > 6:
        constraint_labels.append("ang_vel")
    while len(constraint_labels) < num_constraints:
        constraint_labels.append(f"c{len(constraint_labels)}")

    # Get settings
    lineSearchSettings = planner_settings.pass1.line_search.to_tuple()
    auglagSettings = planner_settings.pass1.aug_lag.to_tuple()
    breakSettings = planner_settings.pass1.convergence.to_tuple(state_len=setup['real_sat'].state_len)
    regSettings = planner_settings.pass1.regularization.to_tuple()

    # Initialize augmented Lagrangian
    lam_init, lam_max, mu_init, mu_max, mu_scale = auglagSettings
    lambdas = np.zeros((num_constraints, num_timesteps), dtype=np.float64, order='F')
    mu = mu_init
    muk = mu * np.ones((num_constraints, num_timesteps), dtype=np.float64, order='F')
    auglag_vals = (lambdas, mu, muk)

    reg_init = regSettings[0]
    regs = (reg_init, reg_init)

    print(f"\nInitial penalty mu: {mu}")
    print(f"RW max torque: {rw_max_torque} Nm")
    print(f"Control limit (75% of max): {0.75 * rw_max_torque} Nm")

    # Storage for analysis
    iteration_data = []

    # Run manual optimization loop
    num_outer = 5
    num_inner = 10

    print("\n" + "=" * 70)
    print("OPTIMIZATION LOOP")
    print("=" * 70)

    for outer in range(num_outer):
        print(f"\n{'='*60}")
        print(f"OUTER ITERATION {outer}")
        print(f"{'='*60}")

        (lambdas, mu, muk) = auglag_vals
        print(f"Penalty mu: {mu:.2e}")
        print(f"Lambda range: [{lambdas.min():.2e}, {lambdas.max():.2e}]")

        for inner in range(num_inner):
            # Get current cost and violations
            cost = planner.cost2Func(traj, vecs_dt, auglag_vals, costSettings)
            (clist, cmax) = planner.maxViol(traj, vecs_dt, auglag_vals)

            # Analyze constraint activity
            active_mask = clist > 0  # Constraint is violated
            (Xset, Uset, _, _) = traj

            iter_info = {
                'outer': outer,
                'inner': inner,
                'cost': cost,
                'cmax': cmax,
                'mu': mu,
                'Uset': Uset.copy(),
                'clist': clist.copy(),
                'lambdas': lambdas.copy(),
                'active_mask': active_mask.copy(),
            }
            iteration_data.append(iter_info)

            if inner == 0:
                print(f"\n  Inner {inner}: cost={cost:.4e}, cmax={cmax:.4e}")

                # Analyze which constraints are active at each timestep
                for c_idx in range(min(6, num_constraints)):
                    active_count = np.sum(active_mask[c_idx, :])
                    print(f"    {constraint_labels[c_idx]}: active at {active_count}/{num_timesteps} timesteps")

                # Look at control pattern
                print(f"\n  Control analysis:")
                for ch in range(min(3, Uset.shape[0])):
                    u_ch = Uset[ch, :]
                    sign_changes = np.sum(np.diff(np.sign(u_ch)) != 0)
                    print(f"    u[{ch}]: range=[{u_ch.min():.4e}, {u_ch.max():.4e}], sign_changes={sign_changes}")

            # Run backward pass
            (bp_results, regs) = planner.backwardPass(
                dt, traj, vecs_dt, auglag_vals, regs,
                costSettings, regSettings, False
            )

            # Run forward pass
            (traj_new, newLA, regs) = planner.forwardPass(
                dt, traj, vecs_dt, auglag_vals, bp_results, regs,
                costSettings, regSettings, lineSearchSettings, False
            )

            traj = traj_new

        # Update augmented Lagrangian at end of inner loop
        (clist, cmax) = planner.maxViol(traj, vecs_dt, auglag_vals)
        auglag_vals = planner.incrementAugLag(auglag_vals, clist, auglagSettings)
        (lambdas, mu, muk) = auglag_vals

    # Final analysis
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS")
    print("=" * 70)

    (Xset_final, Uset_final, _, _) = traj
    (clist_final, cmax_final) = planner.maxViol(traj, vecs_dt, auglag_vals)

    print(f"\nFinal max violation: {cmax_final:.4e}")
    print(f"Final penalty mu: {mu:.2e}")

    # Control analysis
    print("\nFinal control trajectory:")
    for ch in range(min(3, Uset_final.shape[0])):
        u_ch = Uset_final[ch, :]
        sign_changes = np.sum(np.diff(np.sign(u_ch)) != 0)
        rapid_osc = 0
        signs = np.sign(u_ch)
        for i in range(len(signs) - 2):
            if signs[i] != signs[i+1] and signs[i+1] != signs[i+2]:
                rapid_osc += 1
        print(f"  u[{ch}]: range=[{u_ch.min():.6f}, {u_ch.max():.6f}], "
              f"sign_changes={sign_changes}, rapid_osc={rapid_osc}")

    # Constraint activity over time
    print("\nConstraint activity (final):")
    active_final = clist_final > 0
    for c_idx in range(min(6, num_constraints)):
        active_count = np.sum(active_final[c_idx, :])
        lam_c = lambdas[c_idx, :]
        print(f"  {constraint_labels[c_idx]}: active={active_count}/{num_timesteps}, "
              f"lambda range=[{lam_c.min():.2e}, {lam_c.max():.2e}]")

    # Plot results
    plot_analysis(iteration_data, Uset_final, clist_final, lambdas,
                  constraint_labels, rw_max_torque)

    return iteration_data


def plot_analysis(iteration_data, Uset_final, clist_final, lambdas,
                  constraint_labels, rw_max_torque):
    """Create visualization of the analysis."""

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # 1. Final control trajectory
    ax = axes[0, 0]
    for ch in range(min(3, Uset_final.shape[0])):
        ax.plot(Uset_final[ch, :], label=f'u[{ch}]', alpha=0.7)
    ax.axhline(y=0.75 * rw_max_torque, color='r', linestyle='--', alpha=0.5, label='limit')
    ax.axhline(y=-0.75 * rw_max_torque, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Control (Nm)')
    ax.set_title('Final Control Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Constraint violations over time
    ax = axes[0, 1]
    for c_idx in range(min(6, clist_final.shape[0])):
        ax.plot(clist_final[c_idx, :], label=constraint_labels[c_idx], alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Constraint value')
    ax.set_title('Final Constraint Values (>0 = violated)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Lambda (multipliers) over time
    ax = axes[1, 0]
    for c_idx in range(min(6, lambdas.shape[0])):
        ax.plot(lambdas[c_idx, :], label=constraint_labels[c_idx], alpha=0.7)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Lambda')
    ax.set_title('Lagrange Multipliers')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Cost evolution
    ax = axes[1, 1]
    costs = [d['cost'] for d in iteration_data]
    ax.semilogy(costs)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.set_title('Cost Evolution')
    ax.grid(True, alpha=0.3)

    # 5. Control sign changes over iterations
    ax = axes[2, 0]
    sign_changes = []
    for d in iteration_data:
        Uset = d['Uset']
        sc = [np.sum(np.diff(np.sign(Uset[ch, :])) != 0) for ch in range(min(3, Uset.shape[0]))]
        sign_changes.append(sc)
    sign_changes = np.array(sign_changes)
    for ch in range(sign_changes.shape[1]):
        ax.plot(sign_changes[:, ch], label=f'u[{ch}]')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Sign changes')
    ax.set_title('Control Sign Changes Over Iterations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Active constraint count over iterations
    ax = axes[2, 1]
    active_counts = []
    for d in iteration_data:
        active = d['active_mask']
        counts = [np.sum(active[c_idx, :]) for c_idx in range(min(6, active.shape[0]))]
        active_counts.append(counts)
    active_counts = np.array(active_counts)
    for c_idx in range(min(6, active_counts.shape[1])):
        ax.plot(active_counts[:, c_idx], label=constraint_labels[c_idx], alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Active count')
    ax.set_title('Active Constraint Count Over Iterations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'constraint_analysis.png')
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    analyze_constraint_handling()
