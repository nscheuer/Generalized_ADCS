#!/usr/bin/env python3
"""
Test script to save multi-goal planner plots for debugging.
Usage: python test_multi_goal_plots.py [--seed SEED]
"""
import numpy as np
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
sys.path.insert(0, 'papers/Planner')

from mc_planner_settings_experiment import create_optimized_planner_settings
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.CONOPS.goallist import GoalList
from ADCS.CONOPS.goals import ECI_Goal, No_Goal
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_helpers import normalize, rot_exp, rot_mat
from ADCS.helpers.plotting_mc.plot_controller_mc import plot_planned_trajectory, plot_single_run
from scipy.integrate import solve_ivp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    
    seed = args.seed
    print(f"=== Testing multi-goal plots with seed={seed} ===")
    
    sec2cent = TimeConstants.sec2cent
    BODY = np.array([0, 1, 0])
    tf = 1000
    dt = 1.0

    rng = np.random.default_rng(seed + 1000)
    q0 = normalize(rng.standard_normal(4))
    w0 = normalize(rng.standard_normal(3)) * 0.005
    g1 = normalize(rng.standard_normal(3))
    g2 = rot_mat(rot_exp((np.pi/2) * normalize(rng.standard_normal(3)))) @ g1
    g3 = rot_mat(rot_exp((np.pi/2) * normalize(rng.standard_normal(3)))) @ g2

    sat = create_beavercube2_cubesat(estimated=False)
    h0 = np.array([rng.uniform(-0.0001, 0.0001)])
    x0 = np.concatenate([w0, q0, h0])
    for i, rw in enumerate(sat.rw_actuators):
        rw.h = h0[i]

    config = {'goal1': g1, 'goal2': g2, 'goal3': g3}

    settings = create_optimized_planner_settings(sat, duration=tf, dt_planning=1.0, tuning='fast_slew', has_rw=True)
    settings.verbosity = False
    settings.dt_tp = 10
    settings.skip_pass2_optimization = True

    controller = Plan_and_Track_LQR(est_sat=sat, planner_settings=settings)

    t0_j2000 = 0.22
    goals = GoalList({
        t0_j2000: ECI_Goal(g1),
        t0_j2000 + 350 * sec2cent: No_Goal(),
        t0_j2000 + 550 * sec2cent: ECI_Goal(g2),
        t0_j2000 + 700 * sec2cent: No_Goal(),
        t0_j2000 + 900 * sec2cent: ECI_Goal(g3),
    })

    np.random.seed(seed + 42)
    orb = create_random_circular_orbit(radius_km=7000.0, dt=1.0, tf=tf, use_J2=True, fast=True)
    orb.populate_environment(compute_B=True, compute_S=True)
    os0 = orb.get_os(t0_j2000)

    print('Planning...')
    traj = controller.calculate_trajectory(t_start=t0_j2000, duration=tf, x_0=x0, os_0=os0, goals=goals, verbose=False)
    controller.set_active_trajectory(traj)
    print(f'  Trajectory: {traj.states.shape[1]} states, {traj.gains.shape[1]} gains')

    print('Saving post-planning plot...')
    plot_planned_trajectory(traj, config, BODY, title_prefix=f'Post-Planning (seed={seed})', 
                           save_path=f'/tmp/post_planning_seed{seed}.png', goals=goals)
    print(f'  Saved to /tmp/post_planning_seed{seed}.png')

    print('Running tracking simulation...')
    N = int(tf / dt)
    time_hist = np.zeros(N)
    state_hist = np.zeros((N, len(x0)))
    u_hist = np.zeros((N, len(sat.actuators)))
    boresight_hist = np.zeros((N, 3))

    x = x0.copy()
    for i in range(N):
        J2000 = t0_j2000 + i * dt * sec2cent
        os_state = orb.get_os(J2000=J2000)
        sens = sat.sensor_readings(x=x, os=os_state)
        u = controller.find_u(x_hat=x, sens=sens, est_sat=sat, os_hat=os_state)
        
        time_hist[i] = i * dt
        state_hist[i, :] = x
        u_hist[i, :] = u
        eci_goal_ref, _ = goals.to_ref(t=J2000, os0=os_state)
        boresight_hist[i, :] = eci_goal_ref
        
        os_next = orb.get_os(t0_j2000 + (i+1) * dt * sec2cent)
        out = solve_ivp(sat.dynamics_for_solver, (0, dt), x, method='RK45', 
                        args=(u, os_state, os_next), rtol=1e-6, atol=1e-6)
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])
    print('  Tracking complete')

    traj_times_sec = (traj.times - traj.times[0]) * 36525 * 24 * 3600
    traj_state = traj.states.T
    traj_u = traj.controls.T

    result = {
        'time': time_hist, 'state': state_hist, 'u': u_hist,
        'boresight_goal': boresight_hist,
        'traj_time': traj_times_sec, 'traj_state': traj_state, 'traj_u': traj_u
    }

    print('Saving post-sim plot...')
    fig = plot_single_run(result, body_boresight=BODY, title_prefix=f'Post-Sim (seed={seed})', show=False)
    fig.savefig(f'/tmp/post_sim_seed{seed}.png', dpi=150, bbox_inches='tight')
    print(f'  Saved to /tmp/post_sim_seed{seed}.png')

    print(f'\n=== Done! Plots saved to: ===')
    print(f'  /tmp/post_planning_seed{seed}.png')
    print(f'  /tmp/post_sim_seed{seed}.png')

if __name__ == '__main__':
    main()
