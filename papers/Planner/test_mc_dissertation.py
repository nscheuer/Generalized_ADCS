"""
Quick MC test with dissertation-equivalent settings.
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, '/home/pmckeen/Generalized_ADCS')
sys.path.insert(0, '/home/pmckeen/Generalized_ADCS/papers/Planner')

from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import create_planner_settings
from ADCS.controller.helpers.normalized_settings import PlannerPresets
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import normalize, quat_mult

import warnings
warnings.filterwarnings('ignore')

NUM_RUNS = 5

# Create orbit once
np.random.seed(42)
print("Creating orbit...", flush=True)
t0 = time.time()
orb = create_random_circular_orbit(radius_km=7000.0, dt=1, tf=1000, use_J2=True, fast=True)
print(f"Orbit created in {time.time()-t0:.2f}s")

results = []

for run_id in range(NUM_RUNS):
    seed = 1000 + run_id
    np.random.seed(seed)
    
    # Create satellite fresh each run
    real_sat = create_beavercube2_cubesat(estimated=False)
    
    # Random initial state
    q0 = normalize(np.random.randn(4))
    w0 = np.random.randn(3) * 0.5 * np.pi / 180
    h0 = np.array([np.random.uniform(-0.001, 0.001)])
    
    # 90 degree slew
    half_angle = 45 * np.pi / 180
    axis = normalize(np.random.randn(3))
    q_rot = np.concatenate([[np.cos(half_angle)], np.sin(half_angle) * axis])
    q_goal = normalize(quat_mult(q0, q_rot))
    
    x0 = np.concatenate([w0, q0, h0])
    for i, rw in enumerate(real_sat.rw_actuators):
        rw.h = h0[i]
    
    # Use the dissertation equivalent preset
    config = PlannerPresets.dissertation_equivalent()
    planner_settings = create_planner_settings(real_sat, config)
    
    # Key settings
    planner_settings.bdot_on = 0
    planner_settings.dt_tp = 10
    planner_settings.dt_tvlqr = 1
    
    # Convergence settings
    planner_settings.pass1.convergence.max_outer_iter = 10
    planner_settings.pass1.convergence.max_inner_iter = 15
    planner_settings.pass2.convergence.max_outer_iter = 8
    planner_settings.pass2.convergence.max_inner_iter = 15
    
    # Augmented Lagrangian settings
    planner_settings.pass1.aug_lag.penalty_init = 1.0
    planner_settings.pass1.aug_lag.penalty_max = 1e6
    planner_settings.pass2.aug_lag.penalty_init = 1e4
    planner_settings.pass2.aug_lag.penalty_max = 1e16
    
    controller = Plan_and_Track_LQR(est_sat=real_sat, planner_settings=planner_settings)
    
    # Plan
    t_start = 0.22
    os0 = orb.get_os(t_start)
    goals = GoalList({t_start: Fixed_Attitude_Goal(q_goal)})
    
    plan_t0 = time.time()
    traj = controller.calculate_trajectory(
        t_start=t_start,
        duration=120,
        x_0=x0,
        os_0=os0,
        goals=goals,
        verbose=False
    )
    plan_time = time.time() - plan_t0
    
    # Check result
    if traj is not None:
        final_state = traj.get_state_at(traj.end_time)
        q_final = final_state[3:7]
        q_final = q_final / np.linalg.norm(q_final)
        dot = abs(np.dot(q_final, q_goal))
        angle_err = np.degrees(2 * np.arccos(min(dot, 1.0)))
    else:
        angle_err = 180.0
    
    results.append({
        'run_id': run_id,
        'plan_time': plan_time,
        'angle_err': angle_err
    })
    
    print(f"Run {run_id+1}/{NUM_RUNS}: plan_time={plan_time:.2f}s, final_err={angle_err:.1f}°")

# Summary
plan_times = [r['plan_time'] for r in results]
errors = [r['angle_err'] for r in results]

print(f"\n=== Summary ({NUM_RUNS} runs) - Dissertation Equivalent ===")
print(f"Plan time: {np.mean(plan_times):.2f}s ± {np.std(plan_times):.2f}s (min={min(plan_times):.2f}s, max={max(plan_times):.2f}s)")
print(f"Final error: {np.mean(errors):.1f}° ± {np.std(errors):.1f}° (min={min(errors):.1f}°, max={max(errors):.1f}°)")
