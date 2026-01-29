#!/usr/bin/env python3
"""Benchmark planner with current solver options."""
import sys
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/pmckeen/Generalized_ADCS')

from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import PlannerSettings
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.helpers.math_helpers import normalize, quat_mult


def benchmark(n_runs=5, duration=120):
    """Run benchmark and return timing stats."""
    times = []
    errors = []
    
    for seed in range(1000, 1000 + n_runs):
        np.random.seed(seed)
        sat = create_beavercube2_cubesat(estimated=False)
        orb = create_random_circular_orbit(radius_km=7000.0, dt=1, tf=duration+100, use_J2=True, fast=True)
        
        q0 = normalize(np.random.randn(4))
        w0 = np.random.randn(3) * 0.5 * np.pi / 180
        h0 = np.array([np.random.uniform(-0.001, 0.001)])
        axis = normalize(np.random.randn(3))
        q_rot = np.concatenate([[np.cos(np.pi/4)], np.sin(np.pi/4) * axis])
        q_goal = normalize(quat_mult(q0, q_rot))
        x0 = np.concatenate([w0, q0, h0])
        sat.rw_actuators[0].h = h0[0]
        
        settings = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=10, dt_tvlqr=1)
        settings.verbosity = False
        # Use optimized weights
        settings.cost_main.angle = 200
        settings.cost_main.angle_N = 200
        settings.cost_main.ang_vel_N = 1000
        
        controller = Plan_and_Track_LQR(est_sat=sat, planner_settings=settings)
        t_start = orb.times[10]
        goals = GoalList({t_start: Fixed_Attitude_Goal(q_goal)})
        
        try:
            t0 = time.perf_counter()
            traj = controller.calculate_trajectory(t_start, duration, x0, orb.get_os(t_start), goals, verbose=False)
            elapsed = time.perf_counter() - t0
            
            if traj:
                q_final = traj.get_state_at(traj.end_time)[3:7]
                q_final = q_final / np.linalg.norm(q_final)
                err = np.degrees(2 * np.arccos(min(abs(np.dot(q_final, q_goal)), 1.0)))
            else:
                err = 180.0
        except Exception as e:
            elapsed = 0
            err = 180.0
            
        times.append(elapsed)
        errors.append(err)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'mean_err': np.mean(errors),
        'std_err': np.std(errors),
    }


if __name__ == "__main__":
    print("Benchmarking C++ planner (5 runs)...")
    results = benchmark(n_runs=5)
    print(f"Time: {results['mean_time']:.3f}s ± {results['std_time']:.3f}s")
    print(f"Error: {results['mean_err']:.1f}° ± {results['std_err']:.1f}°")
