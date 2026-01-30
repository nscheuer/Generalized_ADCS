#!/usr/bin/env python3
"""
Quick Monte Carlo test with best planner settings.
Tests both 90° and 180° slews with MTQ+RW configuration.
"""
import sys
import os
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../..")))

from scipy.integrate import solve_ivp
from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import normalize, quat_mult

from mc_planner_settings import create_best_planner_settings, create_good_planner_settings, create_adaptive_planner_settings


def run_single_test(seed, slew_angle_deg=90, duration=500, settings_func=create_adaptive_planner_settings):
    """Run a single test and return results."""
    np.random.seed(seed)
    
    sat = create_beavercube2_cubesat(estimated=False)
    orb = create_random_circular_orbit(radius_km=7000.0, dt=1, tf=duration+100, use_J2=True, fast=True)
    
    # Initial state
    q0 = normalize(np.random.randn(4))
    w0 = np.random.randn(3) * 0.5 * np.pi / 180
    h0 = np.array([np.random.uniform(-0.001, 0.001)])
    
    # Goal: rotation by slew_angle about random axis
    half_angle = slew_angle_deg * np.pi / 360  # half angle in radians
    axis = normalize(np.random.randn(3))
    q_rot = np.concatenate([[np.cos(half_angle)], np.sin(half_angle) * axis])
    q_goal = normalize(quat_mult(q0, q_rot))
    
    x0 = np.concatenate([w0, q0, h0])
    sat.rw_actuators[0].h = h0[0]
    
    # Create controller with specified settings
    # Handle both adaptive (needs duration) and non-adaptive settings functions
    if settings_func == create_adaptive_planner_settings:
        settings = settings_func(sat, duration=duration, dt_planning=1.0)
    else:
        settings = settings_func(sat, dt_planning=1.0)
    controller = Plan_and_Track_LQR(est_sat=sat, planner_settings=settings)
    
    t_start = orb.times[10]
    goals = GoalList({t_start: Fixed_Attitude_Goal(q_goal)})
    os0 = orb.get_os(t_start)
    
    # Plan trajectory
    plan_t0 = time.perf_counter()
    try:
        traj = controller.calculate_trajectory(
            t_start=t_start, duration=duration, x_0=x0, os_0=os0, goals=goals, verbose=False
        )
        plan_time = time.perf_counter() - plan_t0
        
        if traj is None:
            return {'error': 180.0, 'plan_time': plan_time, 'final_error': 180.0, 'valid': False}
        
        controller.set_active_trajectory(traj)
    except Exception as e:
        return {'error': 180.0, 'plan_time': 0, 'final_error': 180.0, 'valid': False, 'exception': str(e)}
    
    # Simulate closed-loop
    dt_sim = 2.0
    N = int(duration / dt_sim)
    x = x0.copy()
    t = 0
    sec2cent = TimeConstants.sec2cent
    
    for rw in sat.rw_actuators:
        rw.h = h0[0]
    
    for i in range(N):
        J2000 = t_start + t * sec2cent
        os_state = orb.get_os(J2000=J2000)
        sens = sat.sensor_readings(x=x, os=os_state)
        u = controller.find_u(x_hat=x, sens=sens, est_sat=sat, os_hat=os_state)
        
        t += dt_sim
        os_next = orb.get_os(t_start + t * sec2cent)
        out = solve_ivp(
            sat.dynamics_for_solver, (0, dt_sim), x, method="RK45",
            args=(u, os_state, os_next), rtol=1e-6, atol=1e-6
        )
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])
    
    # Compute final error
    q_final = x[3:7] / np.linalg.norm(x[3:7])
    final_error = np.degrees(2 * np.arccos(min(abs(np.dot(q_final, q_goal)), 1.0)))
    
    # Also get planned trajectory final error
    q_traj_final = traj.get_state_at(traj.end_time)[3:7]
    q_traj_final = q_traj_final / np.linalg.norm(q_traj_final)
    traj_error = np.degrees(2 * np.arccos(min(abs(np.dot(q_traj_final, q_goal)), 1.0)))
    
    return {
        'plan_time': plan_time,
        'traj_error': traj_error,
        'final_error': final_error,
        'valid': True
    }


def run_mc_test(name, settings_func, n_runs=10, slew_angle=90, duration=500):
    """Run Monte Carlo test with given settings."""
    print(f"\n{'='*60}")
    print(f"{name}: {n_runs} runs, {slew_angle}° slew, {duration}s")
    print('='*60)
    
    results = []
    for i in range(n_runs):
        seed = 1000 + i
        result = run_single_test(seed, slew_angle_deg=slew_angle, duration=duration, settings_func=settings_func)
        results.append(result)
        
        status = '✓' if result['valid'] else '✗'
        print(f"  Run {i+1:2d}: {status} plan={result['plan_time']:.1f}s, traj_err={result.get('traj_error', 180):.1f}°, final_err={result['final_error']:.1f}°")
    
    valid = [r for r in results if r['valid']]
    if valid:
        plan_times = [r['plan_time'] for r in valid]
        traj_errors = [r['traj_error'] for r in valid]
        final_errors = [r['final_error'] for r in valid]
        
        print(f"\nSummary ({len(valid)}/{n_runs} valid):")
        print(f"  Plan time:  {np.mean(plan_times):.1f}s ± {np.std(plan_times):.1f}s")
        print(f"  Traj error: {np.mean(traj_errors):.1f}° ± {np.std(traj_errors):.1f}° (median {np.median(traj_errors):.1f}°)")
        print(f"  Final error: {np.mean(final_errors):.1f}° ± {np.std(final_errors):.1f}° (median {np.median(final_errors):.1f}°)")
        print(f"  <10°: {sum(1 for e in final_errors if e < 10)}/{len(valid)}")
        print(f"  <30°: {sum(1 for e in final_errors if e < 30)}/{len(valid)}")
        print(f"  <60°: {sum(1 for e in final_errors if e < 60)}/{len(valid)}")
    
    return results


if __name__ == "__main__":
    N_RUNS = 10
    
    # Test 90° slews
    print("\n" + "="*70)
    print("TESTING 90° SLEWS (300s)")
    print("="*70)
    
    run_mc_test("Adaptive Settings (auto-tuned)", create_adaptive_planner_settings, n_runs=N_RUNS, slew_angle=90, duration=300)
    run_mc_test("Best Settings (manual)", create_best_planner_settings, n_runs=N_RUNS, slew_angle=90, duration=300)
    run_mc_test("Good Settings (old)", create_good_planner_settings, n_runs=N_RUNS, slew_angle=90, duration=300)
    
    # Test 180° slews  
    print("\n" + "="*70)
    print("TESTING 180° SLEWS (500s)")
    print("="*70)
    
    run_mc_test("Adaptive Settings (auto-tuned)", create_adaptive_planner_settings, n_runs=N_RUNS, slew_angle=180, duration=500)
    run_mc_test("Best Settings (manual)", create_best_planner_settings, n_runs=N_RUNS, slew_angle=180, duration=500)
    run_mc_test("Good Settings (old)", create_good_planner_settings, n_runs=N_RUNS, slew_angle=180, duration=500)
