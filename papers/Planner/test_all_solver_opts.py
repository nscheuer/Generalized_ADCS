#!/usr/bin/env python3
"""Test all Armadillo solve() options with proper rebuilds."""
import subprocess
import sys
import os
import re
import time

PLANNER_CPP = "/home/pmckeen/Generalized_ADCS/trajectory_planner/src/planner/OldPlanner.cpp"
BACKUP_CPP = "/home/pmckeen/Generalized_ADCS/trajectory_planner/src/planner/OldPlanner.cpp.backup"
BUILD_DIR = "/home/pmckeen/Generalized_ADCS/trajectory_planner/build"

def update_solver_opts(opts_str):
    """Update the solve() calls in OldPlanner.cpp."""
    with open(PLANNER_CPP, 'r') as f:
        content = f.read()
    
    # Pattern to match the solve calls on lines 2166 and 2170
    # Match: solve(Kk,Qkuureg, Qkux, <anything up to >);
    pattern_kk = r'(reset \|= !solve\(Kk,Qkuureg, Qkux),([^)]*)\);'
    pattern_dk = r'(reset \|= !solve\(dk,Qkuureg, Qku),([^)]*)\);'
    
    if opts_str:
        replacement = rf'\1,{opts_str});'
    else:
        # No options - remove the comma and options
        replacement = r'\1);'
    
    content = re.sub(pattern_kk, replacement, content)
    content = re.sub(pattern_dk, replacement, content)
    
    with open(PLANNER_CPP, 'w') as f:
        f.write(content)


def rebuild():
    """Rebuild the C++ planner."""
    result = subprocess.run(
        ["make", "-j4"],
        cwd=BUILD_DIR,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Build failed: {result.stderr}")
        return False
    return True


def run_benchmark():
    """Run the benchmark and return results."""
    import numpy as np
    sys.path.insert(0, '/home/pmckeen/Generalized_ADCS')
    
    # Force reimport of the module
    import importlib
    if 'trajectory_planner' in sys.modules:
        # Remove all trajectory_planner submodules
        to_remove = [k for k in sys.modules if k.startswith('trajectory_planner')]
        for k in to_remove:
            del sys.modules[k]
    
    # Also remove ADCS modules that use trajectory_planner
    to_remove = [k for k in sys.modules if 'ADCS' in k]
    for k in to_remove:
        del sys.modules[k]
    
    import warnings
    warnings.filterwarnings('ignore')
    
    from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
    from ADCS.CONOPS.goals import Fixed_Attitude_Goal
    from ADCS.CONOPS.goallist import GoalList
    from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
    from ADCS.controller.helpers import PlannerSettings
    from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
    from ADCS.helpers.math_helpers import normalize, quat_mult
    
    n_runs = 5
    times = []
    errors = []
    
    for seed in range(1000, 1000 + n_runs):
        np.random.seed(seed)
        sat = create_beavercube2_cubesat(estimated=False)
        orb = create_random_circular_orbit(radius_km=7000.0, dt=1, tf=220, use_J2=True, fast=True)
        
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
        settings.cost_main.angle = 200
        settings.cost_main.angle_N = 200
        settings.cost_main.ang_vel_N = 1000
        
        controller = Plan_and_Track_LQR(est_sat=sat, planner_settings=settings)
        t_start = orb.times[10]
        goals = GoalList({t_start: Fixed_Attitude_Goal(q_goal)})
        
        try:
            t0 = time.perf_counter()
            traj = controller.calculate_trajectory(t_start, 120, x0, orb.get_os(t_start), goals, verbose=False)
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
    
    return np.mean(times), np.std(times), np.mean(errors), np.std(errors)


def restore_backup():
    """Restore from backup."""
    subprocess.run(["cp", BACKUP_CPP, PLANNER_CPP])


if __name__ == "__main__":
    configs = [
        ("no_approx (baseline)", "solve_opts::no_approx"),
        ("likely_sympd", "solve_opts::likely_sympd"),
        ("fast", "solve_opts::fast"),
        ("no options (default)", ""),
        ("likely_sympd + no_approx", "solve_opts::likely_sympd+solve_opts::no_approx"),
        ("likely_sympd + fast", "solve_opts::likely_sympd+solve_opts::fast"),
        ("equilibrate", "solve_opts::equilibrate"),
        ("equilibrate + no_approx", "solve_opts::equilibrate+solve_opts::no_approx"),
    ]
    
    print("Testing Armadillo solve() options...")
    print()
    print(f"{'Config':<35} | {'Time':>12} | {'Error':>12}")
    print("-" * 65)
    
    results = []
    
    for name, opts in configs:
        # Restore and update
        restore_backup()
        update_solver_opts(opts)
        
        # Verify change
        with open(PLANNER_CPP, 'r') as f:
            content = f.read()
        if opts:
            if opts not in content:
                print(f"WARNING: {opts} not found in file!")
        
        # Rebuild
        if not rebuild():
            print(f"{name:<35} | BUILD FAILED")
            continue
        
        # Run in subprocess to get fresh module import
        result = subprocess.run(
            [sys.executable, "-c", """
import sys
sys.path.insert(0, '/home/pmckeen/Generalized_ADCS')
sys.path.insert(0, '/home/pmckeen/Generalized_ADCS/papers/Planner')
from benchmark_solver_opts import benchmark
r = benchmark(n_runs=5)
print(f"{r['mean_time']:.3f},{r['std_time']:.3f},{r['mean_err']:.1f},{r['std_err']:.1f}")
"""],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # Parse output - find the line with comma-separated values
            for line in result.stdout.strip().split('\n'):
                if ',' in line and not line.startswith('Time') and not line.startswith('Error'):
                    try:
                        parts = line.strip().split(',')
                        if len(parts) == 4:
                            mean_t, std_t, mean_e, std_e = map(float, parts)
                            print(f"{name:<35} | {mean_t:>5.3f}s ±{std_t:>4.2f}s | {mean_e:>5.1f}° ±{std_e:>4.1f}°")
                            results.append((name, mean_t, std_t, mean_e, std_e))
                            break
                    except:
                        pass
            else:
                print(f"{name:<35} | PARSE ERROR")
                print(f"  stdout: {result.stdout[-200:]}")
        else:
            print(f"{name:<35} | RUN FAILED")
            print(f"  stderr: {result.stderr[-200:]}")
    
    # Restore original
    restore_backup()
    rebuild()
    
    print()
    print("Summary (sorted by time):")
    print("-" * 65)
    for name, mean_t, std_t, mean_e, std_e in sorted(results, key=lambda x: x[1]):
        print(f"{name:<35} | {mean_t:>5.3f}s | {mean_e:>5.1f}°")
