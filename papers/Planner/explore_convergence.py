#!/usr/bin/env python3
"""
Fast exploration of planner convergence with different settings.
Saves plots to files for analysis.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/pmckeen/Generalized_ADCS')

from typing import List, Dict, Callable
from ADCS.controller.helpers.python_alilqr_v2 import PythonALILQRv2, IterationData
from ADCS.controller.plan_and_track_python_alilqr import Plan_and_Track_PythonALILQR
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.helpers.math_helpers import normalize, quat_mult
from ADCS.controller.helpers import PlannerSettings
from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.CONOPS.goallist import GoalList


def setup_scenario(duration=120, seed=1001):
    """Create a test scenario."""
    np.random.seed(seed)
    
    sat = create_beavercube2_cubesat(estimated=False)
    orb = create_random_circular_orbit(radius_km=7000.0, dt=1, tf=duration+100, use_J2=True, fast=True)
    
    q0 = normalize(np.random.randn(4))
    w0 = np.random.randn(3) * 0.5 * np.pi / 180
    h0 = np.array([np.random.uniform(-0.001, 0.001)])
    
    half_angle = 45 * np.pi / 180
    axis = normalize(np.random.randn(3))
    q_rot = np.concatenate([[np.cos(half_angle)], np.sin(half_angle) * axis])
    q_goal = normalize(quat_mult(q0, q_rot))
    
    x0 = np.concatenate([w0, q0, h0])
    sat.rw_actuators[0].h = h0[0]
    
    t_start = orb.times[10]
    os0 = orb.get_os(t_start)
    goals = GoalList({t_start: Fixed_Attitude_Goal(q_goal)})
    
    return sat, orb, x0, q_goal, t_start, os0, goals


class IterationCollector:
    """Collects iteration data during optimization."""
    def __init__(self, max_iters=None):
        self.data: List[IterationData] = []
        self.max_iters = max_iters
        
    def __call__(self, d: IterationData):
        self.data.append(d)
        if self.max_iters and len(self.data) >= self.max_iters:
            raise StopIteration()


def run_with_settings(sat, orb, x0, q_goal, t_start, os0, goals, 
                     duration=120, max_iters=50, modify_settings: Callable = None,
                     label=""):
    """Run optimization with custom settings and collect data."""
    
    # Create settings
    settings = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=10, dt_tvlqr=1)
    settings.verbosity = False
    
    # Modify settings if requested
    if modify_settings:
        modify_settings(settings)
    
    # Create controller with Python planner
    controller = Plan_and_Track_PythonALILQR(est_sat=sat, planner_settings=settings)
    
    # Set up iteration collector
    collector = IterationCollector(max_iters=max_iters)
    controller.py_alilqr.set_callback(collector)
    
    try:
        traj = controller.calculate_trajectory(
            t_start=t_start, 
            duration=duration, 
            x_0=x0, 
            os_0=os0, 
            goals=goals, 
            verbose=False
        )
    except StopIteration:
        pass  # Expected if max_iters reached
    except Exception as e:
        print(f"  Error: {e}")
    
    return collector.data


def compute_angle_error(X, q_goal):
    """Compute angle error at each timestep. X is (n_states, N+1)."""
    errors = []
    n_times = X.shape[1]
    for i in range(n_times):
        q = X[3:7, i]
        q = q / np.linalg.norm(q)
        dot = min(abs(np.dot(q, q_goal)), 1.0)
        errors.append(np.degrees(2 * np.arccos(dot)))
    return np.array(errors)


def plot_convergence(all_data: Dict[str, List[IterationData]], q_goal, filename):
    """Plot convergence comparison and save to file."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))
    
    for (name, data), color in zip(all_data.items(), colors):
        if not data:
            continue
            
        iters = [d.total_iter for d in data]
        costs = [d.LA for d in data]
        costs_nc = [d.LA_nc for d in data]
        cmaxs = [d.cmax for d in data]
        grads = [d.grad for d in data]
        rhos = [d.rho for d in data]
        final_errs = [compute_angle_error(d.Xset, q_goal)[-1] for d in data]
        
        axes[0, 0].semilogy(iters, costs, '-', label=name, color=color, alpha=0.8)
        axes[0, 1].semilogy(iters, [max(c, 1e-10) for c in cmaxs], '-', color=color, alpha=0.8)
        axes[0, 2].plot(iters, final_errs, '-', color=color, alpha=0.8)
        axes[1, 0].semilogy(iters, [max(g, 1e-10) for g in grads], '-', color=color, alpha=0.8)
        axes[1, 1].semilogy(iters, [max(r, 1e-10) for r in rhos], '-', color=color, alpha=0.8)
        axes[1, 2].semilogy(iters, costs_nc, '-', color=color, alpha=0.8)
    
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Aug. Lagrangian Cost')
    axes[0, 0].set_title('Total Cost (LA)')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Max Constraint Violation')
    axes[0, 1].set_title('Constraint Violation')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_ylabel('Final Angle Error (deg)')
    axes[0, 2].set_title('Final Pointing Error')
    axes[0, 2].axhline(y=10, color='g', linestyle='--', alpha=0.5, label='10° target')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].set_title('Gradient')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Regularization (rho)')
    axes[1, 1].set_title('Regularization')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].set_xlabel('Iteration')
    axes[1, 2].set_ylabel('Cost (no constraints)')
    axes[1, 2].set_title('Pure Cost (LA_nc)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Saved: {filename}")


def plot_trajectory_evolution(data: List[IterationData], q_goal, filename, name=""):
    """Plot how trajectory evolves over iterations."""
    if not data:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Sample iterations to show
    n_samples = min(8, len(data))
    indices = np.linspace(0, len(data)-1, n_samples, dtype=int)
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_samples))
    
    for idx, color in zip(indices, colors):
        d = data[idx]
        times = d.times - d.times[0]
        X = d.Xset  # Shape: (n_states, N+1)
        U = d.Uset  # Shape: (n_controls, N) or (n_controls, N+1)
        
        N_x = X.shape[1]
        N_u = U.shape[1]
        
        # Angular velocity (X is n_states x N+1)
        axes[0, 0].plot(times[:N_x], np.degrees(X[0, :]), '-', color=color, alpha=0.7)
        
        # Angle error
        errors = compute_angle_error(X, q_goal)
        axes[0, 1].plot(times[:N_x], errors, '-', color=color, alpha=0.7, 
                       label=f'iter {d.total_iter}')
        
        # RW torque (U is n_controls x N)
        axes[1, 0].plot(times[:N_u], U[3, :]*1000, '-', color=color, alpha=0.7)
        
        # MTQ dipole magnitude
        mtq_mag = np.linalg.norm(U[:3, :], axis=0)
        axes[1, 1].plot(times[:N_u], mtq_mag, '-', color=color, alpha=0.7)
    
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('ω_x (deg/s)')
    axes[0, 0].set_title('Angular Velocity')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Angle Error (deg)')
    axes[0, 1].set_title(f'Pointing Error - {name}')
    axes[0, 1].legend(fontsize=7, loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('RW Torque (mNm)')
    axes[1, 0].set_title('Reaction Wheel')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('MTQ Dipole (Am²)')
    axes[1, 1].set_title('Magnetorquer')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Saved: {filename}")


def print_summary(all_data: Dict[str, List[IterationData]], q_goal):
    """Print summary statistics."""
    print("\n=== SUMMARY ===")
    print(f"{'Name':<35} {'Iters':>6} {'Final Err':>10} {'Cmax':>10}")
    print("-" * 65)
    
    for name, data in all_data.items():
        if not data:
            print(f"{name:<35} {'FAILED':>6}")
            continue
        
        final = data[-1]
        final_err = compute_angle_error(final.Xset, q_goal)[-1]
        print(f"{name:<35} {final.total_iter:>6} {final_err:>9.1f}° {final.cmax:>10.2e}")


if __name__ == "__main__":
    print("Setting up scenario...")
    sat, orb, x0, q_goal, t_start, os0, goals = setup_scenario(duration=120)
    
    print("Running baseline...")
    data = run_with_settings(sat, orb, x0, q_goal, t_start, os0, goals, 
                             duration=120, max_iters=30, label="baseline")
    
    print(f"Collected {len(data)} iterations")
    if data:
        final_err = compute_angle_error(data[-1].Xset, q_goal)[-1]
        print(f"Final error: {final_err:.1f}°")
        
        plot_trajectory_evolution(data, q_goal, "/tmp/traj_evolution.png", "Baseline")
