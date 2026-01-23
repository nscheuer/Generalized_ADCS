#!/usr/bin/env python3
"""
3+1 Paper Experiments
=====================

Paper: 3-Magnetorquer, 1-Reaction-Wheel Architecture Demonstration

This script runs all experiments from the 3+1 paper TODO list using the
real simulation infrastructure from the codebase.

Experiments from paper main2.tex:
  A1. PD Control Baseline Comparison (3+0, 3+1, 3+3)
  A2. Planner-Enhanced Comparison
  A3. Goal Formulation Impact
  B1. Continuous vs Scheduled Desaturation
  B2. Long-Duration Stability
  C1. 1U Volume-Constrained Imaging
  C2. Eclipse Power Constraint
  C3. Graceful Degradation (Wheel Failure)

Usage:
    python run_3p1_paper_experiments.py --exp A1 --quick
    python run_3p1_paper_experiments.py --exp A1 A2 --full
    python run_3p1_paper_experiments.py --all --output-dir ./figures
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

# --- Path Setup ---
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../../..")))

# --- ADCS Imports ---
from ADCS.CONOPS.goals import ECI_Goal
from ADCS.controller import MTQ_w_RW_LP
from ADCS.controller.mtq_lovera import MTQ_Lovera
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_factory.satellites.create_cubesats import (
    create_beavercube1_cubesat,
    create_beavercube2_cubesat,
    create_3_3_beavercube2_cubesat,
)
from ADCS.helpers.math_helpers import normalize, rot_mat
from ADCS.helpers.save_and_load.save_and_load import save_data

# --- Plotting ---
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt

# Set up publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.2,
    'legend.frameon': False,
})

# Colors (colorblind-friendly)
COLORS = {
    '3+0': '#0072B2',
    '3+1': '#D55E00', 
    '3+3': '#009E73',
}


# =============================================================================
# CONFIGURATION
# =============================================================================

class ExpConfig:
    """Experiment configuration."""
    def __init__(self, quick: bool = False, full: bool = False):
        if quick:
            self.n_trials = 10
            self.tf = 200  # seconds
            self.dt = 2
        elif full:
            self.n_trials = 100
            self.tf = 1000
            self.dt = 2
        else:  # default
            self.n_trials = 50
            self.tf = 500
            self.dt = 2
        
        self.radius_km = 6771.0  # ~400 km altitude
        self.body_boresight = np.array([0, 1, 0])  # y-axis camera


# =============================================================================
# SIMULATION INFRASTRUCTURE
# =============================================================================

def run_single_trial(
    sat: Satellite,
    controller,
    goal: ECI_Goal,
    orb: Orbit,
    config: ExpConfig,
    x0: np.ndarray,
) -> Dict[str, Any]:
    """
    Run a single simulation trial.
    
    Returns dict with time histories and final metrics.
    """
    N = int(config.tf / config.dt)
    n_act = len(sat.actuators)
    n_rw = len([a for a in sat.actuators if isinstance(a, RW)])
    
    # Initialize
    x = x0.copy()
    for i, rw in enumerate(sat.rw_actuators):
        if n_rw > 0 and len(x) > 7:
            rw.h = x[7 + i]
    
    # Arrays
    time_hist = np.zeros(N)
    state_hist = np.zeros((N, len(x)))
    u_hist = np.zeros((N, n_act))
    error_hist = np.zeros(N)
    
    t = 0
    for i in range(N):
        J2000 = 0.22 + t * TimeConstants.sec2cent
        os = orb.get_os(J2000)
        sens = sat.sensor_readings(x=x, os=os)
        
        u = controller.find_u(x_hat=x, sens=sens, est_sat=sat, os_hat=os, goal=goal)
        
        # Compute pointing error
        q = x[3:7]
        R = rot_mat(q)
        boresight_eci = R @ config.body_boresight
        goal_vec, _ = goal.to_ref(os)
        
        dot = np.clip(np.dot(boresight_eci, goal_vec / np.linalg.norm(goal_vec)), -1, 1)
        error_deg = np.rad2deg(np.arccos(dot))
        
        time_hist[i] = t
        state_hist[i] = x
        u_hist[i] = u
        error_hist[i] = error_deg
        
        # Propagate
        t_next = t + config.dt
        os_next = orb.get_os(0.22 + t_next * TimeConstants.sec2cent)
        
        out = solve_ivp(
            fun=sat.dynamics_for_solver,
            t_span=(0, config.dt),
            y0=x,
            method='RK45',
            args=(u, os, os_next),
            rtol=1e-6,
            atol=1e-6,
        )
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])
        t = t_next
    
    # Compute final metrics (last 20% of simulation)
    final_idx = int(0.8 * N)
    final_error = np.mean(error_hist[final_idx:])
    
    return {
        'time': time_hist,
        'state': state_hist,
        'u': u_hist,
        'error_deg': error_hist,
        'final_error_deg': final_error,
        'max_error_deg': np.max(error_hist),
        'converged': final_error < 10.0,
    }


def run_monte_carlo(
    sat_factory,
    controller_factory,
    config: ExpConfig,
    config_name: str,
) -> Dict[str, Any]:
    """
    Run Monte Carlo campaign for one configuration.
    """
    all_final_errors = []
    all_time_series = []
    
    for trial_id in tqdm(range(config.n_trials), desc=f"  {config_name}"):
        np.random.seed(trial_id + 1000)
        
        # Create orbit (random position)
        orb = create_random_circular_orbit(
            radius_km=config.radius_km,
            dt=config.dt,
            tf=config.tf + 100,
            use_J2=True,
            fast=True,
        )
        
        # Create satellite and controller
        sat = sat_factory()
        controller = controller_factory(sat)
        
        # Random initial conditions
        w0 = normalize(np.random.randn(3)) * np.random.uniform(0.001, 0.01)
        q0 = normalize(np.random.randn(4))
        
        n_rw = len(sat.rw_actuators)
        h0 = np.random.uniform(-0.0001, 0.0001, n_rw) if n_rw > 0 else np.array([])
        x0 = np.concatenate([w0, q0, h0])
        
        # Random goal
        goal_vec = normalize(np.random.randn(3))
        goal = ECI_Goal(goal_vec)
        
        # Run simulation
        result = run_single_trial(sat, controller, goal, orb, config, x0)
        
        all_final_errors.append(result['final_error_deg'])
        all_time_series.append(result['error_deg'])
    
    errors = np.array(all_final_errors)
    
    return {
        'config_name': config_name,
        'n_trials': config.n_trials,
        'all_errors_deg': errors.tolist(),
        'all_time_series': all_time_series,
        'mean_error_deg': float(np.mean(errors)),
        'std_error_deg': float(np.std(errors)),
        'median_error_deg': float(np.median(errors)),
        'pct_within_1deg': float(100 * np.sum(errors < 1) / len(errors)),
        'pct_within_5deg': float(100 * np.sum(errors < 5) / len(errors)),
        'pct_within_10deg': float(100 * np.sum(errors < 10) / len(errors)),
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_error_trajectories(results: Dict[str, Dict], output_dir: Path, config: ExpConfig):
    """Plot pointing error trajectories for all configs."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    times = np.arange(0, config.tf, config.dt)
    
    for name, res in results.items():
        color = COLORS.get(name.split()[0], '#333333')
        
        # Plot all trajectories with low alpha
        for ts in res['all_time_series']:
            ax.plot(times[:len(ts)], ts, color=color, alpha=0.1, linewidth=0.5)
        
        # Plot mean with full alpha
        mean_ts = np.mean(res['all_time_series'], axis=0)
        ax.plot(times[:len(mean_ts)], mean_ts, color=color, linewidth=2, label=name)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pointing Error (deg)')
    ax.set_yscale('log')
    ax.set_ylim(0.01, 200)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    fig.savefig(output_dir / 'fig_error_trajectories.png')
    fig.savefig(output_dir / 'fig_error_trajectories.pdf')
    plt.close(fig)
    print(f"  Saved: fig_error_trajectories")


def plot_error_histogram(results: Dict[str, Dict], output_dir: Path):
    """Plot final error histograms."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    bins = np.logspace(-2, 2, 40)
    
    for name, res in results.items():
        color = COLORS.get(name.split()[0], '#333333')
        errors = res['all_errors_deg']
        ax.hist(errors, bins=bins, alpha=0.6, label=name, color=color, edgecolor='white')
    
    ax.set_xscale('log')
    ax.set_xlabel('Final Pointing Error (deg)')
    ax.set_ylabel('Count')
    ax.axvline(1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(5.0, color='k', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.legend()
    
    fig.savefig(output_dir / 'fig_error_histogram.png')
    fig.savefig(output_dir / 'fig_error_histogram.pdf')
    plt.close(fig)
    print(f"  Saved: fig_error_histogram")


def plot_success_rates(results: Dict[str, Dict], output_dir: Path):
    """Plot success rate bar chart."""
    fig, ax = plt.subplots(figsize=(5, 4))
    
    configs = list(results.keys())
    x = np.arange(len(configs))
    width = 0.25
    
    pct_1 = [results[c]['pct_within_1deg'] for c in configs]
    pct_5 = [results[c]['pct_within_5deg'] for c in configs]
    pct_10 = [results[c]['pct_within_10deg'] for c in configs]
    
    ax.bar(x - width, pct_1, width, label=r'$<1^\circ$', color='#2ecc71')
    ax.bar(x, pct_5, width, label=r'$<5^\circ$', color='#f39c12')
    ax.bar(x + width, pct_10, width, label=r'$<10^\circ$', color='#e74c3c')
    
    ax.set_ylabel('Success Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha='right')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right')
    
    fig.savefig(output_dir / 'fig_success_rates.png')
    fig.savefig(output_dir / 'fig_success_rates.pdf')
    plt.close(fig)
    print(f"  Saved: fig_success_rates")


def plot_cdf(results: Dict[str, Dict], output_dir: Path):
    """Plot CDF of pointing errors."""
    fig, ax = plt.subplots(figsize=(5, 4))
    
    for name, res in results.items():
        color = COLORS.get(name.split()[0], '#333333')
        errors = np.sort(res['all_errors_deg'])
        cdf = np.arange(1, len(errors) + 1) / len(errors) * 100
        ax.plot(errors, cdf, label=name, color=color, linewidth=1.5)
    
    ax.set_xscale('log')
    ax.set_xlabel('Final Pointing Error (deg)')
    ax.set_ylabel('Cumulative %')
    ax.axvline(1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axhline(90, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_xlim(0.01, 100)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    fig.savefig(output_dir / 'fig_cdf.png')
    fig.savefig(output_dir / 'fig_cdf.pdf')
    plt.close(fig)
    print(f"  Saved: fig_cdf")


def generate_latex_table(results: Dict[str, Dict], output_dir: Path, table_num: int = 1):
    """Generate LaTeX table."""
    latex = r"""\begin{table}[htbp]
    \centering
    \caption{Monte Carlo Results: PD Control Comparison}
    \label{tab:mc_pd}
    \begin{tabular}{lccccc}
        \toprule
        Configuration & Mean & Std & $<1^\circ$ & $<5^\circ$ & $<10^\circ$ \\
        \midrule
"""
    
    for name, res in results.items():
        latex += f"        {name} & {res['mean_error_deg']:.2f}$^\\circ$ & "
        latex += f"{res['std_error_deg']:.2f}$^\\circ$ & "
        latex += f"{res['pct_within_1deg']:.0f}\\% & "
        latex += f"{res['pct_within_5deg']:.0f}\\% & "
        latex += f"{res['pct_within_10deg']:.0f}\\% \\\\\n"
    
    latex += r"""        \bottomrule
    \end{tabular}
\end{table}
"""
    
    with open(output_dir / f'table{table_num}_results.tex', 'w') as f:
        f.write(latex)
    print(f"  Saved: table{table_num}_results.tex")


# =============================================================================
# EXPERIMENT A1: PD BASELINE COMPARISON
# =============================================================================

def run_experiment_A1(config: ExpConfig, output_dir: Path):
    """
    Experiment A1: PD Control Baseline Comparison.
    
    Compare 3+0, 3+1, 3+3 with identical PD gains.
    """
    print("\n" + "="*60)
    print("  Experiment A1: PD Control Baseline Comparison")
    print("="*60)
    
    # Define configurations
    # Note: MTQ_Lovera uses (p_gain, d_gain, eps) signature
    # MTQ_w_RW_LP uses (p_gain, d_gain, c_gain, h_target) signature
    # h_target is always 3D (total momentum target in body frame)
    configs = {
        '3+0': (
            lambda: create_beavercube1_cubesat(estimated=False),
            lambda sat: MTQ_Lovera(est_sat=sat, p_gain=0.001, d_gain=0.005, eps=1.0),
        ),
        '3+1': (
            lambda: create_beavercube2_cubesat(estimated=False),
            lambda sat: MTQ_w_RW_LP(est_sat=sat, p_gain=0.00005, d_gain=0.002, c_gain=0.001, h_target=np.array([0.0, 0.0, 0.0])),
        ),
        '3+3': (
            lambda: create_3_3_beavercube2_cubesat(estimated=False),
            lambda sat: MTQ_w_RW_LP(est_sat=sat, p_gain=0.00005, d_gain=0.002, c_gain=0.001, h_target=np.array([0.0, 0.0, 0.0])),
        ),
    }
    
    results = {}
    for name, (sat_factory, ctrl_factory) in configs.items():
        res = run_monte_carlo(sat_factory, ctrl_factory, config, name)
        results[name] = res
    
    # Generate outputs
    print("\n  Generating figures...")
    plot_error_trajectories(results, output_dir, config)
    plot_error_histogram(results, output_dir)
    plot_success_rates(results, output_dir)
    plot_cdf(results, output_dir)
    generate_latex_table(results, output_dir, table_num=1)
    
    # Save raw data
    with open(output_dir / 'experiment_A1_data.json', 'w') as f:
        # Remove time series for smaller file
        save_results = {k: {kk: vv for kk, vv in v.items() if kk != 'all_time_series'} 
                        for k, v in results.items()}
        json.dump(save_results, f, indent=2)
    
    # Print summary
    print("\n  Results Summary:")
    print(f"  {'Config':<10} {'Mean':>10} {'<1°':>8} {'<5°':>8} {'<10°':>8}")
    print("  " + "-"*46)
    for name, res in results.items():
        print(f"  {name:<10} {res['mean_error_deg']:>9.2f}° "
              f"{res['pct_within_1deg']:>7.0f}% "
              f"{res['pct_within_5deg']:>7.0f}% "
              f"{res['pct_within_10deg']:>7.0f}%")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="3+1 Paper Experiments")
    parser.add_argument('--exp', nargs='+', default=['A1'], 
                        choices=['A1', 'A2', 'A3', 'B1', 'B2', 'C1', 'C2', 'C3', 'all'],
                        help='Experiments to run')
    parser.add_argument('--quick', action='store_true', help='Quick run (10 trials, 200s)')
    parser.add_argument('--full', action='store_true', help='Full run (100 trials, 1000s)')
    parser.add_argument('--output-dir', type=str, default='./output_3p1',
                        help='Output directory')
    args = parser.parse_args()
    
    config = ExpConfig(quick=args.quick, full=args.full)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("  3+1 Paper Experiments")
    print("="*60)
    print(f"  Trials: {config.n_trials}")
    print(f"  Duration: {config.tf}s")
    print(f"  Output: {output_dir}")
    
    experiments = args.exp
    if 'all' in experiments:
        experiments = ['A1', 'A2', 'A3', 'B1', 'B2', 'C1', 'C2', 'C3']
    
    for exp in experiments:
        if exp == 'A1':
            run_experiment_A1(config, output_dir)
        else:
            print(f"\n  Experiment {exp} not yet implemented")
    
    print(f"\n  All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
