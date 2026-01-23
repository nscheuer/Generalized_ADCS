#!/usr/bin/env python3
"""
LP vs QP Torque Allocation Comparison Experiment
=================================================

Paper: Generalized Attitude Control System with Magnetic Torquers and Reaction Wheels

This is the CORE contribution of the paper: demonstrating that LP-based torque
allocation preserves direction better than QP approaches.

Key Result from thesis (Table 5.1):
  - LP:  Direction error = 0.0036° (preserves direction)
  - QP:  Direction error = 33.01° (magnitude-optimal but direction-wrong)
  - LP Final pointing = 17.02° (better)
  - QP Final pointing = 25.70° (worse)

The LP approach solves:
    maximize alpha (torque scaling factor)
    subject to: alpha * tau_des = tau_mtq + tau_rw
                actuator limits

This preserves the DIRECTION of the desired torque, which is critical for
closed-loop stability.

The QP approaches solve:
    minimize ||tau_des - (tau_mtq + tau_rw)||^2
    subject to: actuator limits

This minimizes MAGNITUDE error but can have large DIRECTION errors when
the desired torque is infeasible.

Usage:
    python run_lp_vs_qp_experiment.py --quick
    python run_lp_vs_qp_experiment.py --full --output-dir ./figures
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import time

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

# --- Path Setup ---
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../../..")))

# --- ADCS Imports ---
from ADCS.CONOPS.goals import ECI_Goal
from ADCS.controller import MTQ_w_RW_LP, MTQ_w_RW_QP
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import normalize, rot_mat

# --- Plotting ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Publication style
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

COLORS = {
    'LP': '#0072B2',
    'QP': '#D55E00',
}


# =============================================================================
# CONFIGURATION
# =============================================================================

class ExpConfig:
    """Experiment configuration."""
    def __init__(self, quick: bool = False, full: bool = False):
        if quick:
            self.n_trials = 10
            self.tf = 300
            self.dt = 2
        elif full:
            self.n_trials = 100
            self.tf = 1000
            self.dt = 2
        else:
            self.n_trials = 50
            self.tf = 500
            self.dt = 2
        
        self.radius_km = 6771.0
        self.body_boresight = np.array([0, 1, 0])


# =============================================================================
# SIMULATION
# =============================================================================

def run_single_trial(
    controller,
    controller_name: str,
    sat,
    goal: ECI_Goal,
    orb: Orbit,
    config: ExpConfig,
    x0: np.ndarray,
) -> Dict[str, Any]:
    """Run single simulation trial."""
    N = int(config.tf / config.dt)
    n_act = len(sat.actuators)
    
    x = x0.copy()
    for i, rw in enumerate(sat.rw_actuators):
        if len(x) > 7:
            rw.h = x[7 + i]
    
    time_hist = np.zeros(N)
    state_hist = np.zeros((N, len(x)))
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
    
    final_idx = int(0.8 * N)
    
    return {
        'time': time_hist,
        'error_deg': error_hist,
        'final_error_deg': float(np.mean(error_hist[final_idx:])),
        'max_error_deg': float(np.max(error_hist)),
    }


def run_monte_carlo(config: ExpConfig) -> Dict[str, Dict[str, Any]]:
    """Run Monte Carlo comparison of LP vs QP."""
    results = {'LP': [], 'QP': []}
    
    for trial_id in tqdm(range(config.n_trials), desc="Trials"):
        np.random.seed(trial_id + 1000)
        
        # Create orbit
        orb = create_random_circular_orbit(
            radius_km=config.radius_km,
            dt=config.dt,
            tf=config.tf + 100,
            use_J2=True,
            fast=True,
        )
        
        # Random initial conditions (same for both)
        w0 = normalize(np.random.randn(3)) * np.random.uniform(0.001, 0.01)
        q0 = normalize(np.random.randn(4))
        h0 = np.random.uniform(-0.0001, 0.0001, 1)
        x0 = np.concatenate([w0, q0, h0])
        
        # Random goal (same for both)
        goal_vec = normalize(np.random.randn(3))
        goal = ECI_Goal(goal_vec)
        
        # LP controller
        sat_lp = create_beavercube2_cubesat(estimated=False)
        ctrl_lp = MTQ_w_RW_LP(
            est_sat=sat_lp, 
            p_gain=0.00005, d_gain=0.002, c_gain=0.001,
            h_target=np.array([0.0, 0.0, 0.0])
        )
        res_lp = run_single_trial(ctrl_lp, 'LP', sat_lp, goal, orb, config, x0.copy())
        results['LP'].append(res_lp)
        
        # QP controller
        sat_qp = create_beavercube2_cubesat(estimated=False)
        ctrl_qp = MTQ_w_RW_QP(
            est_sat=sat_qp,
            p_gain=0.00005, d_gain=0.002, c_gain=0.001,
            h_target=np.array([0.0, 0.0, 0.0])
        )
        res_qp = run_single_trial(ctrl_qp, 'QP', sat_qp, goal, orb, config, x0.copy())
        results['QP'].append(res_qp)
    
    return results


def aggregate_results(results: Dict[str, List]) -> Dict[str, Dict]:
    """Aggregate trial results into statistics."""
    aggregated = {}
    
    for name, trials in results.items():
        errors = [t['final_error_deg'] for t in trials]
        errors = np.array(errors)
        
        aggregated[name] = {
            'n_trials': len(trials),
            'all_errors_deg': errors.tolist(),
            'all_time_series': [t['error_deg'] for t in trials],
            'mean_error_deg': float(np.mean(errors)),
            'std_error_deg': float(np.std(errors)),
            'median_error_deg': float(np.median(errors)),
            'pct_within_1deg': float(100 * np.sum(errors < 1) / len(errors)),
            'pct_within_5deg': float(100 * np.sum(errors < 5) / len(errors)),
            'pct_within_10deg': float(100 * np.sum(errors < 10) / len(errors)),
        }
    
    return aggregated


# =============================================================================
# PLOTTING
# =============================================================================

def plot_error_trajectories(results: Dict[str, Dict], output_dir: Path, config: ExpConfig):
    """Plot LP vs QP error trajectories."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    times = np.arange(0, config.tf, config.dt)
    
    for name in ['LP', 'QP']:
        color = COLORS[name]
        
        # Plot all with low alpha
        for ts in results[name]['all_time_series']:
            ax.plot(times[:len(ts)], ts, color=color, alpha=0.15, linewidth=0.5)
        
        # Plot mean
        mean_ts = np.mean(results[name]['all_time_series'], axis=0)
        ax.plot(times[:len(mean_ts)], mean_ts, color=color, linewidth=2.5, label=name)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pointing Error (deg)')
    ax.set_yscale('log')
    ax.set_ylim(0.1, 200)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    
    fig.savefig(output_dir / 'fig_lp_vs_qp_trajectories.png')
    fig.savefig(output_dir / 'fig_lp_vs_qp_trajectories.pdf')
    plt.close(fig)
    print(f"  Saved: fig_lp_vs_qp_trajectories")


def plot_final_error_comparison(results: Dict[str, Dict], output_dir: Path):
    """Plot box comparison of final errors."""
    fig, ax = plt.subplots(figsize=(4, 4))
    
    data = [results['LP']['all_errors_deg'], results['QP']['all_errors_deg']]
    positions = [0, 1]
    colors = [COLORS['LP'], COLORS['QP']]
    
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['LP\n(Direction-Preserving)', 'QP\n(Magnitude-Optimal)'])
    ax.set_ylabel('Final Pointing Error (deg)')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(5.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    
    # Add annotations
    for i, (name, pos) in enumerate(zip(['LP', 'QP'], positions)):
        mean = results[name]['mean_error_deg']
        ax.annotate(f'μ={mean:.1f}°', xy=(pos, mean), xytext=(pos + 0.3, mean),
                    fontsize=8, ha='left')
    
    fig.savefig(output_dir / 'fig_lp_vs_qp_boxplot.png')
    fig.savefig(output_dir / 'fig_lp_vs_qp_boxplot.pdf')
    plt.close(fig)
    print(f"  Saved: fig_lp_vs_qp_boxplot")


def plot_cdf_comparison(results: Dict[str, Dict], output_dir: Path):
    """Plot CDF comparison."""
    fig, ax = plt.subplots(figsize=(5, 4))
    
    for name in ['LP', 'QP']:
        errors = np.sort(results[name]['all_errors_deg'])
        cdf = np.arange(1, len(errors) + 1) / len(errors) * 100
        ax.plot(errors, cdf, label=name, color=COLORS[name], linewidth=1.5)
    
    ax.set_xscale('log')
    ax.set_xlabel('Final Pointing Error (deg)')
    ax.set_ylabel('Cumulative %')
    ax.axvline(1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(5.0, color='k', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axhline(90, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_xlim(0.1, 200)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    fig.savefig(output_dir / 'fig_lp_vs_qp_cdf.png')
    fig.savefig(output_dir / 'fig_lp_vs_qp_cdf.pdf')
    plt.close(fig)
    print(f"  Saved: fig_lp_vs_qp_cdf")


def generate_latex_table(results: Dict[str, Dict], output_dir: Path):
    """Generate LaTeX comparison table."""
    latex = r"""\begin{table}[htbp]
    \centering
    \caption{LP vs QP Torque Allocation: Monte Carlo Pointing Results}
    \label{tab:lp_vs_qp}
    \begin{tabular}{lcccccc}
        \toprule
        Method & Mean & Std & Median & $<1^\circ$ & $<5^\circ$ & $<10^\circ$ \\
        \midrule
"""
    
    for name in ['LP', 'QP']:
        res = results[name]
        latex += f"        {name} & {res['mean_error_deg']:.2f}$^\\circ$ & "
        latex += f"{res['std_error_deg']:.2f}$^\\circ$ & "
        latex += f"{res['median_error_deg']:.2f}$^\\circ$ & "
        latex += f"{res['pct_within_1deg']:.0f}\\% & "
        latex += f"{res['pct_within_5deg']:.0f}\\% & "
        latex += f"{res['pct_within_10deg']:.0f}\\% \\\\\n"
    
    latex += r"""        \bottomrule
    \end{tabular}
\end{table}
"""
    
    with open(output_dir / 'table_lp_vs_qp.tex', 'w') as f:
        f.write(latex)
    print(f"  Saved: table_lp_vs_qp.tex")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LP vs QP Comparison Experiment")
    parser.add_argument('--quick', action='store_true', help='Quick run (10 trials, 300s)')
    parser.add_argument('--full', action='store_true', help='Full run (100 trials, 1000s)')
    parser.add_argument('--output-dir', type=str, default='./output_lp_qp',
                        help='Output directory')
    args = parser.parse_args()
    
    config = ExpConfig(quick=args.quick, full=args.full)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("  LP vs QP Torque Allocation Comparison")
    print("="*60)
    print(f"  Trials: {config.n_trials}")
    print(f"  Duration: {config.tf}s")
    print(f"  Output: {output_dir}")
    print()
    print("  This experiment demonstrates the CORE contribution:")
    print("  LP preserves torque DIRECTION (critical for stability)")
    print("  QP minimizes MAGNITUDE error (can have large direction error)")
    print("="*60)
    
    # Run Monte Carlo
    print("\n  Running simulations...")
    raw_results = run_monte_carlo(config)
    results = aggregate_results(raw_results)
    
    # Generate figures
    print("\n  Generating figures...")
    plot_error_trajectories(results, output_dir, config)
    plot_final_error_comparison(results, output_dir)
    plot_cdf_comparison(results, output_dir)
    generate_latex_table(results, output_dir)
    
    # Save data
    save_results = {k: {kk: vv for kk, vv in v.items() if kk != 'all_time_series'} 
                    for k, v in results.items()}
    with open(output_dir / 'experiment_data.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("  Results Summary")
    print("="*60)
    print(f"  {'Method':<12} {'Mean':>10} {'Median':>10} {'<1°':>8} {'<5°':>8}")
    print("  " + "-"*48)
    for name in ['LP', 'QP']:
        res = results[name]
        print(f"  {name:<12} {res['mean_error_deg']:>9.2f}° "
              f"{res['median_error_deg']:>9.2f}° "
              f"{res['pct_within_1deg']:>7.0f}% "
              f"{res['pct_within_5deg']:>7.0f}%")
    
    # Compute improvement
    if results['QP']['mean_error_deg'] > 0:
        improvement = (results['QP']['mean_error_deg'] - results['LP']['mean_error_deg']) / results['QP']['mean_error_deg'] * 100
        print(f"\n  LP improvement over QP: {improvement:.1f}% lower mean error")
    
    print(f"\n  All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
