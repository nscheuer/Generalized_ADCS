#!/usr/bin/env python3
"""
Plot Thesis Figures from Saved Data
====================================

Reload saved simulation data and regenerate plots without re-running simulations.

Usage:
    python plot_thesis_figures.py --all                    # Plot all available data
    python plot_thesis_figures.py --test lovera            # Plot specific test
    python plot_thesis_figures.py --data thesis_figures/   # Specify data directory
"""

import sys
import os
import argparse
import pickle
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Publication-quality settings
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['legend.fontsize'] = 10
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'


def load_data(filepath: Path) -> dict:
    """Load pickled simulation data."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def plot_controller_comparison(data: dict, controller_name: str, output_dir: Path):
    """
    Plot comparison of disturbance cases for a controller.
    
    Thesis Figures:
    - angular_error_{controller}.png (comparison of all cases)
    - log_angular_error_{controller}.png
    - ctrl_{controller}.png
    """
    time = data['time']
    cases = data.get('cases', {})
    config = data.get('config', {})
    
    time_hours = time / 3600
    
    # If old format (single case), convert to new format
    if not cases and 'error' in data:
        cases = {'Single': {'error': data['error'], 'state': data['state'], 
                           'control': data['control'], 'color': 'blue'}}
    
    # Figure 1: Angular Error Comparison (log scale) - MAIN THESIS FIGURE
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, case_data in cases.items():
        color = case_data.get('color', 'blue')
        ax.semilogy(time_hours, case_data['error'], color=color, 
                    label=name, linewidth=1.5)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Angular Error (degrees)')
    ax.set_title(f'{controller_name.title()} Controller: Disturbance Comparison')
    ax.legend(loc='best')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim([0.1, 200])
    plt.tight_layout()
    plt.savefig(output_dir / f'log_angular_error_{controller_name}.png')
    plt.close()
    
    # Figure 2: Angular Error (linear scale)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, case_data in cases.items():
        color = case_data.get('color', 'blue')
        ax.plot(time_hours, case_data['error'], color=color,
                label=name, linewidth=1.5)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Angular Error (degrees)')
    ax.set_title(f'{controller_name.title()} Controller: Disturbance Comparison')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'angular_error_{controller_name}.png')
    plt.close()
    
    # Figure 3: Control Effort Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, case_data in cases.items():
        color = case_data.get('color', 'blue')
        ctrl_mag = np.linalg.norm(case_data['control'], axis=1)
        ax.plot(time_hours, ctrl_mag, color=color, label=name, linewidth=1, alpha=0.8)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('|m| (Am²)')
    ax.set_title(f'{controller_name.title()} Controller: Control Effort')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'ctrl_{controller_name}.png')
    plt.close()
    
    # Figure 4: Combined 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Log error
    ax = axes[0, 0]
    for name, case_data in cases.items():
        ax.semilogy(time_hours, case_data['error'], color=case_data.get('color', 'blue'),
                    label=name, linewidth=1.5)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Angular Error (deg)')
    ax.set_title('Angular Error (Log Scale)')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    
    # Linear error
    ax = axes[0, 1]
    for name, case_data in cases.items():
        ax.plot(time_hours, case_data['error'], color=case_data.get('color', 'blue'),
                label=name, linewidth=1.5)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Angular Error (deg)')
    ax.set_title('Angular Error (Linear Scale)')
    ax.grid(True, alpha=0.3)
    
    # Angular velocity magnitude
    ax = axes[1, 0]
    for name, case_data in cases.items():
        omega = case_data['state'][:, 0:3] * 180 / np.pi
        ax.plot(time_hours, np.linalg.norm(omega, axis=1), 
                color=case_data.get('color', 'blue'), label=name, linewidth=1, alpha=0.8)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('|ω| (deg/s)')
    ax.set_title('Angular Velocity Magnitude')
    ax.grid(True, alpha=0.3)
    
    # Control magnitude
    ax = axes[1, 1]
    for name, case_data in cases.items():
        ctrl_mag = np.linalg.norm(case_data['control'], axis=1)
        ax.plot(time_hours, ctrl_mag, color=case_data.get('color', 'blue'),
                label=name, linewidth=1, alpha=0.8)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('|m| (Am²)')
    ax.set_title('Control Effort Magnitude')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{controller_name.title()} Controller: Disturbance Comparison', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'{controller_name}_combined.png')
    plt.close()
    
    print(f"  Generated: log_angular_error_{controller_name}.png, angular_error_{controller_name}.png, "
          f"ctrl_{controller_name}.png, {controller_name}_combined.png")


def plot_lovera(data: dict, output_dir: Path, show_config: bool = True):
    """Plot Lovera MTQ-PD controller results."""
    plot_controller_comparison(data, 'lovera', output_dir)


def plot_wisniewski(data: dict, output_dir: Path, show_config: bool = True):
    """Plot Wisniewski Sliding Mode controller results."""
    plot_controller_comparison(data, 'wisniewski', output_dir)


def plot_spinning(data: dict, output_dir: Path):
    """Plot spinning solution results."""
    time = data['time']
    state = data['state']
    control = data['control']
    error = data['error']
    
    # Figure 1: Pointing Error
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time, error, 'b-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pointing Error (degrees)')
    ax.set_title('Spinning Solution: Pointing Error')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'spinning_ang.png')
    plt.close()
    
    # Figure 2: Angular Velocity
    fig, ax = plt.subplots(figsize=(8, 5))
    omega = state[:, 0:3] * 180 / np.pi
    ax.plot(time, omega[:, 0], 'r-', label='ωx', linewidth=1)
    ax.plot(time, omega[:, 1], 'g-', label='ωy', linewidth=1)
    ax.plot(time, omega[:, 2], 'b-', label='ωz', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Velocity (deg/s)')
    ax.set_title('Spinning Solution: Angular Velocity')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'spinning_av.png')
    plt.close()
    
    # Figure 3: Control Commands
    fig, ax = plt.subplots(figsize=(8, 5))
    n_mtq = min(3, control.shape[1])
    ax.plot(time, control[:, 0], 'r-', label='mx', linewidth=1)
    ax.plot(time, control[:, 1], 'g-', label='my', linewidth=1)
    ax.plot(time, control[:, 2], 'b-', label='mz', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MTQ Command (Am²)')
    ax.set_title('Spinning Solution: MTQ Commands')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'spinning_cmd.png')
    plt.close()
    
    # Figure 4: RW momentum if present
    if state.shape[1] > 7:
        fig, ax = plt.subplots(figsize=(8, 5))
        h_rw = state[:, 7:] * 1000  # Convert to mNms
        for i in range(h_rw.shape[1]):
            ax.plot(time, h_rw[:, i], label=f'h_rw{i+1}', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('RW Momentum (mNms)')
        ax.set_title('Spinning Solution: RW Stored Momentum')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'spinning_cmd_rw.png')
        plt.close()
    
    print(f"  Generated: spinning_ang.png, spinning_av.png, spinning_cmd.png")


def main():
    parser = argparse.ArgumentParser(description="Plot Thesis Figures from Saved Data")
    parser.add_argument('--all', action='store_true', help="Plot all available data")
    parser.add_argument('--test', type=str, choices=['lovera', 'wisniewski', 'spinning'],
                        help="Specific test to plot")
    parser.add_argument('--data', type=str, default="thesis_figures", help="Data directory")
    parser.add_argument('--no-config', action='store_true', help="Don't show config info on plots")
    
    args = parser.parse_args()
    
    if not (args.all or args.test):
        parser.print_help()
        return
    
    data_dir = Path(args.data)
    ch6_dir = data_dir / "chapter6"
    ch7_dir = data_dir / "chapter7"
    
    show_config = not args.no_config
    
    print(f"\n{'='*60}")
    print(f"PLOTTING THESIS FIGURES FROM SAVED DATA")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    
    # Chapter 6
    if args.all or args.test == 'lovera':
        lovera_file = ch6_dir / "lovera_data.pkl"
        if lovera_file.exists():
            print(f"\nPlotting Lovera from {lovera_file}")
            data = load_data(lovera_file)
            plot_lovera(data, ch6_dir, show_config=show_config)
        else:
            print(f"\n[SKIP] Lovera data not found: {lovera_file}")
    
    if args.all or args.test == 'wisniewski':
        wis_file = ch6_dir / "wisniewski_data.pkl"
        if wis_file.exists():
            print(f"\nPlotting Wisniewski from {wis_file}")
            data = load_data(wis_file)
            plot_wisniewski(data, ch6_dir, show_config=show_config)
        else:
            print(f"\n[SKIP] Wisniewski data not found: {wis_file}")
    
    # Chapter 7
    if args.all or args.test == 'spinning':
        spin_file = ch7_dir / "spinning_data.pkl"
        if spin_file.exists():
            print(f"\nPlotting Spinning from {spin_file}")
            data = load_data(spin_file)
            plot_spinning(data, ch7_dir)
        else:
            print(f"\n[SKIP] Spinning data not found: {spin_file}")
    
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
