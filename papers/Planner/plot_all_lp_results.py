#!/usr/bin/env python3
"""
Plot results from all LP (non-planner) Monte Carlo tests.

Generates comparison plots and statistics for:
- 3MTQ+0RW (Lovera) vs 3MTQ+1RW (LP)
- Reduced attitude vs Full 180° vs Multi-goal scenarios

Saves figures to papers/Planner/figures/
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from glob import glob

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.helpers.save_and_load.save_and_load import load_data
from ADCS.helpers.math_helpers import rot_mat, normalize, quat_mult, quat_conj

FIGURE_DIR = Path(__file__).parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)
DATA_DIR = Path(__file__).parent / "output_data"

# BC2 boresight is Y-axis
BODY_BORESIGHT = np.array([0, 1, 0])


def find_latest_data(pattern: str) -> Optional[Path]:
    """Find the most recent data file matching pattern."""
    matches = list(DATA_DIR.glob(f"{pattern}*"))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def compute_reduced_errors(results: List[Dict], body_boresight: np.ndarray) -> np.ndarray:
    """Compute pointing errors for reduced attitude (boresight alignment)."""
    valid = [r for r in results if r is not None and 'state' in r and 'boresight_goal' in r]
    
    final_errors = []
    for r in valid:
        q = r['state'][-1, 3:7]
        R = rot_mat(q)
        boresight_eci = R @ body_boresight
        goal_eci = r['boresight_goal'][-1]
        
        if np.linalg.norm(goal_eci) < 1e-6:
            continue
        
        goal_eci = goal_eci / np.linalg.norm(goal_eci)
        dot = np.clip(np.dot(boresight_eci, goal_eci), -1, 1)
        final_errors.append(np.arccos(dot) * 180 / np.pi)
    
    return np.array(final_errors)


def compute_full_attitude_errors(results: List[Dict]) -> np.ndarray:
    """Compute quaternion errors for full attitude tracking."""
    valid = [r for r in results if r is not None and 'state' in r and 'q_goal' in r]
    
    final_errors = []
    for r in valid:
        q = r['state'][-1, 3:7]
        q_goal = r['q_goal'][-1]
        
        # Quaternion error: q_err = q_goal^-1 * q
        q_err = quat_mult(quat_conj(q_goal), q)
        
        # Error angle from quaternion
        angle = 2 * np.arccos(np.clip(np.abs(q_err[0]), 0, 1)) * 180 / np.pi
        final_errors.append(angle)
    
    return np.array(final_errors)


def compute_statistics(errors: np.ndarray) -> Dict[str, float]:
    """Compute statistics for errors."""
    if len(errors) == 0:
        return {"mean": np.nan, "median": np.nan, "std": np.nan, "max": np.nan,
                "pct_0.5deg": 0, "pct_1deg": 0, "pct_5deg": 0, "pct_10deg": 0, "n_valid": 0}
    
    return {
        "mean": np.mean(errors),
        "median": np.median(errors),
        "std": np.std(errors),
        "max": np.max(errors),
        "pct_0.5deg": 100 * np.sum(errors < 0.5) / len(errors),
        "pct_1deg": 100 * np.sum(errors < 1) / len(errors),
        "pct_5deg": 100 * np.sum(errors < 5) / len(errors),
        "pct_10deg": 100 * np.sum(errors < 10) / len(errors),
        "n_valid": len(errors),
    }


def plot_histogram(errors: np.ndarray, title: str, save_path: Optional[Path] = None):
    """Plot histogram of final errors."""
    if len(errors) == 0:
        print(f"Skipping {title} - no valid data")
        return None
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    max_err = min(180, np.ceil(np.max(errors) / 10) * 10)
    bins = np.arange(0, max_err + 5, 5)
    
    ax.hist(errors, bins=bins, edgecolor='black', alpha=0.7)
    
    stats = compute_statistics(errors)
    text = f"Mean: {stats['mean']:.2f}°\nMedian: {stats['median']:.2f}°\n"
    text += f"<0.5°: {stats['pct_0.5deg']:.0f}%\n<1°: {stats['pct_1deg']:.0f}%\n"
    text += f"<5°: {stats['pct_5deg']:.0f}%\n<10°: {stats['pct_10deg']:.0f}%"
    ax.text(0.95, 0.95, text, transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Final Error [deg]')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_comparison_grid(all_results: Dict[str, Dict], save_path: Optional[Path] = None):
    """Create a 2x3 comparison grid of all test cases."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    configs = [
        ("3MTQ+0RW", "reduced", "Reduced (180° boresight)"),
        ("3MTQ+0RW", "full", "Full Attitude (180°)"),
        ("3MTQ+0RW", "multi", "Multi-Goal"),
        ("3MTQ+1RW", "reduced", "Reduced (180° boresight)"),
        ("3MTQ+1RW", "full", "Full Attitude (180°)"),
        ("3MTQ+1RW", "multi", "Multi-Goal"),
    ]
    
    for idx, (arch, goal_type, goal_label) in enumerate(configs):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        key = f"{arch}_{goal_type}"
        if key in all_results and all_results[key] is not None:
            errors = all_results[key]["errors"]
            
            if len(errors) > 0:
                max_err = min(90, np.ceil(np.max(errors) / 5) * 5)
                bins = np.arange(0, max_err + 5, 5)
                ax.hist(errors, bins=bins, edgecolor='black', alpha=0.7)
                
                stats = all_results[key]["stats"]
                text = f"μ={stats['mean']:.1f}°, med={stats['median']:.1f}°\n"
                text += f"<0.5°:{stats['pct_0.5deg']:.0f}% <1°:{stats['pct_1deg']:.0f}%\n"
                text += f"<5°:{stats['pct_5deg']:.0f}% <10°:{stats['pct_10deg']:.0f}%"
                ax.text(0.95, 0.95, text, transform=ax.transAxes, ha='right', va='top',
                        fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        controller = "Lovera" if "0RW" in arch else "LP"
        ax.set_title(f"{arch} ({controller})\n{goal_label}")
        ax.set_xlabel('Final Error [deg]')
        if col == 0:
            ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("LP vs Lovera Controller Monte Carlo Results (100 runs, 1000s)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def print_summary_table(all_results: Dict[str, Dict]):
    """Print a summary table of all results."""
    print("\n" + "=" * 100)
    print("LP/Lovera Controller Monte Carlo Results Summary (BC2 Satellite, 1000s)")
    print("=" * 100)
    print(f"{'Configuration':<30} {'N':>5} {'Mean':>8} {'Median':>8} {'<0.5°':>8} {'<1°':>8} {'<5°':>8} {'<10°':>8}")
    print("-" * 100)
    
    for key, data in all_results.items():
        if data is None:
            print(f"{key:<30} {'N/A':>5}")
            continue
        
        stats = data["stats"]
        print(f"{key:<30} {stats['n_valid']:>5} {stats['mean']:>8.2f} {stats['median']:>8.2f} "
              f"{stats['pct_0.5deg']:>7.1f}% {stats['pct_1deg']:>7.1f}% "
              f"{stats['pct_5deg']:>7.1f}% {stats['pct_10deg']:>7.1f}%")
    
    print("=" * 100)


def load_and_process(pattern: str, is_full_attitude: bool = False) -> Optional[Dict]:
    """Load data and compute errors."""
    data_path = find_latest_data(pattern)
    if data_path is None:
        print(f"No data found for pattern: {pattern}")
        return None
    
    print(f"Loading: {data_path.name}")
    results = load_data(str(data_path))
    if isinstance(results, tuple):
        results = results[0]
    
    if is_full_attitude:
        errors = compute_full_attitude_errors(results)
    else:
        errors = compute_reduced_errors(results, BODY_BORESIGHT)
    
    stats = compute_statistics(errors)
    
    return {
        "results": results,
        "errors": errors,
        "stats": stats,
    }


def main():
    print("=" * 60)
    print("Plotting LP/Lovera Monte Carlo Results")
    print("=" * 60)
    
    all_results = {}
    
    test_configs = [
        ("3MTQ+0RW_Lovera_reduced", "3MTQ+0RW_reduced", False),
        ("3MTQ+0RW_Lovera_full180", "3MTQ+0RW_full", True),
        ("3MTQ+0RW_Lovera_multi", "3MTQ+0RW_multi", False),
        ("3MTQ+1RW_LP_reduced", "3MTQ+1RW_reduced", False),
        ("3MTQ+1RW_LP_full180", "3MTQ+1RW_full", True),
        ("3MTQ+1RW_LP_multi", "3MTQ+1RW_multi", False),
    ]
    
    for pattern, key, is_full in test_configs:
        data = load_and_process(pattern, is_full)
        all_results[key] = data
        
        if data is not None:
            goal_type = key.split("_")[1]
            arch = key.split("_")[0]
            controller = "Lovera" if "0RW" in arch else "LP"
            
            plot_histogram(
                data["errors"],
                f"{arch} {controller} {goal_type.title()} - Final Error Distribution",
                FIGURE_DIR / f"{key}_histogram.png"
            )
    
    # Comparison grid
    plot_comparison_grid(all_results, FIGURE_DIR / "lp_comparison_grid.png")
    
    # Summary table
    print_summary_table(all_results)
    
    print(f"\nFigures saved to: {FIGURE_DIR}")
    
    plt.show()


if __name__ == "__main__":
    main()
