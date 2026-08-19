"""
Plot all Monte Carlo results for Planner paper.

Generates figures for each test configuration showing:
1. Time series with mean and 10/90 percentiles
2. Final error histogram
3. Summary statistics

Saves all figures to papers/Planner/figures/
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.helpers.save_and_load.save_and_load import load_data
from ADCS.helpers.math_helpers import rot_mat, quat_mult, quat_inv

# Configuration
BODY_BORESIGHT = np.array([0, 1, 0])  # BC2 boresight is Y-axis
DATA_DIR = Path("papers/Planner/output_data")
FIG_DIR = Path("papers/Planner/figures")
FIG_DIR.mkdir(exist_ok=True)


def compute_boresight_errors(results: List[Dict], goal_key: str = "boresight_goal") -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute boresight errors over time for all runs.
    
    Returns:
        time: (N,) time array
        errors: (M, N) array of errors in degrees for M runs, N timesteps
    """
    valid = [r for r in results if r is not None and "state" in r]
    if not valid:
        return np.array([]), np.array([])
    
    time = valid[0]["time"]
    N = len(time)
    errors = []
    
    for r in valid:
        state = r["state"]
        goal = r.get(goal_key, None)
        
        if goal is None:
            continue
            
        run_errors = np.zeros(N)
        for i in range(N):
            q = state[i, 3:7]
            R = rot_mat(q)
            bore = R @ BODY_BORESIGHT
            
            # Get goal - could be constant or time-varying
            if goal.ndim == 1:
                goal_vec = goal
            else:
                goal_vec = goal[i]
            
            if np.linalg.norm(goal_vec) < 0.1:  # No_Goal period
                run_errors[i] = np.nan
            else:
                goal_vec = goal_vec / np.linalg.norm(goal_vec)
                dot = np.clip(np.dot(bore, goal_vec), -1, 1)
                run_errors[i] = np.arccos(dot) * 180 / np.pi
        
        errors.append(run_errors)
    
    return time, np.array(errors)


def compute_quaternion_errors(results: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute quaternion errors over time for full attitude tests.
    
    Returns:
        time: (N,) time array
        errors: (M, N) array of errors in degrees for M runs, N timesteps
    """
    valid = [r for r in results if r is not None and "state" in r and "q_goal" in r]
    if not valid:
        return np.array([]), np.array([])
    
    time = valid[0]["time"]
    N = len(time)
    errors = []
    
    for r in valid:
        state = r["state"]
        q_goal_hist = r["q_goal"]
        
        run_errors = np.zeros(N)
        for i in range(N):
            q = state[i, 3:7]
            
            if q_goal_hist.ndim == 1:
                q_goal = q_goal_hist
            else:
                q_goal = q_goal_hist[i]
            
            q_err = quat_mult(quat_inv(q_goal), q)
            run_errors[i] = 2 * np.arccos(np.clip(np.abs(q_err[0]), 0, 1)) * 180 / np.pi
        
        errors.append(run_errors)
    
    return time, np.array(errors)


def plot_error_timeseries(time: np.ndarray, errors: np.ndarray, 
                          title: str, ylabel: str = "Pointing Error [deg]",
                          save_path: Optional[Path] = None) -> None:
    """
    Plot error time series with mean and 10/90 percentiles.
    Generates two figures: one with all runs, one clean with just statistics.
    """
    if len(errors) == 0:
        print(f"No data for {title}")
        return
    
    # Compute statistics (ignoring NaN for No_Goal periods)
    with np.errstate(all='ignore'):
        mean = np.nanmean(errors, axis=0)
        p10 = np.nanpercentile(errors, 10, axis=0)
        p90 = np.nanpercentile(errors, 90, axis=0)
        median = np.nanmedian(errors, axis=0)
    
    # Figure 1: All runs with statistics overlay
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot individual runs with low opacity
    for i in range(min(len(errors), 100)):  # Limit to 100 runs for visibility
        ax.plot(time, errors[i], 'b-', alpha=0.1, linewidth=0.5)
    
    # Plot statistics
    ax.fill_between(time, p10, p90, alpha=0.3, color='blue', label='10-90 percentile')
    ax.plot(time, mean, 'b-', linewidth=2, label=f'Mean (final: {mean[-1]:.2f}°)')
    ax.plot(time, median, 'r--', linewidth=1.5, label=f'Median (final: {median[-1]:.2f}°)')
    
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (N={len(errors)})")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylim([0, min(200, np.nanmax(p90) * 1.1)])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    
    # Figure 2: Clean version with just statistics (no individual runs)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.fill_between(time, p10, p90, alpha=0.3, color='blue', label='10-90 percentile')
    ax.plot(time, mean, 'b-', linewidth=2, label=f'Mean (final: {mean[-1]:.2f}°)')
    ax.plot(time, median, 'r--', linewidth=1.5, label=f'Median (final: {median[-1]:.2f}°)')
    
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (N={len(errors)})")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylim([0, min(200, np.nanmax(p90) * 1.1)])
    
    plt.tight_layout()
    
    if save_path:
        clean_path = save_path.parent / save_path.name.replace("_timeseries", "_timeseries_clean")
        plt.savefig(clean_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {clean_path}")
    
    plt.close()


def plot_histogram(errors: np.ndarray, title: str, 
                   save_path: Optional[Path] = None,
                   threshold_lines: List[float] = [1, 5, 10]) -> Dict[str, float]:
    """
    Plot histogram of final errors with statistics.
    """
    # Get final errors (last timestep, ignoring NaN)
    final_errors = errors[:, -1]
    final_errors = final_errors[~np.isnan(final_errors)]
    
    if len(final_errors) == 0:
        print(f"No valid final errors for {title}")
        return {}
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Determine bins
    max_err = min(180, np.percentile(final_errors, 99))
    if max_err < 20:
        bins = np.arange(0, max_err + 1, 0.5)
    elif max_err < 50:
        bins = np.arange(0, max_err + 2, 2)
    else:
        bins = np.arange(0, max_err + 5, 5)
    
    ax.hist(final_errors, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    
    # Add threshold lines
    colors = ['green', 'orange', 'red']
    for thresh, color in zip(threshold_lines, colors):
        pct = 100 * np.sum(final_errors < thresh) / len(final_errors)
        ax.axvline(thresh, color=color, linestyle='--', linewidth=2, 
                   label=f'<{thresh}°: {pct:.1f}%')
    
    # Statistics text
    stats = {
        'mean': np.mean(final_errors),
        'median': np.median(final_errors),
        'std': np.std(final_errors),
        'min': np.min(final_errors),
        'max': np.max(final_errors),
        'pct_1deg': 100 * np.sum(final_errors < 1) / len(final_errors),
        'pct_5deg': 100 * np.sum(final_errors < 5) / len(final_errors),
        'pct_10deg': 100 * np.sum(final_errors < 10) / len(final_errors),
    }
    
    stats_text = f"Mean: {stats['mean']:.2f}°\nMedian: {stats['median']:.2f}°\nStd: {stats['std']:.2f}°"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)
    
    ax.set_xlabel("Final Pointing Error [deg]")
    ax.set_ylabel("Count")
    ax.set_title(f"{title} (N={len(final_errors)})")
    ax.legend(loc='upper left' if stats['mean'] > 10 else 'upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    
    return stats


def plot_multi_goal_breakdown(results: List[Dict], title: str,
                               save_path: Optional[Path] = None) -> None:
    """
    Plot error at end of each goal period for multi-goal tests.
    
    Timeline: Goal1 (0-300s) → No_Goal → Goal2 (350-650s) → No_Goal → Goal3 (700-1000s)
    """
    valid = [r for r in results if r is not None and "state" in r]
    if not valid:
        return
    
    dt = 2  # timestep
    # Goal end indices (just before No_Goal)
    goal1_idx = 149  # t=298s
    goal2_idx = 324  # t=648s  
    goal3_idx = 499  # t=998s
    
    errs = {1: [], 2: [], 3: []}
    
    for r in valid:
        state = r["state"]
        goal = r["boresight_goal"]
        
        for goal_num, idx in [(1, goal1_idx), (2, goal2_idx), (3, goal3_idx)]:
            q = state[idx, 3:7]
            R = rot_mat(q)
            bore = R @ BODY_BORESIGHT
            goal_vec = goal[idx]
            
            if np.linalg.norm(goal_vec) > 0.1:
                goal_vec = goal_vec / np.linalg.norm(goal_vec)
                dot = np.clip(np.dot(bore, goal_vec), -1, 1)
                errs[goal_num].append(np.arccos(dot) * 180 / np.pi)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for i, (goal_num, goal_errs) in enumerate(errs.items()):
        if not goal_errs:
            continue
        goal_errs = np.array(goal_errs)
        
        ax = axes[i]
        max_err = min(180, np.percentile(goal_errs, 99))
        bins = np.arange(0, max_err + 2, max(1, max_err/20))
        
        ax.hist(goal_errs, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        
        mean = np.mean(goal_errs)
        pct_1 = 100 * np.sum(goal_errs < 1) / len(goal_errs)
        pct_5 = 100 * np.sum(goal_errs < 5) / len(goal_errs)
        pct_10 = 100 * np.sum(goal_errs < 10) / len(goal_errs)
        
        ax.set_title(f"Goal {goal_num}\nMean: {mean:.1f}°, <5°: {pct_5:.0f}%")
        ax.set_xlabel("Error [deg]")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"{title} - Multi-Goal Breakdown", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def process_test(name: str, pattern: str, goal_type: str = "reduced") -> Optional[Dict]:
    """
    Process a single test: load data, generate plots, return stats.
    
    Args:
        name: Display name for the test
        pattern: Glob pattern to find data directory
        goal_type: "reduced" (boresight), "full" (quaternion), or "multi"
    """
    matches = list(DATA_DIR.glob(pattern))
    if not matches:
        print(f"No data found for: {pattern}")
        return None
    
    data_path = max(matches, key=lambda p: p.stat().st_mtime)
    print(f"\nProcessing: {name}")
    print(f"  Data: {data_path.name}")
    
    results = load_data(str(data_path))
    if isinstance(results, tuple):
        results = results[0]
    
    valid = [r for r in results if r is not None and "state" in r]
    print(f"  Valid runs: {len(valid)}/{len(results)}")
    
    if not valid:
        return None
    
    # Sanitize name for filename
    safe_name = name.replace(" ", "_").replace("+", "p").replace("/", "_")
    
    # Compute errors based on goal type
    if goal_type == "full":
        time, errors = compute_quaternion_errors(valid)
        ylabel = "Quaternion Error [deg]"
    else:
        time, errors = compute_boresight_errors(valid)
        ylabel = "Boresight Error [deg]"
    
    if len(errors) == 0:
        print("  No valid error data")
        return None
    
    # Plot time series
    plot_error_timeseries(
        time, errors, name, ylabel=ylabel,
        save_path=FIG_DIR / f"{safe_name}_timeseries.png"
    )
    
    # Plot histogram
    stats = plot_histogram(
        errors, name,
        save_path=FIG_DIR / f"{safe_name}_histogram.png"
    )
    
    # Multi-goal breakdown
    if goal_type == "multi":
        plot_multi_goal_breakdown(
            valid, name,
            save_path=FIG_DIR / f"{safe_name}_multigoal.png"
        )
    
    return stats


def main():
    print("=" * 60)
    print("Generating MC Result Figures")
    print("=" * 60)
    
    # Define all tests to process
    tests = [
        # LP tests
        ("3MTQ+1RW LP Reduced 180°", "3MTQ+1RW_LP_reduced_mc_*", "reduced"),
        ("3MTQ+1RW LP Full 180°", "3MTQ+1RW_LP_full180_mc_*", "full"),
        ("3MTQ+1RW LP Multi-Goal", "3MTQ+1RW_LP_multi_mc_*", "multi"),
        
        ("3MTQ+0RW Lovera Reduced 180°", "3MTQ+0RW_Lovera_reduced_mc_*", "reduced"),
        ("3MTQ+0RW Lovera Full 180°", "3MTQ+0RW_Lovera_full180_mc_*", "full"),
        ("3MTQ+0RW Lovera Multi-Goal", "3MTQ+0RW_Lovera_multi_mc_*", "multi"),
        
        # Planner tests (if available)
        ("3MTQ+1RW Planner Reduced 180°", "3MTQ+1RW_plan_reduced_mc_*", "reduced"),
        ("3MTQ+1RW Planner Full 180°", "3MTQ+1RW_plan_full180_mc_*", "full"),
        ("3MTQ+1RW Planner Multi-Goal", "3MTQ+1RW_plan_multi_mc_*", "multi"),
        
        ("3MTQ+0RW Planner Reduced 180°", "3MTQ+0RW_plan_reduced_mc_*", "reduced"),
        ("3MTQ+0RW Planner Full 180°", "3MTQ+0RW_plan_full180_mc_*", "full"),
        ("3MTQ+0RW Planner Multi-Goal", "3MTQ+0RW_plan_multi_mc_*", "multi"),
    ]
    
    all_stats = {}
    
    for name, pattern, goal_type in tests:
        stats = process_test(name, pattern, goal_type)
        if stats:
            all_stats[name] = stats
    
    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Test':<40} {'Mean':>8} {'Median':>8} {'<1°':>8} {'<5°':>8} {'<10°':>8}")
    print("-" * 100)
    
    for name, stats in all_stats.items():
        print(f"{name:<40} {stats['mean']:>8.2f} {stats['median']:>8.2f} "
              f"{stats['pct_1deg']:>7.1f}% {stats['pct_5deg']:>7.1f}% {stats['pct_10deg']:>7.1f}%")
    
    print("=" * 100)
    print(f"\nFigures saved to: {FIG_DIR.absolute()}")


if __name__ == "__main__":
    main()
