"""
Test: Anti-spike planner settings for 3MTQ+1RW Full Attitude 180° slews.

Compared to the current fast_slew tuning, this enables:
1. Path length cost (ang_vel_mag) - penalizes geodesic arc length, prevents long-way-around
2. Cross-term (ang_vel_err_dir) - penalizes spinning in wrong direction
3. Cost function type 2 (geodesic/acos) - strong gradient at 180° (vs type 4 flat top)

Uses C++ planner (Plan_and_Track_LQR) for speed.
Saves figures for visual inspection.
"""
import os
os.environ.setdefault('DISPLAY', ':0')
os.environ['MPLBACKEND'] = 'Agg'  # Non-interactive backend for figure saving
import sys
import numpy as np
import time as time_module
from scipy.integrate import solve_ivp
from typing import Dict, Any

# --- Path Setup ---
_this_file = os.path.abspath(__file__)
_this_dir = os.path.dirname(_this_file)
_root_dir = os.path.abspath(os.path.join(_this_dir, "../../.."))
sys.path.insert(0, _root_dir)
sys.path.insert(0, _this_dir)

from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import PlannerSettings
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import normalize, quat_mult
from ADCS.helpers.plotting_mc.plot_controller_mc import (
    plot_mc_summary, plot_single_run
)

# Import base settings factory - we'll override specific values
from mc_planner_settings import create_optimized_planner_settings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BODY_BORESIGHT = np.array([0, 1, 0])
OUTPUT_DIR = os.path.join(_this_dir, "test_anti_spike_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Test configurations to compare
# ============================================================================
CONFIGS = {
    "baseline": {
        "description": "Current fast_slew (baseline)",
        "ang_vel_mag": 0.0,
        "ang_vel_mag_N": 0.0,
        "cross_term_fraction": 0.0,
        "ang_cost_func_type": None,  # Don't override (keep type 4 from fast_slew)
    },
    "pathlen_only": {
        "description": "fast_slew + path length cost",
        "ang_vel_mag": 1.0,
        "ang_vel_mag_N": 5.0,
        "cross_term_fraction": 0.0,
        "ang_cost_func_type": None,
    },
    "crossterm_2pct": {
        "description": "fast_slew + cross-term 2%",
        "ang_vel_mag": 0.0,
        "ang_vel_mag_N": 0.0,
        "cross_term_fraction": 0.02,  # 2% of PSD max (conservative)
        "ang_cost_func_type": None,
    },
    "geodesic_only": {
        "description": "fast_slew + geodesic cost func (type 2)",
        "ang_vel_mag": 0.0,
        "ang_vel_mag_N": 0.0,
        "cross_term_fraction": 0.0,
        "ang_cost_func_type": 2,
    },
    "pathlen_geodesic": {
        "description": "fast_slew + pathlen + geodesic",
        "ang_vel_mag": 1.0,
        "ang_vel_mag_N": 5.0,
        "cross_term_fraction": 0.0,
        "ang_cost_func_type": 2,
    },
    "all_conservative": {
        "description": "pathlen + cross 2% + geodesic",
        "ang_vel_mag": 1.0,
        "ang_vel_mag_N": 5.0,
        "cross_term_fraction": 0.02,  # 2% of PSD max
        "ang_cost_func_type": 2,
    },
}


def create_test_settings(sat, tf, dt_planning, test_config):
    """Create planner settings with anti-spike modifications applied on top of fast_slew."""
    settings = create_optimized_planner_settings(
        sat, duration=tf, dt_planning=dt_planning, tuning="fast_slew"
    )
    settings.verbosity = False
    settings.dt_tp = 10
    settings.skip_pass2_optimization = True

    # Apply anti-spike overrides
    if test_config["ang_vel_mag"] > 0:
        settings.cost_main.ang_vel_mag = test_config["ang_vel_mag"]
        settings.cost_main.ang_vel_mag_N = test_config["ang_vel_mag_N"]
        settings.cost_second.ang_vel_mag = test_config["ang_vel_mag"]
        settings.cost_second.ang_vel_mag_N = test_config["ang_vel_mag_N"]

    if test_config["cross_term_fraction"] > 0:
        settings.cost_main.set_cross_term_auto(fraction=test_config["cross_term_fraction"])
        settings.cost_second.set_cross_term_auto(fraction=test_config["cross_term_fraction"])

    if test_config["ang_cost_func_type"] is not None:
        settings.cost_main.ang_cost_func_type = test_config["ang_cost_func_type"]
        settings.cost_second.ang_cost_func_type = test_config["ang_cost_func_type"]

    # Verify PSD
    settings.cost_main.check_psd(warn=True)
    settings.cost_second.check_psd(warn=True)

    return settings


def generate_mc_config(run_id: int, tf=1000, dt=1, dt_planning=1) -> Dict[str, Any]:
    """Generate a 180° full-attitude slew config (same as production MC)."""
    rng = np.random.default_rng(seed=run_id + 1000)
    q0 = normalize(rng.standard_normal(4))
    rand_angle = rng.uniform(0, 2 * np.pi)
    axis_body = np.array([np.cos(rand_angle), 0, np.sin(rand_angle)])
    q_180_body = np.array([0, axis_body[0], axis_body[1], axis_body[2]])
    q_goal = quat_mult(q0, q_180_body)
    q_goal = normalize(q_goal)

    return {
        "run_id": run_id,
        "seed": run_id,
        "tf": tf,
        "dt": dt,
        "dt_planning": dt_planning,
        "radius_km": 7000.0,
        "w0": normalize(rng.standard_normal(3)) * (rng.uniform(0.1, 1.0) * np.pi / 180.0),
        "q0": q0,
        "q_goal": q_goal,
        "h0": rng.uniform(-0.0001, 0.0001, size=1),
    }


def compute_quat_error_timeseries(state_hist, q_goal):
    """Compute quaternion error angle at each timestep."""
    q_goal_inv = np.array([q_goal[0], -q_goal[1], -q_goal[2], -q_goal[3]])
    N = state_hist.shape[0]
    angles = np.zeros(N)
    for k in range(N):
        qk = state_hist[k, 3:7]
        qerr_w = q_goal_inv[0]*qk[0] - np.dot(q_goal_inv[1:], qk[1:])
        angles[k] = np.degrees(2 * np.arccos(np.clip(np.abs(qerr_w), 0, 1)))
    return angles


def run_single_sim(config, planner_settings, orb):
    """Run a single closed-loop simulation with C++ planner."""
    run_id = config["run_id"]
    tf = config["tf"]
    dt = config["dt"]
    N = int(tf / dt)

    np.random.seed(config["seed"])
    real_sat = create_beavercube2_cubesat(estimated=False)
    rws = real_sat.rw_actuators

    x0 = np.concatenate([config["w0"], config["q0"], config["h0"]])
    for i, rw in enumerate(rws):
        rw.h = config["h0"][i]

    controller = Plan_and_Track_LQR(est_sat=real_sat, planner_settings=planner_settings)

    goals = GoalList({0.22: Fixed_Attitude_Goal(config["q_goal"])})
    os0 = orb.get_os(0.22)

    try:
        t_plan_start = time_module.time()
        traj = controller.calculate_trajectory(
            t_start=0.22, duration=tf, x_0=x0, os_0=os0, goals=goals, verbose=False
        )
        plan_time = time_module.time() - t_plan_start
        controller.set_active_trajectory(traj)
    except Exception as e:
        return {"run_id": run_id, "config": config, "error": str(e), "traj_valid": False}

    # Closed-loop simulation
    time_hist = np.zeros(N)
    state_hist = np.zeros((N, len(x0)))
    u_hist = np.zeros((N, len(real_sat.actuators)))
    q_goal_hist = np.zeros((N, 4))

    for i, rw in enumerate(rws):
        rw.h = config["h0"][i]

    x = x0.copy()
    t = 0
    sec2cent = TimeConstants.sec2cent

    for i in range(N):
        J2000 = 0.22 + t * sec2cent
        os_state = orb.get_os(J2000=J2000)
        sens = real_sat.sensor_readings(x=x, os=os_state)
        u = controller.find_u(x_hat=x, sens=sens, est_sat=real_sat, os_hat=os_state)

        time_hist[i] = t
        state_hist[i, :] = x
        u_hist[i, :] = u
        q_goal_hist[i, :] = config["q_goal"]

        t += dt
        t_next_clamped = min(t, tf - 0.01)
        os_next = orb.get_os(0.22 + t_next_clamped * sec2cent)
        out = solve_ivp(
            real_sat.dynamics_for_solver, (0, dt), x, method="RK45",
            args=(u, os_state, os_next), rtol=1e-6, atol=1e-6
        )
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])

    # Extract trajectory data
    traj_times_sec = (traj.times - traj.times[0]) * 36525 * 24 * 3600
    traj_state = traj.states.T
    traj_u = traj.controls.T

    return {
        "run_id": run_id, "config": config, "traj_valid": True,
        "time": time_hist, "state": state_hist, "u": u_hist,
        "q_goal": q_goal_hist, "goal_type": "full_attitude",
        "traj_time": traj_times_sec, "traj_state": traj_state, "traj_u": traj_u,
        "plan_time": plan_time,
    }


def plot_comparison(all_results, output_dir):
    """Plot comparison across all test configurations."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for idx, (config_name, results) in enumerate(all_results.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        valid = [r for r in results if r.get("traj_valid", False)]
        if not valid:
            ax.set_title(f"{config_name}: NO VALID RUNS")
            continue

        # Plot all error traces
        for r in valid:
            angles = compute_quat_error_timeseries(r["state"], r["config"]["q_goal"])
            ax.plot(r["time"], angles, alpha=0.3, linewidth=0.5, color='steelblue')

        # Mean and median
        all_angles = []
        for r in valid:
            angles = compute_quat_error_timeseries(r["state"], r["config"]["q_goal"])
            all_angles.append(angles)
        all_angles = np.array(all_angles)

        time_arr = valid[0]["time"]
        mean_angles = np.mean(all_angles, axis=0)
        median_angles = np.median(all_angles, axis=0)
        p10 = np.percentile(all_angles, 10, axis=0)
        p90 = np.percentile(all_angles, 90, axis=0)

        ax.fill_between(time_arr, p10, p90, alpha=0.2, color='blue', label='10-90 pctl')
        ax.plot(time_arr, mean_angles, 'b-', linewidth=2, label=f'Mean (final: {mean_angles[-1]:.1f}°)')
        ax.plot(time_arr, median_angles, 'r--', linewidth=2, label=f'Median (final: {median_angles[-1]:.1f}°)')

        # Spike detection: count runs that go above 150° after initially being below 90°
        n_spikes = 0
        for angles in all_angles:
            # Find first time below 90°
            below_90 = np.where(angles < 90)[0]
            if len(below_90) > 0:
                first_below = below_90[0]
                # Check if it goes back above 150° after that
                if np.any(angles[first_below:] > 150):
                    n_spikes += 1

        final_errors = all_angles[:, -1]
        pct_below_1 = np.mean(final_errors < 1) * 100
        pct_below_5 = np.mean(final_errors < 5) * 100
        pct_below_10 = np.mean(final_errors < 10) * 100

        desc = CONFIGS[config_name]["description"]
        ax.set_title(f"{desc}\n<1°:{pct_below_1:.0f}% <5°:{pct_below_5:.0f}% <10°:{pct_below_10:.0f}% spikes:{n_spikes}/{len(valid)}", fontsize=10)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Quaternion Error [deg]')
        ax.set_ylim(0, 200)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(len(all_results), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Anti-Spike Settings Comparison: 3MTQ+1RW Full 180° Slew", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison_timeseries.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {os.path.join(output_dir, 'comparison_timeseries.png')}")

    # === Histogram comparison ===
    fig2, axes2 = plt.subplots(2, 3, figsize=(20, 12))
    axes2 = axes2.flatten()

    for idx, (config_name, results) in enumerate(all_results.items()):
        if idx >= len(axes2):
            break
        ax = axes2[idx]
        valid = [r for r in results if r.get("traj_valid", False)]
        if not valid:
            continue

        final_errors = []
        plan_times = []
        for r in valid:
            angles = compute_quat_error_timeseries(r["state"], r["config"]["q_goal"])
            final_errors.append(angles[-1])
            if "plan_time" in r:
                plan_times.append(r["plan_time"])

        final_errors = np.array(final_errors)
        ax.hist(final_errors, bins=np.arange(0, max(72, np.max(final_errors)+2), 2),
                color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(1, color='green', linestyle='--', linewidth=2, label=f'<1°: {np.mean(final_errors<1)*100:.0f}%')
        ax.axvline(5, color='orange', linestyle='--', linewidth=2, label=f'<5°: {np.mean(final_errors<5)*100:.0f}%')
        ax.axvline(10, color='red', linestyle='--', linewidth=2, label=f'<10°: {np.mean(final_errors<10)*100:.0f}%')

        desc = CONFIGS[config_name]["description"]
        mean_plan = np.mean(plan_times) if plan_times else 0
        ax.set_title(f"{desc}\nMean:{np.mean(final_errors):.1f}° Med:{np.median(final_errors):.1f}° PlanT:{mean_plan:.1f}s", fontsize=10)
        ax.set_xlabel('Final Error [deg]')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(len(all_results), len(axes2)):
        axes2[idx].set_visible(False)

    fig2.suptitle("Final Error Histograms: 3MTQ+1RW Full 180° Slew", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "comparison_histograms.png"), dpi=150)
    plt.close(fig2)
    print(f"Saved: {os.path.join(output_dir, 'comparison_histograms.png')}")


def plot_individual_config(config_name, results, output_dir):
    """Plot detailed timeseries + histogram for a single config."""
    valid = [r for r in results if r.get("traj_valid", False)]
    if not valid:
        return

    all_angles = []
    for r in valid:
        angles = compute_quat_error_timeseries(r["state"], r["config"]["q_goal"])
        all_angles.append(angles)
    all_angles = np.array(all_angles)
    time_arr = valid[0]["time"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Timeseries
    for i, angles in enumerate(all_angles):
        ax1.plot(time_arr, angles, alpha=0.25, linewidth=0.5, color='steelblue')
    mean_angles = np.mean(all_angles, axis=0)
    median_angles = np.median(all_angles, axis=0)
    p10 = np.percentile(all_angles, 10, axis=0)
    p90 = np.percentile(all_angles, 90, axis=0)
    ax1.fill_between(time_arr, p10, p90, alpha=0.2, color='blue', label='10-90 pctl')
    ax1.plot(time_arr, mean_angles, 'b-', linewidth=2, label=f'Mean (final: {mean_angles[-1]:.2f}°)')
    ax1.plot(time_arr, median_angles, 'r--', linewidth=2, label=f'Median (final: {median_angles[-1]:.2f}°)')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Quaternion Error [deg]')
    desc = CONFIGS[config_name]["description"]
    ax1.set_title(f"{desc} (N={len(valid)})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Histogram
    final_errors = all_angles[:, -1]
    ax2.hist(final_errors, bins=np.arange(0, max(72, np.max(final_errors)+2), 2),
             color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(1, color='green', linestyle='--', linewidth=2, label=f'<1°: {np.mean(final_errors<1)*100:.0f}%')
    ax2.axvline(5, color='orange', linestyle='--', linewidth=2, label=f'<5°: {np.mean(final_errors<5)*100:.0f}%')
    ax2.axvline(10, color='red', linestyle='--', linewidth=2, label=f'<10°: {np.mean(final_errors<10)*100:.0f}%')
    ax2.set_xlabel('Final Error [deg]')
    ax2.set_ylabel('Count')
    ax2.set_title(f"Mean: {np.mean(final_errors):.2f}°  Std: {np.std(final_errors):.2f}°")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{config_name}_detail.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {os.path.join(output_dir, f'{config_name}_detail.png')}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test anti-spike planner settings")
    parser.add_argument("-n", "--num-runs", type=int, default=20, help="Number of MC runs per config")
    parser.add_argument("--configs", nargs="+", default=None,
                        help=f"Which configs to test (default: all). Choices: {list(CONFIGS.keys())}")
    parser.add_argument("--tf", type=float, default=1000, help="Simulation duration [s]")
    parser.add_argument("--dt", type=float, default=1, help="Sim timestep [s]")
    parser.add_argument("--dt-planning", type=float, default=1, help="Planning timestep [s]")
    args = parser.parse_args()

    NUM_RUNS = args.num_runs
    configs_to_test = args.configs or list(CONFIGS.keys())
    tf = args.tf
    dt = args.dt
    dt_planning = args.dt_planning

    print(f"=== Anti-Spike Settings Test ===")
    print(f"NUM_RUNS={NUM_RUNS}, tf={tf}s, dt={dt}s, dt_planning={dt_planning}s")
    print(f"Testing: {configs_to_test}")
    print()

    # Create orbit once (shared across all tests for fair comparison)
    print("Creating orbit...", flush=True)
    np.random.seed(100_000)
    orb = create_random_circular_orbit(radius_km=7000.0, dt=dt_planning, tf=tf, use_J2=True, fast=True)
    orb.populate_environment(compute_B=True, compute_S=True)
    print("Orbit ready.", flush=True)

    # Generate configs (same seeds for all test variants)
    mc_configs = [generate_mc_config(i, tf=tf, dt=dt, dt_planning=dt_planning) for i in range(NUM_RUNS)]

    all_results = {}

    for config_name in configs_to_test:
        if config_name not in CONFIGS:
            print(f"WARNING: Unknown config '{config_name}', skipping")
            continue

        test_config = CONFIGS[config_name]
        print(f"\n{'='*60}")
        print(f"Testing: {config_name} - {test_config['description']}")
        print(f"{'='*60}")

        # Create satellite and settings for this config
        real_sat_template = create_beavercube2_cubesat(estimated=False)
        settings = create_test_settings(real_sat_template, tf, dt_planning, test_config)

        # Print key settings
        print(f"  ang_vel_mag:      {settings.cost_main.ang_vel_mag}")
        print(f"  ang_vel_mag_N:    {settings.cost_main.ang_vel_mag_N}")
        print(f"  ang_vel_err_dir:  {settings.cost_main.ang_vel_err_dir}")
        print(f"  ang_cost_func:    {settings.cost_main.ang_cost_func_type}")
        print(f"  angle:            {settings.cost_main.angle:.1f}")
        print(f"  ang_vel:          {settings.cost_main.ang_vel:.1f}")
        print(f"  angle_N:          {settings.cost_main.angle_N:.1f}")
        print(f"  ang_vel_N:        {settings.cost_main.ang_vel_N:.1f}")
        print()

        results = []
        for i, mc_config in enumerate(mc_configs):
            t0 = time_module.time()
            # Fresh satellite for each run
            real_sat = create_beavercube2_cubesat(estimated=False)
            run_settings = create_test_settings(real_sat, tf, dt_planning, test_config)

            result = run_single_sim(mc_config, run_settings, orb)
            elapsed = time_module.time() - t0

            if result.get("traj_valid", False):
                angles = compute_quat_error_timeseries(result["state"], mc_config["q_goal"])
                final_err = angles[-1]
                max_err = np.max(angles)
                # Spike detection
                below_90 = np.where(angles < 90)[0]
                has_spike = False
                if len(below_90) > 0:
                    has_spike = np.any(angles[below_90[0]:] > 150)
                spike_str = " ** SPIKE **" if has_spike else ""
                print(f"  Run {i:3d}: final={final_err:6.1f}°  max={max_err:6.1f}°  plan={result.get('plan_time',0):.1f}s  total={elapsed:.1f}s{spike_str}", flush=True)
            else:
                print(f"  Run {i:3d}: FAILED - {result.get('error', 'unknown')}", flush=True)

            results.append(result)

        all_results[config_name] = results

        # Summary for this config
        valid = [r for r in results if r.get("traj_valid", False)]
        if valid:
            final_errors = []
            spike_count = 0
            for r in valid:
                angles = compute_quat_error_timeseries(r["state"], r["config"]["q_goal"])
                final_errors.append(angles[-1])
                below_90 = np.where(angles < 90)[0]
                if len(below_90) > 0 and np.any(angles[below_90[0]:] > 150):
                    spike_count += 1
            final_errors = np.array(final_errors)
            print(f"\n  SUMMARY ({config_name}):")
            print(f"    Valid: {len(valid)}/{len(results)}")
            print(f"    Final error: mean={np.mean(final_errors):.2f}°  median={np.median(final_errors):.2f}°  std={np.std(final_errors):.2f}°")
            print(f"    <1°: {np.mean(final_errors<1)*100:.0f}%  <5°: {np.mean(final_errors<5)*100:.0f}%  <10°: {np.mean(final_errors<10)*100:.0f}%")
            print(f"    Spikes (>150° after <90°): {spike_count}/{len(valid)}")
            mean_plan = np.mean([r.get("plan_time", 0) for r in valid])
            print(f"    Mean plan time: {mean_plan:.1f}s")

        # Save individual detail plot
        plot_individual_config(config_name, results, OUTPUT_DIR)

    # Save comparison plot
    if len(all_results) > 1:
        plot_comparison(all_results, OUTPUT_DIR)

    print(f"\n=== All figures saved to {OUTPUT_DIR} ===")
