"""
Trajectory-Based Monte Carlo for 3MTQ + 1RW Configuration (BC2 Satellite).

This script generates Monte Carlo simulations comparing:
1. Raw trajectory (open-loop feedforward from ALTRO planner)
2. Simulated LQR tracking (closed-loop TVLQR feedback control)

Configuration matches generate_bc2_lp.py:
- Uses BC2 satellite configuration
- Uses GG, SRP, and Drag Disturbances
- Initial RW momentum randomized between +/- 0.0001 Nms
- Randomized circular orbit position per worker
"""
import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any, Tuple, Optional, List

# --- Path Setup ---
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

# --- ADCS Imports ---
from ADCS.CONOPS.goals import ECI_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import PlannerSettings, Trajectory
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants, EarthConstants
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize
from ADCS.helpers.save_and_load.save_and_load import save_data, load_data
from ADCS.helpers.plotting_mc.plot_controller_mc import (
    plot_target_tracking_mc,
    plot_convergence_histogram_mc,
    plot_h_tracking_mc,
)
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window

# --- MC Runner Imports ---
from ADCS.helpers.mc.monte_carlo_runner import (
    MonteCarloRunner,
    claim_worker_slot,
    release_worker_slot,
    update_worker_progress,
)

# --- GLOBAL WORKER CACHE ---
_CACHED_ORBIT: Optional[Orbit] = None
_CACHED_ORBIT_KEY: Optional[Tuple] = None


def run_single_sim(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function for trajectory-based Monte Carlo.

    Runs two simulations per config:
    1. Raw trajectory (open-loop feedforward)
    2. LQR tracking (closed-loop TVLQR)

    Returns results for both modes.
    """
    global _CACHED_ORBIT, _CACHED_ORBIT_KEY

    slot_id = claim_worker_slot()
    run_id = config["run_id"]

    try:
        tf = config.get("tf", 1000)
        dt = config.get("dt", 2)
        dt_planning = config.get("dt_planning", 1)
        t0 = 0
        N = int((tf - t0) / dt)

        # --- Orbit Retrieval (Cached) -> ONE RANDOM CIRCULAR ORBIT PER WORKER SLOT ---
        radius_km = float(config.get("radius_km", 7000.0))
        orbit_key = (slot_id, radius_km, tf, dt_planning, True, False)

        if _CACHED_ORBIT is None or _CACHED_ORBIT_KEY != orbit_key:
            rng_state = np.random.get_state()
            try:
                np.random.seed(100_000 + int(slot_id))
                _CACHED_ORBIT = create_random_circular_orbit(
                    radius_km=radius_km,
                    dt=dt_planning,
                    tf=tf,
                    use_J2=True,
                    fast=True,
                )
                _CACHED_ORBIT_KEY = orbit_key
            finally:
                np.random.set_state(rng_state)

        orb = _CACHED_ORBIT

        # --- Setup Randomness (per run) ---
        np.random.seed(config["seed"])

        # --- Hardware Setup (BC2 satellite) ---
        real_sat = create_beavercube2_cubesat(estimated=False)
        rws = real_sat.rw_actuators
        rwN = len(rws)

        # --- Initial Conditions ---
        x0 = np.concatenate([config["w0"], config["q0"], config["h0"]])
        for i, rw in enumerate(rws):
            rw.h = config["h0"][i]

        # --- Planner Setup ---
        planner_settings = PlannerSettings(
            est_sat=real_sat,
            bdot_on=0,
            dt_tp=dt_planning,
            dt_tvlqr=dt_planning,
        )
        planner_settings.verbosity = False

        controller = Plan_and_Track_LQR(
            est_sat=real_sat,
            planner_settings=planner_settings,
        )

        # --- Goal Setup ---
        goals = GoalList({0.22: ECI_Goal(config["goal_eci_vec"])})
        os0 = orb.get_os(0.22)

        # --- Generate Trajectory ---
        try:
            traj: Trajectory = controller.calculate_trajectory(
                t_start=0.22,
                duration=tf - t0,
                x_0=x0,
                os_0=os0,
                goals=goals,
                verbose=False,
            )
            controller.set_active_trajectory(traj)
            traj_valid = True
        except Exception as e:
            return {
                "run_id": run_id,
                "config": config,
                "error": str(e),
                "traj_valid": False,
            }

        # --- Initialize Arrays ---
        # Raw trajectory results
        time_hist_raw = np.zeros(N)
        state_hist_raw = np.zeros((N, len(x0)))
        u_hist_raw = np.zeros((N, len(real_sat.actuators)))
        boresight_hist_raw = np.zeros((N, 3))

        # LQR tracking results
        time_hist_lqr = np.zeros(N)
        state_hist_lqr = np.zeros((N, len(x0)))
        u_hist_lqr = np.zeros((N, len(real_sat.actuators)))
        boresight_hist_lqr = np.zeros((N, 3))

        # Cache function references
        orb_get_os = orb.get_os
        sat_sensor_readings = real_sat.sensor_readings
        sat_dynamics = real_sat.dynamics_for_solver
        goal_to_ref = goals.to_ref
        sec2cent = TimeConstants.sec2cent

        # =====================
        # SIMULATION 1: Raw Trajectory (Open-Loop)
        # =====================
        x_raw = x0.copy()
        t = t0

        for i in range(N):
            if i % 20 == 0:
                update_worker_progress(slot_id, run_id, i, N * 2)

            J2000 = 0.22 + t * sec2cent
            os_state = orb_get_os(J2000=J2000)

            # Get feedforward control from trajectory (no feedback)
            u_ff = traj.get_control_at(J2000)

            time_hist_raw[i] = t
            state_hist_raw[i, :] = x_raw
            u_hist_raw[i, :] = u_ff
            eci_goal_ref, _ = goal_to_ref(t=J2000, os0=os_state)
            boresight_hist_raw[i, :] = eci_goal_ref

            # Integrate dynamics
            t += dt
            prev_os = os_state
            os_next = orb_get_os(0.22 + t * sec2cent)

            out = solve_ivp(
                fun=sat_dynamics,
                t_span=(0, dt),
                y0=x_raw,
                method="RK45",
                args=(u_ff, prev_os, os_next),
                rtol=1e-6,
                atol=1e-6,
            )
            x_raw = out.y[:, -1]
            x_raw[3:7] = normalize(x_raw[3:7])

        # =====================
        # SIMULATION 2: LQR Tracking (Closed-Loop)
        # =====================
        # Reset RW momentum state
        for i, rw in enumerate(rws):
            rw.h = config["h0"][i]

        x_lqr = x0.copy()
        t = t0

        for i in range(N):
            if i % 20 == 0:
                update_worker_progress(slot_id, run_id, N + i, N * 2)

            J2000 = 0.22 + t * sec2cent
            os_state = orb_get_os(J2000=J2000)

            # Get sensor readings and compute closed-loop control
            sens = sat_sensor_readings(x=x_lqr, os=os_state)
            u_lqr = controller.find_u(
                x_hat=x_lqr, sens=sens, est_sat=real_sat, os_hat=os_state
            )

            time_hist_lqr[i] = t
            state_hist_lqr[i, :] = x_lqr
            u_hist_lqr[i, :] = u_lqr
            eci_goal_ref, _ = goal_to_ref(t=J2000, os0=os_state)
            boresight_hist_lqr[i, :] = eci_goal_ref

            # Integrate dynamics
            t += dt
            prev_os = os_state
            os_next = orb_get_os(0.22 + t * sec2cent)

            out = solve_ivp(
                fun=sat_dynamics,
                t_span=(0, dt),
                y0=x_lqr,
                method="RK45",
                args=(u_lqr, prev_os, os_next),
                rtol=1e-6,
                atol=1e-6,
            )
            x_lqr = out.y[:, -1]
            x_lqr[3:7] = normalize(x_lqr[3:7])

        # Final progress update
        update_worker_progress(slot_id, run_id, N * 2, N * 2)

        return {
            "run_id": run_id,
            "config": config,
            "traj_valid": True,
            # Raw trajectory results
            "time_raw": time_hist_raw,
            "state_raw": state_hist_raw,
            "u_raw": u_hist_raw,
            "boresight_goal_raw": boresight_hist_raw,
            # LQR tracking results
            "time_lqr": time_hist_lqr,
            "state_lqr": state_hist_lqr,
            "u_lqr": u_hist_lqr,
            "boresight_goal_lqr": boresight_hist_lqr,
            # Also provide standard keys for compatibility (LQR as default)
            "time": time_hist_lqr,
            "state": state_hist_lqr,
            "u": u_hist_lqr,
            "boresight_goal": boresight_hist_lqr,
        }

    finally:
        release_worker_slot(slot_id)


def generate_mc_config(run_id: int) -> Dict[str, Any]:
    """Generate randomized configuration for a single MC run.

    Matches the BC2 LP MC initial conditions:
    - tf: 1000s
    - dt: 2s
    - w0: random direction, magnitude 0.1-1.0 deg/s
    - q0: random quaternion
    - h0: uniform [-0.0001, 0.0001] Nms (1 RW)
    - Randomized circular orbit per worker
    - Perfect state knowledge (no estimator noise)
    """
    rng = np.random.default_rng(seed=run_id + 1000)
    return {
        "run_id": run_id,
        "seed": run_id,
        "tf": 1000,
        "dt": 2,
        "dt_planning": 1,
        "radius_km": 7000.0,
        "w0": normalize(rng.standard_normal(3)) * (rng.uniform(0.1, 1.0) * np.pi / 180.0),
        "q0": normalize(rng.standard_normal(4)),
        "h0": rng.uniform(-0.0001, 0.0001, size=1),  # 1 RW
        "goal_eci_vec": normalize(rng.standard_normal(3)),
    }


def extract_raw_results(full_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract raw trajectory results in standard format for plotting."""
    raw_results = []
    for res in full_results:
        if res is None:
            continue
        if not res.get("traj_valid", False):
            continue
        raw_results.append({
            "run_id": res["run_id"],
            "time": res["time_raw"],
            "state": res["state_raw"],
            "u": res["u_raw"],
            "boresight_goal": res["boresight_goal_raw"],
        })
    return raw_results


def extract_lqr_results(full_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract LQR tracking results in standard format for plotting."""
    lqr_results = []
    for res in full_results:
        if res is None:
            continue
        if not res.get("traj_valid", False):
            continue
        lqr_results.append({
            "run_id": res["run_id"],
            "time": res["time_lqr"],
            "state": res["state_lqr"],
            "u": res["u_lqr"],
            "boresight_goal": res["boresight_goal_lqr"],
        })
    return lqr_results


def plot_comparison(full_results: List[Dict[str, Any]]) -> None:
    """Plot comparison between raw trajectory and LQR tracking."""
    import matplotlib.pyplot as plt

    raw_results = extract_raw_results(full_results)
    lqr_results = extract_lqr_results(full_results)

    # Count valid runs
    n_valid = len(raw_results)
    n_total = len(full_results)
    n_failed = n_total - n_valid

    print(f"\n--- Results Summary ---")
    print(f"Total runs: {n_total}")
    print(f"Valid trajectories: {n_valid}")
    print(f"Failed trajectories: {n_failed}")

    if n_valid == 0:
        print("No valid results to plot!")
        return

    # Plot tracking error time series
    plot_target_tracking_mc(raw_results, body_boresight=np.array([0, 1, 0]), title=f"BC2 3MTQ+1RW Raw Trajectory (Open-Loop) N={n_valid}")
    plot_target_tracking_mc(lqr_results, body_boresight=np.array([0, 1, 0]), title=f"BC2 3MTQ+1RW LQR Tracking (Closed-Loop) N={n_valid}")

    # Plot RW momentum
    plot_h_tracking_mc(raw_results, title=f"BC2 Raw Trajectory RW Momentum N={n_valid}")
    plot_h_tracking_mc(lqr_results, title=f"BC2 LQR Tracking RW Momentum N={n_valid}")

    # Plot convergence histograms
    errors_raw, stats_raw = plot_convergence_histogram_mc(
        raw_results, title=f"Raw Trajectory Final Error N={n_valid}"
    )
    errors_lqr, stats_lqr = plot_convergence_histogram_mc(
        lqr_results, title=f"LQR Tracking Final Error N={n_valid}"
    )

    # Print comparison statistics
    print(f"\n--- Raw Trajectory Statistics ---")
    print(f"  Mean error: {stats_raw['mean']:.3f} deg")
    print(f"  Median error: {stats_raw['median']:.3f} deg")
    print(f"  Max error: {stats_raw['max']:.3f} deg")
    print(f"  % < 0.5 deg: {stats_raw['pct_under_thresh']:.1f}%")

    print(f"\n--- LQR Tracking Statistics ---")
    print(f"  Mean error: {stats_lqr['mean']:.3f} deg")
    print(f"  Median error: {stats_lqr['median']:.3f} deg")
    print(f"  Max error: {stats_lqr['max']:.3f} deg")
    print(f"  % < 0.5 deg: {stats_lqr['pct_under_thresh']:.1f}%")

    # Create side-by-side comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if len(errors_raw) > 0 and len(errors_lqr) > 0:
        max_edge = np.ceil(max(errors_raw.max(), errors_lqr.max()) / 5) * 5
        bins = np.arange(0, max_edge + 5, 5)

        # Raw trajectory histogram
        axes[0].hist(errors_raw, bins=bins, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel("Final Tracking Error [deg]")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Raw Trajectory (Open-Loop)")
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].text(0.95, 0.95, f"mean: {stats_raw['mean']:.2f}\nmedian: {stats_raw['median']:.2f}",
                     transform=axes[0].transAxes, ha='right', va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # LQR tracking histogram
        axes[1].hist(errors_lqr, bins=bins, edgecolor='black', alpha=0.7, color='tab:orange')
        axes[1].set_xlabel("Final Tracking Error [deg]")
        axes[1].set_ylabel("Count")
        axes[1].set_title("LQR Tracking (Closed-Loop)")
        axes[1].grid(True, linestyle='--', alpha=0.6)
        axes[1].text(0.95, 0.95, f"mean: {stats_lqr['mean']:.2f}\nmedian: {stats_lqr['median']:.2f}",
                     transform=axes[1].transAxes, ha='right', va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle("BC2 3MTQ + 1RW Trajectory-Based MC Comparison", fontsize=12)
    plt.tight_layout()


if __name__ == "__main__":
    RUN_MC: bool = True
    OUTPUT_DIR = "papers/3MTQ+1RW/output_data"

    if RUN_MC:
        print("Starting trajectory-based Monte Carlo with 100 runs...")

        runner = MonteCarloRunner(
            sim_func=run_single_sim,
            config_generator=generate_mc_config,
            num_runs=1,
            max_workers=1,
        )
        full_results = runner.run()

        print(f"\n--- Monte Carlo Complete: Generated {len(full_results)} histories ---")

        # Save results
        save_data("3MTQ+1RW_trajectory_mc_100_1000s", full_results, out_dir=OUTPUT_DIR)

        # Plot comparison
        plot_comparison(full_results)
        create_close_all_button_window()
    else:
        # Load existing results
        results = load_data("papers/3MTQ+1RW/output_data/3MTQ+1RW_trajectory_mc_100_1000s")
        if isinstance(results, tuple):
            full_results = results[0]
        else:
            full_results = results
        plot_comparison(full_results)
        create_close_all_button_window()
