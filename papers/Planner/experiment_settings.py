#!/usr/bin/env python3
"""
Fast experiment harness for planner settings tuning.
Runs C++ planner (Plan_and_Track_LQR) on multiple seeds, measures:
  - Planning time
  - Trajectory quality: angle error over time, constraint violations
  - Stability: spikes, oscillations

Usage:
  python experiment_settings.py              # Run baseline
  python experiment_settings.py --seeds 5    # Run 5 seeds
  python experiment_settings.py --rw         # Include RW config
"""
import sys, os, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.CONOPS.goallist import GoalList
from ADCS.CONOPS.goals import ECI_Goal, No_Goal
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_helpers import normalize, rot_exp, rot_mat
from ADCS.controller.helpers.live_planner_viz import quat_to_rotation_matrix, compute_angle_error
from papers.Planner.mc_planner_settings import create_optimized_planner_settings


def make_config(seed, has_rw=True, tf=1000, dt_planning=1.0):
    rng = np.random.default_rng(seed=seed + 1000)
    q0 = normalize(rng.standard_normal(4))
    goal1 = normalize(rng.standard_normal(3))
    goal2 = rot_mat(rot_exp((np.pi/2) * normalize(rng.standard_normal(3)))) @ goal1
    goal3 = rot_mat(rot_exp((np.pi/2) * normalize(rng.standard_normal(3)))) @ goal2
    return {
        "seed": seed,
        "tf": tf,
        "dt_planning": dt_planning,
        "w0": normalize(rng.standard_normal(3)) * (rng.uniform(0.1, 1.0) * np.pi / 180.0),
        "q0": q0,
        "h0": rng.uniform(-0.0001, 0.0001, size=1) if has_rw else np.array([]),
        "goal1": goal1,
        "goal2": goal2,
        "goal3": goal3,
    }


def run_one(config, has_rw, tuning, settings_overrides=None, multi_goal=True):
    """Run one planning test, return metrics dict."""
    tf = config["tf"]
    dt_planning = config["dt_planning"]
    
    np.random.seed(config["seed"])
    sat = create_beavercube2_cubesat(estimated=False)
    if not has_rw:
        sat.rw_actuators = []
        
    rws = sat.rw_actuators
    x0 = np.concatenate([config["w0"], config["q0"], config["h0"]])
    for i, rw in enumerate(rws):
        rw.h = config["h0"][i]

    settings = create_optimized_planner_settings(
        sat, duration=tf, dt_planning=dt_planning, tuning=tuning, has_rw=has_rw
    )
    settings.verbosity = False
    
    # Apply overrides
    if settings_overrides:
        for key, val in settings_overrides.items():
            parts = key.split(".")
            obj = settings
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], val)

    controller = Plan_and_Track_LQR(est_sat=sat, planner_settings=settings)

    sec2cent = TimeConstants.sec2cent
    t0_j2000 = 0.22

    if multi_goal:
        goals = GoalList({
            t0_j2000: ECI_Goal(config["goal1"]),
            t0_j2000 + 350 * sec2cent: No_Goal(),
            t0_j2000 + 550 * sec2cent: ECI_Goal(config["goal2"]),
            t0_j2000 + 700 * sec2cent: No_Goal(),
            t0_j2000 + 900 * sec2cent: ECI_Goal(config["goal3"]),
        })
    else:
        goals = GoalList({t0_j2000: ECI_Goal(config["goal1"])})

    orb = create_random_circular_orbit(radius_km=7000.0, dt=dt_planning, tf=tf, use_J2=True, fast=True)
    orb.populate_environment(compute_B=True, compute_S=True)
    os0 = orb.get_os(t0_j2000)

    t_start = time.time()
    try:
        traj = controller.calculate_trajectory(
            t_start=t0_j2000, duration=tf, x_0=x0, os_0=os0, goals=goals, verbose=False
        )
        elapsed = time.time() - t_start
        traj_valid = True
    except Exception as e:
        elapsed = time.time() - t_start
        return {"valid": False, "error": str(e), "time_s": elapsed, "seed": config["seed"]}

    # Extract metrics from planned trajectory
    Xset = traj.states  # (n_state, N)
    Uset = traj.controls  # (n_control, N-1) - already reordered to python ordering
    times_sec = (traj.times - traj.times[0]) / sec2cent  # seconds

    N = Xset.shape[1]
    n_state = Xset.shape[0]
    
    BODY_BORESIGHT = np.array([0, 0, 1.0])
    omega_norms = np.linalg.norm(Xset[0:3, :], axis=0) * 180 / np.pi  # deg/s

    # Compute angle error per goal segment
    angle_errors = np.full(N, np.nan)
    if multi_goal:
        # Segments: [0,350)=goal1, [350,550)=no_goal, [550,700)=goal2, [700,900)=no_goal, [900,end]=goal3
        segments = [(0, 350, config["goal1"]), (550, 700, config["goal2"]), (900, tf, config["goal3"])]
    else:
        segments = [(0, tf, config["goal1"])]
    
    for t_start_seg, t_end_seg, goal_vec in segments:
        mask = (times_sec >= t_start_seg) & (times_sec < t_end_seg + 1)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        seg_Xset = Xset[:, idx]
        seg_errors = compute_angle_error(seg_Xset, goal_vec, BODY_BORESIGHT)
        angle_errors[idx] = seg_errors

    # Key metrics
    valid_errors = angle_errors[~np.isnan(angle_errors)]
    
    # Time at < 5° and < 1°
    dt_avg = np.mean(np.diff(times_sec))
    time_under_5deg = np.sum(valid_errors < 5.0) * dt_avg
    time_under_1deg = np.sum(valid_errors < 1.0) * dt_avg
    total_goal_time = len(valid_errors) * dt_avg
    
    # Max omega spike
    max_omega = np.max(omega_norms)
    
    # Control violations (check if controls exceed limits)
    # C++ reorders, so just check magnitudes
    max_control = np.max(np.abs(Uset)) if Uset.size > 0 else 0
    
    # Final error for each goal segment
    # Goal 1: around t=349
    final_errors = []
    if multi_goal:
        for t_target in [349, 699, 999]:
            idx = np.argmin(np.abs(times_sec - t_target))
            if not np.isnan(angle_errors[idx]):
                final_errors.append(angle_errors[idx])
    else:
        final_errors.append(angle_errors[-1])
    
    # Check for spikes: large sudden increases in angle error
    valid_mask = ~np.isnan(angle_errors)
    valid_err = angle_errors[valid_mask]
    if len(valid_err) > 10:
        diffs = np.diff(valid_err)
        max_spike = np.max(diffs)  # largest single-step increase
        n_spikes = np.sum(diffs > 5.0)  # jumps > 5°
    else:
        max_spike = 0
        n_spikes = 0

    # RW utilization
    if has_rw:
        # RW is last control channel 
        rw_usage = np.mean(np.abs(Uset[-1, :])) if Uset.shape[0] > 3 else 0
        mtq_usage = np.mean(np.abs(Uset[:3, :]).max(axis=0))
    else:
        rw_usage = 0
        mtq_usage = np.mean(np.abs(Uset[:3, :]).max(axis=0)) if Uset.shape[0] >= 3 else 0

    return {
        "valid": True,
        "seed": config["seed"],
        "time_s": elapsed,
        "pct_under_5deg": time_under_5deg / total_goal_time * 100,
        "pct_under_1deg": time_under_1deg / total_goal_time * 100,
        "max_omega_dps": max_omega,
        "max_spike_deg": max_spike,
        "n_spikes": n_spikes,
        "final_errors_deg": final_errors,
        "mean_final_err": np.mean(final_errors),
        "max_control": max_control,
        "rw_usage": rw_usage,
        "mtq_usage": mtq_usage,
    }


def run_experiment(seeds, has_rw, tuning, settings_overrides=None, multi_goal=True, label=""):
    """Run experiment across multiple seeds, print summary."""
    results = []
    for s in seeds:
        config = make_config(s, has_rw=has_rw)
        r = run_one(config, has_rw, tuning, settings_overrides, multi_goal)
        results.append(r)
        status = "OK" if r["valid"] else f"FAIL: {r.get('error','?')[:40]}"
        if r["valid"]:
            print(f"  seed {s:3d}: {r['time_s']:5.1f}s | <5°:{r['pct_under_5deg']:5.1f}% | <1°:{r['pct_under_1deg']:5.1f}% | "
                  f"final:{r['mean_final_err']:5.1f}° | ω_max:{r['max_omega_dps']:5.2f}°/s | "
                  f"spikes:{r['n_spikes']} | RW:{r['rw_usage']:.4f} | {status}")
        else:
            print(f"  seed {s:3d}: {r['time_s']:5.1f}s | {status}")

    valid = [r for r in results if r["valid"]]
    if not valid:
        print(f"\n{'='*60}\n{label}: ALL FAILED\n{'='*60}")
        return results

    avg_time = np.mean([r["time_s"] for r in valid])
    avg_5 = np.mean([r["pct_under_5deg"] for r in valid])
    avg_1 = np.mean([r["pct_under_1deg"] for r in valid])
    avg_final = np.mean([r["mean_final_err"] for r in valid])
    avg_spikes = np.mean([r["n_spikes"] for r in valid])
    avg_rw = np.mean([r["rw_usage"] for r in valid])
    max_omega = np.max([r["max_omega_dps"] for r in valid])
    n_valid = len(valid)

    print(f"\n{'='*60}")
    print(f"{label} ({n_valid}/{len(seeds)} valid)")
    print(f"  Time: {avg_time:.1f}s avg")
    print(f"  <5°:  {avg_5:.1f}% avg")
    print(f"  <1°:  {avg_1:.1f}% avg")
    print(f"  Final err: {avg_final:.1f}° avg")
    print(f"  Spikes: {avg_spikes:.1f} avg")
    print(f"  Max ω: {max_omega:.2f}°/s")
    print(f"  RW usage: {avg_rw:.4f}")
    print(f"{'='*60}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds to test")
    parser.add_argument("--rw", action="store_true", help="Include RW")
    parser.add_argument("--no-rw", action="store_true", help="MTQ only")
    parser.add_argument("--both", action="store_true", help="Test both configs")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--tuning", type=str, default="fast_slew", help="Tuning preset")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    
    configs_to_test = []
    if args.both:
        configs_to_test = [(True, "1RW"), (False, "0RW")]
    elif args.no_rw:
        configs_to_test = [(False, "0RW")]
    else:
        configs_to_test = [(True, "1RW")]
    
    if args.sweep:
        # Parameter sweep
        has_rw = True
        experiments = {
            # Each: (tuning, overrides_dict, label)
            "A: baseline fast_slew": ("fast_slew", {}, "baseline"),
            "B: fast_slew, angvel*10 not *100": ("fast_slew", {
                "_custom": "angvel_x10",
            }, "angvel_x10"),
            "C: balanced (no fast_slew)": ("balanced", {}, "balanced"),
            "D: balanced + loose z_count": ("balanced", {
                "pass2.convergence.z_count_lim": 5,
            }, "balanced_z5"),
            "E: none (raw auto-scale)": ("none", {}, "raw_autoscale"),
        }
        for name, (tuning, overrides, label) in experiments.items():
            print(f"\n{'#'*60}")
            print(f"# {name}")
            print(f"{'#'*60}")
            
            # Handle custom overrides that need code changes
            if overrides.pop("_custom", None) == "angvel_x10":
                # Temporarily patch fast_slew to use *10 instead of *100
                import papers.Planner.mc_planner_settings as mps
                orig_fn = mps.apply_fast_slew_tuning
                def patched_fast_slew(settings, verbose=False):
                    orig_fn(settings, verbose)
                    # Undo the 100x and apply 10x instead
                    settings.cost_main.ang_vel /= 10  # was *100, now *10
                    settings.cost_main.ang_vel_N = settings.cost_main.ang_vel
                mps.apply_fast_slew_tuning = patched_fast_slew
                run_experiment(seeds, has_rw, tuning, label=name)
                mps.apply_fast_slew_tuning = orig_fn
            else:
                run_experiment(seeds, has_rw, tuning, settings_overrides=overrides if overrides else None, label=name)
    else:
        for has_rw, label in configs_to_test:
            print(f"\n{'#'*60}")
            print(f"# {label} - {args.tuning} tuning")
            print(f"{'#'*60}")
            run_experiment(seeds, has_rw, args.tuning, label=f"{label} {args.tuning}")
