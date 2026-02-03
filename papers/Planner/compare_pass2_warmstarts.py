"""
Compare pass2 warm-start discretization methods against a reference trajectory.

Generates a reference pass2 trajectory using Python ALILQR, then builds
fine-grid trajectories using:
  - FOH control interpolation (propagated)
  - SLERP state interpolation + FOH controls (propagated)
  - SLERP state interpolation + LSQ control reconstruction (propagated)

Reports how closely each propagated trajectory matches the reference.
"""
import argparse
import sys
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, "/home/pmckeen/Generalized_ADCS")
sys.path.insert(0, "/home/pmckeen/Generalized_ADCS/papers/Planner")

from ADCS.CONOPS.goals import ECI_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_python_alilqr import Plan_and_Track_PythonALILQR
from ADCS.controller.helpers.mtq_warm_start import (
    interpolate_trajectory_to_finer_grid,
    solve_controls_from_trajectory_regularized,
)
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import normalize, rot_mat
from mc_planner_settings import create_optimized_planner_settings
from ADCS.orbits.universal_constants import TimeConstants


BODY_BORESIGHT = np.array([0, 1, 0])


def quat_angle_error_deg(q_ref: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    """Quaternion angle error in degrees for arrays shaped (4, N)."""
    dot = np.sum(q_ref * q_est, axis=0)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def compute_metrics(X_ref: np.ndarray, X_cmp: np.ndarray) -> Dict[str, float]:
    n = min(X_ref.shape[1], X_cmp.shape[1])
    Xr = X_ref[:, :n]
    Xc = X_cmp[:, :n]

    w_err = Xc[0:3, :] - Xr[0:3, :]
    w_err_norm = np.linalg.norm(w_err, axis=0)

    q_ref = Xr[3:7, :]
    q_cmp = Xc[3:7, :]
    ang_err = quat_angle_error_deg(q_ref, q_cmp)

    return {
        "w_err_rms": float(np.sqrt(np.mean(w_err_norm ** 2))),
        "w_err_max": float(np.max(w_err_norm)),
        "q_err_rms_deg": float(np.sqrt(np.mean(ang_err ** 2))),
        "q_err_mean_deg": float(np.mean(ang_err)),
        "q_err_max_deg": float(np.max(ang_err)),
    }


def describe_error_profile(angles_deg: np.ndarray, times: np.ndarray) -> str:
    """Summarize rough shape of angular error over time."""
    if angles_deg.size == 0:
        return "no data"
    # Basic stats
    start = float(angles_deg[0])
    end = float(angles_deg[-1])
    max_val = float(np.max(angles_deg))
    max_idx = int(np.argmax(angles_deg))
    max_t = float(times[max_idx]) if times is not None and times.size > max_idx else float(max_idx)

    # Early vs late trend
    n = angles_deg.size
    mid = n // 2
    first_mean = float(np.mean(angles_deg[:max(1, mid)]))
    second_mean = float(np.mean(angles_deg[mid:]))
    trend = "decreasing" if second_mean < first_mean else "increasing"
    if abs(second_mean - first_mean) < 1e-6:
        trend = "flat"

    return (
        f"start={start:.2f}deg end={end:.2f}deg "
        f"max={max_val:.2f}deg at t={max_t:.1f}s "
        f"trend={trend} (mean_first={first_mean:.2f}, mean_second={second_mean:.2f})"
    )


def angular_error_series(
    X: np.ndarray,
    goal_vecs: np.ndarray,
    body_boresight: np.ndarray,
) -> np.ndarray:
    """
    Compute angular error series.
    - If goal_vecs is (3, N): boresight alignment error.
    - If goal_vecs is (4, N): quaternion error vs goal quaternion.
    """
    n = X.shape[1]
    if goal_vecs.shape[1] != n:
        n = min(n, goal_vecs.shape[1])
        X = X[:, :n]
        goal_vecs = goal_vecs[:, :n]

    if goal_vecs.shape[0] == 4:
        q = X[3:7, :]
        qg = goal_vecs
        dot = np.sum(q * qg, axis=0)
        dot = np.clip(np.abs(dot), -1.0, 1.0)
        return np.degrees(2.0 * np.arccos(dot))

    # Vector goal case
    v_b = body_boresight / np.linalg.norm(body_boresight)
    q = X[3:7, :].T  # (N,4)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x*y - z*w)
    R[:, 0, 2] = 2 * (x*z + y*w)
    R[:, 1, 0] = 2 * (x*y + z*w)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y*z - x*w)
    R[:, 2, 0] = 2 * (x*z - y*w)
    R[:, 2, 1] = 2 * (y*z + x*w)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    v_b_eci = np.einsum("nij,j->ni", R, v_b)
    v_b_eci /= np.linalg.norm(v_b_eci, axis=1, keepdims=True)
    v_g = goal_vecs.T
    v_g /= np.linalg.norm(v_g, axis=1, keepdims=True)
    dot = np.sum(v_b_eci * v_g, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def summarize_pass1_behavior(
    X: np.ndarray,
    times: np.ndarray,
    goal_vecs: np.ndarray,
    body_boresight: np.ndarray,
) -> str:
    err = angular_error_series(X, goal_vecs, body_boresight)
    w_norm = np.linalg.norm(X[0:3, :], axis=0)
    w_deg_s = np.degrees(w_norm)

    # Spike detection: large step increases
    if err.size > 1:
        d_err = np.diff(err)
        spike_count = int(np.sum(d_err > 10.0))
        max_jump = float(np.max(d_err))
    else:
        spike_count = 0
        max_jump = 0.0

    # Settling: time to stay below 5 deg
    settle_5 = np.nan
    for i in range(err.size):
        if np.all(err[i:] < 5.0):
            settle_5 = float(times[i])
            break

    # Final rate to indicate spin
    final_rate = float(w_deg_s[-1]) if w_deg_s.size else float("nan")
    max_rate = float(np.max(w_deg_s)) if w_deg_s.size else float("nan")

    return (
        f"{describe_error_profile(err, times)} | "
        f"spikes>10deg={spike_count} (max_jump={max_jump:.2f}deg) | "
        f"max_rate={max_rate:.2f}deg/s final_rate={final_rate:.2f}deg/s | "
        f"settle_5deg={settle_5:.1f}s"
    )


def print_metrics(label: str, metrics: Dict[str, float]) -> None:
    print(
        f"{label}: q_err_rms={metrics['q_err_rms_deg']:.3f}deg "
        f"q_err_mean={metrics['q_err_mean_deg']:.3f}deg "
        f"q_err_max={metrics['q_err_max_deg']:.3f}deg "
        f"w_err_rms={metrics['w_err_rms']:.5f}rad/s "
        f"w_err_max={metrics['w_err_max']:.5f}rad/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pass2 warm-start methods")
    parser.add_argument("--tf", type=float, default=60.0, help="Duration [s]")
    parser.add_argument("--radius-km", type=float, default=7000.0, help="Orbit radius [km]")
    parser.add_argument("--seed", type=int, default=0, help="Base seed (rng uses seed+1000)")
    parser.add_argument(
        "--tuning",
        type=str,
        default="aggressive",
        help="Planner tuning preset (e.g., aggressive, balanced, fast_slew, anti_spin, smooth, none)",
    )
    parser.add_argument("--pass2-iters", type=int, default=None, help="Override pass2 max inner iters")
    parser.add_argument("--save", type=str, default=None, help="Save reference to npz")
    parser.add_argument("--lsq-lambda", type=float, default=0.0, help="LSQ regularization lambda")
    parser.add_argument("--lsq-use-prior", action="store_true", help="Use FOH controls as LSQ prior")
    args = parser.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(seed=args.seed + 1000)

    q0 = normalize(rng.standard_normal(4))
    R0 = rot_mat(q0)
    initial_boresight_eci = R0 @ BODY_BORESIGHT
    goal_eci_vec = -initial_boresight_eci

    w0 = normalize(rng.standard_normal(3)) * (rng.uniform(0.1, 1.0) * np.pi / 180.0)
    h0 = rng.uniform(-0.0001, 0.0001, size=1)
    x0 = np.concatenate([w0, q0, h0])

    orb = create_random_circular_orbit(radius_km=args.radius_km, dt=1, tf=args.tf, use_J2=True, fast=True)
    orb.populate_environment(compute_B=True, compute_S=True)

    real_sat = create_beavercube2_cubesat(estimated=False)
    for i, rw in enumerate(real_sat.rw_actuators):
        rw.h = h0[i]

    planner_settings = create_optimized_planner_settings(
        real_sat, duration=args.tf, dt_planning=1, tuning=args.tuning
    )
    if args.pass2_iters is not None:
        planner_settings.pass2.convergence.max_inner_iter = int(args.pass2_iters)

    controller = Plan_and_Track_PythonALILQR(
        est_sat=real_sat,
        planner_settings=planner_settings,
        use_v2=True,
        verbose=False,
    )

    goals = GoalList({0.22: ECI_Goal(goal_eci_vec)})
    os0 = orb.get_os(0.22)

    try:
        traj_ref = controller.calculate_trajectory(
            t_start=0.22,
            duration=args.tf,
            x_0=x0,
            os_0=os0,
            goals=goals,
            verbose=False,
            collect_all_iterations=False,
            skip_pass2=False,
        )
    except RuntimeError as e:
        if planner_settings.bdot_on != 0:
            print(f"WARNING: calculate_trajectory failed with bdot_on={planner_settings.bdot_on}; retrying with bdot_on=0")
            planner_settings.bdot_on = 0
            controller = Plan_and_Track_PythonALILQR(
                est_sat=real_sat,
                planner_settings=planner_settings,
                use_v2=True,
                verbose=False,
            )
            traj_ref = controller.calculate_trajectory(
                t_start=0.22,
                duration=args.tf,
                x_0=x0,
                os_0=os0,
                goals=goals,
                verbose=False,
                collect_all_iterations=False,
                skip_pass2=False,
            )
        else:
            raise e

    times_ref = traj_ref.times
    X_ref = traj_ref.states.T if traj_ref.states.shape[0] == times_ref.shape[0] else traj_ref.states
    U_ref = traj_ref.controls.T if traj_ref.controls.shape[0] == times_ref.shape[0] else traj_ref.controls

    if args.save:
        np.savez(
            args.save,
            X_ref=X_ref,
            U_ref=U_ref,
            times_ref=times_ref,
            seed=args.seed,
            tf=args.tf,
        )
        print(f"Saved reference to {args.save}")

    # Pass1 coarse result
    result1 = controller.pass1_result
    X_coarse = result1.Xset
    U_coarse = result1.Uset

    dt_coarse = planner_settings.dt_tp
    dt_fine = planner_settings.dt_tvlqr
    N_fine = int(np.ceil(args.tf / dt_fine)) + 1

    # Prepare fine vecs for propagation
    t_end = 0.22 + args.tf * TimeConstants.sec2cent
    vecsPy_fine = controller._propagate_environment(os0, 0.22, t_end, dt_fine, N_fine, goals)
    initial_result_2 = controller.planner.prepareForAlilqr(
        vecsPy_fine, dt_fine, 0.22, t_end, x0.copy(), 0
    )
    _, vecs_dt_fine, _ = initial_result_2

    # Pass1 goal vectors for error profile
    N_coarse = int(np.ceil(args.tf / dt_coarse)) + 1
    vecsPy_coarse = controller._propagate_environment(os0, 0.22, t_end, dt_coarse, N_coarse, goals)
    goal_vecs_coarse = vecsPy_coarse[6]

    # Pass1 status
    print("\n=== Pass1 Convergence ===")
    print(
        f"final_cost={result1.final_cost:.6e} "
        f"final_cmax={result1.final_cmax:.6e} "
        f"final_grad={result1.final_grad:.6e} "
        f"iters={result1.total_inner_iters} "
        f"break_reason={result1.break_reason}"
    )
    t_pass1 = np.linspace(0.0, args.tf, X_coarse.shape[1])
    print("pass1 angular error profile:", summarize_pass1_behavior(X_coarse, t_pass1, goal_vecs_coarse, BODY_BORESIGHT))

    # FOH control interpolation
    t_coarse = np.linspace(0.0, args.tf, U_coarse.shape[1])
    t_fine = np.linspace(0.0, args.tf, N_fine)
    U_fine_foh = np.zeros((U_coarse.shape[0], N_fine))
    for i in range(U_coarse.shape[0]):
        U_fine_foh[i, :] = np.interp(t_fine, t_coarse, U_coarse[i, :])

    # FOH propagation
    traj_foh = controller.planner.generateInitialTrajectory(
        dt_fine, X_coarse[:, 0].copy(), U_fine_foh, vecs_dt_fine
    )
    X_foh = traj_foh[0]

    # SLERP state interpolation (pass1 output on fine grid)
    X_slerp = interpolate_trajectory_to_finer_grid(
        X_coarse, dt_coarse, dt_fine, args.tf, use_slerp=True
    )

    # ZOH control interpolation (baseline)
    idx_fine = np.clip((t_fine / dt_coarse).astype(int), 0, U_coarse.shape[1] - 1)
    U_fine_zoh = U_coarse[:, idx_fine]

    # ZOH propagation (baseline)
    traj_zoh = controller.planner.generateInitialTrajectory(
        dt_fine, X_coarse[:, 0].copy(), U_fine_zoh, vecs_dt_fine
    )
    X_zoh = traj_zoh[0]

    # SLERP propagation using FOH controls (dynamics-consistent)
    traj_slerp_prop = controller.planner.generateInitialTrajectory(
        dt_fine, X_slerp[:, 0].copy(), U_fine_foh, vecs_dt_fine
    )
    X_slerp_prop = traj_slerp_prop[0]

    # LSQ reconstruction from SLERP states (lambda=0, no clamping)
    B_eci = vecs_dt_fine[3]
    J = real_sat.J_COM
    from ADCS.satellite_hardware.actuators.reaction_wheel import RW
    rw_axes = [act.axis for act in real_sat.actuators if isinstance(act, RW)]
    rw_axes = np.array(rw_axes) if rw_axes else None

    U_lsq = solve_controls_from_trajectory_regularized(
        X_slerp, B_eci, dt_fine, J, rw_axes,
        u_prior=(U_fine_foh if args.lsq_use_prior else None),
        reg_lambda=args.lsq_lambda,
        m_max=None, rw_torq_max=None
    )

    traj_lsq = controller.planner.generateInitialTrajectory(
        dt_fine, X_slerp[:, 0].copy(), U_lsq, vecs_dt_fine
    )
    X_lsq = traj_lsq[0]

    # Initial pass1 trajectory (before optimization) upsampled and propagated
    initial_traj_1, _, _ = controller.planner.prepareForAlilqr(
        vecsPy_coarse, dt_coarse, 0.22, t_end, x0.copy(), int(planner_settings.bdot_on)
    )
    X_init = initial_traj_1[0]
    U_init = initial_traj_1[1]
    t_init = np.linspace(0.0, args.tf, U_init.shape[1])
    U_fine_init = np.zeros((U_init.shape[0], N_fine))
    for i in range(U_init.shape[0]):
        U_fine_init[i, :] = np.interp(t_fine, t_init, U_init[i, :])
    traj_init = controller.planner.generateInitialTrajectory(
        dt_fine, X_init[:, 0].copy(), U_fine_init, vecs_dt_fine
    )
    X_init_fine = traj_init[0]

    print("\n=== Warm-start Propagation Match to Reference (Pass2 Result) ===")
    print_metrics("PASS1_INIT_FOH", compute_metrics(X_ref, X_init_fine))
    print_metrics("PASS1_OPT_FOH", compute_metrics(X_ref, X_foh))
    print_metrics("ZOH", compute_metrics(X_ref, X_zoh))
    print_metrics("SLERP+FOH", compute_metrics(X_ref, X_slerp_prop))
    print_metrics("SLERP+LSQ", compute_metrics(X_ref, X_lsq))

    print("\n=== Pass2 Input Match to Pass1 Output (SLERP-upsampled) ===")
    print_metrics("PASS1_OPT_SLERP", compute_metrics(X_slerp, X_slerp))
    print_metrics("FOH", compute_metrics(X_slerp, X_foh))
    print_metrics("ZOH", compute_metrics(X_slerp, X_zoh))
    print_metrics("SLERP+FOH", compute_metrics(X_slerp, X_slerp_prop))
    print_metrics("SLERP+LSQ", compute_metrics(X_slerp, X_lsq))


if __name__ == "__main__":
    main()
