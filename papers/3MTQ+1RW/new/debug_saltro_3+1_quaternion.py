import os
import sys
import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
import ADCS as ADCS
from ADCS.CONOPS.goallist import GoalList
from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.controller.neo_planner.NEO_planner_settings import PlannerSettings
from ADCS.helpers.math_helpers import normalize
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import TimeConstants


def _tracking_error_stats_deg(
    state_hist: np.ndarray,
    target_hist: np.ndarray,
    body_boresight: np.ndarray,
) -> Dict[str, float]:
    err = _tracking_error_series_deg(state_hist, target_hist, body_boresight)
    valid = err[np.isfinite(err)]
    if valid.size == 0:
        return {"final": np.nan, "mean": np.nan, "max": np.nan}

    return {
        "final": float(valid[-1]),
        "mean": float(np.mean(valid)),
        "max": float(np.max(valid)),
    }
 
def _tracking_error_series_deg(
    state_hist: np.ndarray,
    target_hist: np.ndarray,
    body_boresight: np.ndarray,
) -> np.ndarray:
    x = np.asarray(state_hist, dtype=np.float64)
    t = np.asarray(target_hist, dtype=np.float64)
    _ = np.asarray(body_boresight, dtype=np.float64).reshape(3)

    n = min(x.shape[0], t.shape[0])
    err = np.full(n, np.nan, dtype=np.float64)

    for k in range(n):
        row = t[k, :]
        if row.size < 4:
            continue
        if np.isnan(row[0]):
            continue

        q = x[k, 3:7]
        qg = row[:4]
        nq = np.linalg.norm(q)
        ng = np.linalg.norm(qg)
        if nq < 1e-12 or ng < 1e-12:
            continue
        q = q / nq
        qg = qg / ng

        q_inv = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)
        w0, x0, y0, z0 = q_inv
        w1, x1, y1, z1 = qg
        q_err = np.array(
            [
                w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            ],
            dtype=np.float64,
        )
        w = float(np.clip(abs(q_err[0]), -1.0, 1.0))
        err[k] = np.rad2deg(2.0 * np.arccos(w))
    return err


def _configure_like_saltro_3_1_python_debug_dt10(settings: PlannerSettings) -> None:
    settings.init_traj.initcontroller = 1

    p0 = settings.passes[0]
    p0.dt = 10.0
    p0.ilqr.cost_tol = 1e-3
    p0.ilqr.max_iters = 20

    p0.aug_lag.max_outer_iters = 10
    p0.aug_lag.constraint_tol = 1e-3

    cost = p0.cost
    cost.angle = 1e2
    cost.ang_vel = 1e1
    cost.ang_vel_mag = 0.0
    cost.ang_vel_err_dir = 0.0
    cost.control_mult = 1.0
    cost.mtq_control_weight = 1e-1
    cost.rw_control_weight = 1.0
    cost.magic_control_weight = 0.0
    cost.rw_AM_weight = 0.0
    cost.rw_stic_weight = 0.0
    cost.RWh_max_mult = 0.0
    cost.RWh_stiction_mult = 0.0
    cost.RWh_ok_mult = 0.0
    cost.angle_N = 1e2
    cost.ang_vel_N = 1e1
    cost.ang_vel_mag_N = 0.0
    cost.ang_vel_err_dir_N = 0.0
    cost.ang_cost_func_type = 3
    cost.use_cost_hess = 1

    settings.disturbances.plan_for_aero = 0
    settings.disturbances.plan_for_gg = 0
    settings.disturbances.plan_for_srp = 0
    settings.disturbances.plan_for_prop = 0
    settings.disturbances.plan_for_gendist = 0
    settings.disturbances.plan_for_resdipole = 0

    p0.reg.reg_init = 1e-6
    p0.reg.reg_max = 1e10
    p0.reg.reg_scale = 10.0
    p0.reg.use_dynamics_hess = 0
    p0.reg.use_constraint_hess = 0

    p0.linesearch.max_iters = 24
    p0.linesearch.beta1 = 1e-10
    p0.linesearch.beta2 = 5000.0


def _build_target_row(goal_vec: np.ndarray) -> np.ndarray:
    row = np.asarray(goal_vec, dtype=np.float64).reshape(-1)
    if row.size >= 4:
        return row[:4]
    out = np.full(4, np.nan, dtype=np.float64)
    out[:row.size] = row
    return out


def _extract_nominal_optimizer_output(traj) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if getattr(traj, "_is_row_major", False):
        state_hist = np.asarray(traj.states, dtype=np.float64)
    else:
        state_hist = np.asarray(traj.states, dtype=np.float64).T

    duration_s = float(np.asarray(traj.times, dtype=np.float64)[-1] - np.asarray(traj.times, dtype=np.float64)[0]) / TimeConstants.sec2cent
    time_s = np.linspace(0.0, duration_s, state_hist.shape[0], dtype=np.float64)

    if traj.controls.shape[0] == time_s.size or traj.controls.shape[0] == time_s.size - 1:
        control_hist = np.asarray(traj.controls, dtype=np.float64)
    else:
        control_hist = np.asarray(traj.controls, dtype=np.float64).T

    if control_hist.shape[0] != time_s.size:
        last = control_hist[-1:, :]
        control_hist = np.vstack([control_hist, last])

    return time_s, state_hist, control_hist


def _build_nominal_open_loop(
    traj,
    goals: GoalList,
    os0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_s, state_hist, control_hist = _extract_nominal_optimizer_output(traj)
    jtime = float(os0.J2000) + time_s * TimeConstants.sec2cent
    control_time_s = np.linspace(time_s[0], time_s[-1], control_hist.shape[0], dtype=np.float64)

    orb_os0 = os0.copy()
    orb_os0.J2000 = float(jtime[0])
    orb = Orbit(os0=orb_os0, end_time=float(jtime[-1]), dt=max(1.0, float(time_s[1] - time_s[0]) if time_s.size > 1 else 1.0), use_J2=True, fast=False)

    target_hist = np.zeros((time_s.size, 4), dtype=np.float64)
    for k, t in enumerate(jtime):
        os_k = orb.get_os(J2000=float(t))
        target, _ = goals.to_ref(float(t), os_k, time_units="centuries")
        target_hist[k, :] = _build_target_row(target)

    return time_s, control_time_s, state_hist, control_hist, target_hist


def _plot_rollout(tag: str, time_s: np.ndarray, control_time_s: np.ndarray, state_hist: np.ndarray, control_hist: np.ndarray, target_hist: np.ndarray, body_boresight: np.ndarray) -> None:
    err_deg = _tracking_error_series_deg(state_hist, target_hist, body_boresight)

    fig, axes = plt.subplots(2, 2, num=tag, figsize=(13, 8), clear=True)
    fig.suptitle(tag)

    ax = axes[0, 0]
    ax.plot(time_s, err_deg, lw=1.4)
    ax.set_title("Angle Error")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Error [deg]")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if control_hist.ndim == 2 and control_hist.shape[1] > 0:
        x_ctrl = control_time_s
        for i in range(control_hist.shape[1]):
            ax.plot(x_ctrl, control_hist[:, i], lw=1.2, label=f"u{i}")
        ax.legend(loc="best", fontsize=8)
    ax.set_title("Actuator Usage")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Control")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    rw_hist = state_hist[:, 7:] if state_hist.shape[1] > 7 else np.zeros((state_hist.shape[0], 0))
    if rw_hist.shape[1] > 0:
        for i in range(rw_hist.shape[1]):
            ax.plot(time_s, rw_hist[:, i], lw=1.2, label=f"h{i}")
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No RW momentum states", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Reaction Wheel Momentum")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("h [Nms]")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    w = state_hist[:, 0:3]
    for i, lab in enumerate(("wx", "wy", "wz")):
        ax.plot(time_s, w[:, i], lw=1.2, label=lab)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Angular Velocity")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("rad/s")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])


def run_saltro_3p1_quaternion(
    tf: float = 1000.0,
    dt: float = 1.0,
    plot: bool = True,
    verbose: bool = True,
):
    real_sat = ADCS.satellite_factory.create_beavercube2_cubesat(estimated=False)

    x_0 = np.array([0.01, 0.01, 0.01] + [1.0, 0.0, 0.0, 0.0] + [0.0], dtype=np.float64)
    os0 = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]),
    )

    q_goal = np.array([np.sqrt(2.0) / 2.0, 0.0, 0.0, np.sqrt(2.0) / 2.0], dtype=np.float64)
    goal = Fixed_Attitude_Goal(q_goal)
    goals = GoalList({os0.J2000: goal})

    planner_settings = PlannerSettings(est_sat=real_sat)
    _configure_like_saltro_3_1_python_debug_dt10(planner_settings)
    controller = ADCS.controller.SALTRO(est_sat=real_sat, planner_settings=planner_settings)

    t0 = time.perf_counter()
    traj = controller.calculate_trajectory(
        t_start=os0.J2000,
        duration=tf,
        x_0=x_0,
        os_0=os0,
        goals=goals,
        verbose=verbose,
    )
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    time_ol, control_time_ol, x_ol, u_ol, target_ol = _build_nominal_open_loop(
        traj=traj,
        goals=goals,
        os0=os0,
    )
    t3 = time.perf_counter()

    controller.set_active_trajectory(traj)
    t4 = time.perf_counter()
    results = ADCS.simulate(
        x=x_0,
        satellite=real_sat,
        controller=controller,
        goal=goal,
        os0=os0,
        dt=dt,
        tf=tf,
    )
    t5 = time.perf_counter()

    run = results.first()
    x_cl = np.asarray(run.state_hist, dtype=np.float64)
    u_cl = np.asarray(run.control_hist, dtype=np.float64)
    target_cl = np.asarray(run.target_hist, dtype=np.float64)
    time_cl = np.asarray(run.time_s, dtype=np.float64)

    body_boresight = np.asarray(real_sat.get_boresight(), dtype=np.float64)

    open_stats = _tracking_error_stats_deg(x_ol, target_ol, body_boresight)
    closed_stats = _tracking_error_stats_deg(x_cl, target_cl, body_boresight)

    timing = {
        "planning_s": float(t1 - t0),
        "open_loop_rollout_s": float(t3 - t2),
        "closed_loop_sim_s": float(t5 - t4),
        "total_s": float((t1 - t0) + (t3 - t2) + (t5 - t4)),
    }

    if verbose:
        print("\n==== SALTRO 3+1 QUATERNION ====")
        print(f"Planning [s]       : {timing['planning_s']:.3f}")
        print(f"Open-loop sim [s]  : {timing['open_loop_rollout_s']:.3f}")
        print(f"Closed-loop sim [s]: {timing['closed_loop_sim_s']:.3f}")
        print(f"Open-loop final err [deg] : {open_stats['final']:.4f}")
        print(f"Closed-loop final err [deg]: {closed_stats['final']:.4f}")
        print("================================\n")

    if plot:
        _plot_rollout(
            tag="open-loop",
            time_s=time_ol,
            control_time_s=control_time_ol,
            state_hist=x_ol,
            control_hist=u_ol,
            target_hist=target_ol,
            body_boresight=body_boresight,
        )
        _plot_rollout(
            tag="closed-loop",
            time_s=time_cl,
            control_time_s=time_cl,
            state_hist=x_cl,
            control_hist=u_cl,
            target_hist=target_cl,
            body_boresight=body_boresight,
        )
        plt.show()

    return {
        "results": results,
        "open_loop": {"time_s": time_ol, "state_hist": x_ol, "control_hist": u_ol, "target_hist": target_ol, "tracking": open_stats},
        "closed_loop": {"time_s": time_cl, "state_hist": x_cl, "control_hist": u_cl, "target_hist": target_cl, "tracking": closed_stats},
        "timing": timing,
    }


if __name__ == "__main__":
    run_saltro_3p1_quaternion(
        tf=1000.0,
        dt=1.0,
        plot=True,
        verbose=True,
    )
