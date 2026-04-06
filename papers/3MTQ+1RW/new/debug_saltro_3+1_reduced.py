import os
import sys
import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
import ADCS as ADCS
from ADCS.CONOPS.goallist import GoalList
from ADCS.CONOPS.goals import ECI_Goal
from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.controller.neo_planner.NEO_planner_settings import PlannerSettings
from ADCS.helpers.plotting.plot_controller import plot_target_tracking


def _rot_mat(q: np.ndarray) -> np.ndarray:
    """Quaternion (q0,q1,q2,q3) to body-to-inertial rotation matrix."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    q = q / n
    q0, q1, q2, q3 = q

    return np.array(
        [
            [
                1.0 - 2.0 * (q2 * q2 + q3 * q3),
                2.0 * (q1 * q2 - q0 * q3),
                2.0 * (q1 * q3 + q0 * q2),
            ],
            [
                2.0 * (q1 * q2 + q0 * q3),
                1.0 - 2.0 * (q1 * q1 + q3 * q3),
                2.0 * (q2 * q3 - q0 * q1),
            ],
            [
                2.0 * (q1 * q3 - q0 * q2),
                2.0 * (q2 * q3 + q0 * q1),
                1.0 - 2.0 * (q1 * q1 + q2 * q2),
            ],
        ],
        dtype=np.float64,
    )


def _tracking_error_stats_deg(
    state_hist: np.ndarray,
    target_hist: np.ndarray,
    body_boresight: np.ndarray,
) -> Dict[str, float]:
    """Compute tracking error statistics in degrees for vector/quaternion targets."""
    x = np.asarray(state_hist, dtype=np.float64)
    t = np.asarray(target_hist, dtype=np.float64)
    bore_b = np.asarray(body_boresight, dtype=np.float64).reshape(3)
    bore_b /= max(np.linalg.norm(bore_b), 1e-12)

    n = min(x.shape[0], t.shape[0])
    err = np.full(n, np.nan, dtype=np.float64)

    for k in range(n):
        row = t[k, :]
        if row.size < 4:
            continue

        q = x[k, 3:7]

        # Vector-goal rows are encoded as [nan, gx, gy, gz].
        if np.isnan(row[0]):
            r_b2i = _rot_mat(q)
            bore_i = r_b2i @ bore_b
            nb = np.linalg.norm(bore_i)
            if nb < 1e-12:
                continue
            bore_i /= nb

            goal_i = row[1:4]
            ng = np.linalg.norm(goal_i)
            if ng < 1e-12 or not np.all(np.isfinite(goal_i)):
                continue
            goal_i /= ng

            dot = np.clip(np.dot(bore_i, goal_i), -1.0, 1.0)
            err[k] = np.rad2deg(np.arccos(dot))
        else:
            # Quaternion-goal row: [q0,q1,q2,q3].
            q_curr = np.asarray(q, dtype=np.float64)
            q_goal = np.asarray(row[:4], dtype=np.float64)
            nq = np.linalg.norm(q_curr)
            ng = np.linalg.norm(q_goal)
            if nq < 1e-12 or ng < 1e-12:
                continue
            q_curr = q_curr / nq
            q_goal = q_goal / ng

            q_inv = np.array([q_curr[0], -q_curr[1], -q_curr[2], -q_curr[3]], dtype=np.float64)
            w0, x0, y0, z0 = q_inv
            w1, x1, y1, z1 = q_goal
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

    valid = err[np.isfinite(err)]
    if valid.size == 0:
        return {"final": np.nan, "mean": np.nan, "max": np.nan}

    return {
        "final": float(valid[-1]),
        "mean": float(np.mean(valid)),
        "max": float(np.max(valid)),
    }


def _configure_like_saltro_3_1_debug(settings: PlannerSettings) -> None:
    """Match SALTRO tests/debug/optimizer/alilqr_cpp/debug_3_1_slew90_dt10.py settings."""
    settings.init_traj.initcontroller = 2

    p0 = settings.passes[0]
    p0.dt = 5.0
    p0.ilqr.cost_tol = 1e-5
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


def run_saltro_3p1_reduced(
    tf: float = 1000.0,
    dt: float = 1.0,
    plot: bool = True,
    verbose: bool = True,
    goal_mode: str = "quat_slew90",
    use_saltro_3_1_debug_settings: bool = True,
) -> Tuple[object, Dict[str, float], Dict[str, float]]:
    real_sat = ADCS.satellite_factory.create_beavercube2_cubesat(estimated=False)
    x_0 = np.array([0.0, 0.0, 0.0] + [1, 0, 0, 0] + [0.0], dtype=np.float64)

    # Use default NEO planner settings as requested (no manual tuning overrides).
    planner_settings = PlannerSettings(est_sat=real_sat)
    if use_saltro_3_1_debug_settings:
        _configure_like_saltro_3_1_debug(planner_settings)

    controller = ADCS.controller.SALTRO(est_sat=real_sat, planner_settings=planner_settings)

    os0 = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(),
        J2000=0.22,
        # SALTRO validates LEO orbital bounds; this circular seed matches SALTRO debug usage.
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]),
    )
    if goal_mode == "eci_vector":
        goal = ECI_Goal(np.array([0.0, 0.0, -1.0]))
    elif goal_mode == "quat_slew90":
        q_goal = np.array([np.sqrt(2.0) / 2.0, 0.0, 0.0, np.sqrt(2.0) / 2.0], dtype=np.float64)
        goal = Fixed_Attitude_Goal(q_goal)
    else:
        raise ValueError(f"Unknown goal_mode={goal_mode!r}, expected 'eci_vector' or 'quat_slew90'")

    t0 = time.perf_counter()
    traj = controller.calculate_trajectory(
        t_start=os0.J2000,
        duration=tf,
        x_0=x_0,
        os_0=os0,
        goals=GoalList({os0.J2000: goal}),
        verbose=verbose,
    )

    t1 = time.perf_counter()

    controller.set_active_trajectory(traj)

    t2 = time.perf_counter()
    results = ADCS.simulate(
        x=x_0,
        satellite=real_sat,
        controller=controller,
        goal=goal,
        os0=os0,
        dt=dt,
        tf=tf,
    )
    t3 = time.perf_counter()

    run = results.first()
    state_hist = np.asarray(run.state_hist, dtype=np.float64)
    target_hist = np.asarray(run.target_hist, dtype=np.float64)
    time_s = np.asarray(run.time_s, dtype=np.float64)
    control_hist = np.asarray(run.control_hist, dtype=np.float64)

    body_boresight = np.asarray(real_sat.get_boresight(), dtype=np.float64)
    tracking_stats = _tracking_error_stats_deg(
        state_hist=state_hist,
        target_hist=target_hist,
        body_boresight=body_boresight,
    )

    timing_stats = {
        "planning_s": float(t1 - t0),
        "simulation_s": float(t3 - t2),
        "total_s": float((t1 - t0) + (t3 - t2)),
    }

    if verbose:
        print("\n========== SALTRO 3+1 REDUCED ==========")
        print(f"Planning wall time [s]: {timing_stats['planning_s']:.3f}")
        print(f"Simulation time [s]   : {timing_stats['simulation_s']:.3f}")
        print(f"Total time [s]        : {timing_stats['total_s']:.3f}")
        print(f"Final tracking err [deg]: {tracking_stats['final']:.4f}")
        print(f"Mean tracking err  [deg]: {tracking_stats['mean']:.4f}")
        print(f"Max tracking err   [deg]: {tracking_stats['max']:.4f}")
        print("========================================\n")

    if plot:
        ADCS.plot(
            results,
            ADCS.plots.AnimationPlot(),
            layout=(1, 1),
            title="3+1 SALTRO Reduced",
        )

        ADCS.plot(
            results,
            ADCS.plots.AttitudePlot(sources=["real", "reference"]),
            layout=(1, 1),
            title="3+1 SALTRO Reduced",
        )

        ADCS.plot(
            results,
            ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
            ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
            ADCS.plots.TargetHistogram(bin_width=5.0),
            ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
            layout=(2, 2),
            title="3+1 SALTRO Reduced",
        )

        plot_target_tracking(
            state_hist=state_hist,
            boresight_hist=target_hist,
            body_boresight=body_boresight,
            time=time_s,
        )

        plt.show()

    return results, tracking_stats, timing_stats


if __name__ == "__main__":
    run_saltro_3p1_reduced(
        tf=1000.0,
        dt=1.0,
        plot=True,
        verbose=True,
        goal_mode="quat_slew90",
        use_saltro_3_1_debug_settings=True,
    )