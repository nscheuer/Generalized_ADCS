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

        # Vector-goal rows are encoded as [nan, gx, gy, gz].
        if not np.isnan(row[0]):
            continue

        q = x[k, 3:7]
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

    valid = err[np.isfinite(err)]
    if valid.size == 0:
        return {"final": np.nan, "mean": np.nan, "max": np.nan}

    return {
        "final": float(valid[-1]),
        "mean": float(np.mean(valid)),
        "max": float(np.max(valid)),
    }


def run_saltro_3p1_reduced(
    tf: float = 1000.0,
    dt: float = 1.0,
    plot: bool = True,
    verbose: bool = True,
) -> Tuple[object, Dict[str, float], Dict[str, float]]:
    real_sat = ADCS.satellite_factory.create_beavercube2_cubesat(estimated=False)
    x_0 = np.array([0.0, 0.0, 0.0] + [1, 0, 0, 0] + [0.0], dtype=np.float64)

    # Use default NEO planner settings as requested (no manual tuning overrides).
    planner_settings = PlannerSettings(est_sat=real_sat)
    controller = ADCS.controller.SALTRO(est_sat=real_sat, planner_settings=planner_settings)

    os0 = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(),
        J2000=0.22,
        # SALTRO validates LEO orbital bounds; this circular seed matches SALTRO debug usage.
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]),
    )
    goal = ECI_Goal(np.array([0.0, 0.0, -1.0]))

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
    run_saltro_3p1_reduced(tf=1000.0, dt=1.0, plot=True, verbose=True)