"""
Track and compare open-loop vs closed-loop SALTRO control for BC2 vector-goal case.
"""
import sys
import os
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy.integrate import solve_ivp
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

# Ensure local SALTRO build is importable when running from Generalized_ADCS.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
saltro_path = os.path.join(parent_dir, "SALTRO", "build")
sys.path.append(saltro_path)

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.controller.helpers.trajectory import Trajectory
from ADCS.helpers.math_helpers import normalize
from ADCS.state import State

# Path-based import so this script works regardless of current working directory.
debug_module_path = os.path.join(current_dir, "debug_saltro_bc2.py")
spec = importlib.util.spec_from_file_location("debug_saltro_bc2_module", debug_module_path)
debug_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(debug_module)

debug_saltro = debug_module.debug_saltro
_build_common_setup = debug_module._build_common_setup
_run_open_loop_trajopt = debug_module._run_open_loop_trajopt
_build_orbit = debug_module._build_orbit
_goal_hist_from_knots = debug_module._goal_hist_from_knots
_boresight_eci = debug_module._boresight_eci
_vec_angle_deg = debug_module._vec_angle_deg


def _reshape_saltro_gains(K_flat: np.ndarray, n_steps: int, n_ctrl: int, n_red: int) -> np.ndarray:
    if K_flat.shape != (n_ctrl, n_red * n_steps):
        raise ValueError(f"Unexpected K shape {K_flat.shape}, expected {(n_ctrl, n_red * n_steps)}")

    K_time = np.zeros((n_steps, n_ctrl, n_red), dtype=np.float64)
    for k in range(n_steps):
        c0 = k * n_red
        c1 = c0 + n_red
        K_time[k, :, :] = K_flat[:, c0:c1]
    return K_time


def _debug_saltro_closed_loop_bc2(
    verbose: bool = False,
    tf: float = 1000.0,
    planner_dt: float = 5.0,
    sim_dt: float = 1.0,
    real_orbit: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Orbital_State], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _ = real_orbit

    if sim_dt <= 0.0:
        raise ValueError(f"sim_dt must be > 0, got {sim_dt}")

    real_sat, x0, os0, t_start = _build_common_setup()
    X_ref, U_ref, K_flat, jtime_req, vector_goal, boresight = _run_open_loop_trajopt(
        real_sat=real_sat,
        x0=x0,
        os0=os0,
        t_start=t_start,
        tf=tf,
        planner_dt=planner_dt,
    )

    n_ref = X_ref.shape[1]
    n_ctrl = U_ref.shape[0]
    n_red = x0.size - 1
    K_time = _reshape_saltro_gains(K_flat=K_flat, n_steps=n_ref, n_ctrl=n_ctrl, n_red=n_red)
    # SALTRO returns gains for u = u_nom + K*dx; Trajectory expects u = u_ref - K*dx.
    K_time *= -1.0

    S_dummy = np.zeros(n_ref, dtype=np.float64)
    traj = Trajectory.from_arrays(
        t=np.asarray(np.linspace(t_start, t_start + tf * TimeConstants.sec2cent, n_ref), dtype=np.float64),
        x=np.asarray(X_ref, dtype=np.float64),
        u=np.asarray(U_ref, dtype=np.float64),
        K=K_time,
        S=S_dummy,
    )

    n_out = int(np.floor(tf / sim_dt)) + 1
    time_hist = np.arange(n_out, dtype=np.float64) * sim_dt
    jtime = t_start + time_hist * TimeConstants.sec2cent

    orb = _build_orbit(os0=os0, t_start=t_start, t_end=float(jtime[-1]), dt=sim_dt)

    state_hist = np.zeros((n_out, x0.size), dtype=np.float64)
    u_hist = np.zeros((n_out, n_ctrl), dtype=np.float64)
    sensor_hist = np.nan * np.zeros((n_out, len(real_sat.sensors + real_sat.rw_actuators)))
    os_hist: List[Orbital_State] = []

    state_hist[0, :] = np.asarray(x0, dtype=np.float64)

    for k in tqdm(range(n_out), desc="Simulating SALTRO closed-loop (bc2)"):
        os_k = orb.get_os(J2000=float(jtime[k]))
        os_hist.append(os_k)
        sensor_hist[k, :] = real_sat.sensor_readings(x=State.from_array(state_hist[k, :]), os=os_k)

        u_cmd = traj.compute_tracking_control(float(jtime[k]), State.from_array(state_hist[k, :]))
        u_hist[k, :] = u_cmd

        if k < n_out - 1:
            os_next = orb.get_os(J2000=float(jtime[k + 1]))
            dt_step = float(time_hist[k + 1] - time_hist[k])
            sol = solve_ivp(
                real_sat.dynamics_for_solver,
                (0.0, dt_step),
                y0=state_hist[k, :],
                args=(u_cmd, os_k, os_next),
                atol=1e-9,
                rtol=1e-7,
            )
            x_next = sol.y[:, -1]
            x_next[3:7] = normalize(x_next[3:7])
            state_hist[k + 1, :] = x_next

    vector_goal_hist = _goal_hist_from_knots(jtime_req=jtime_req, vector_goal=vector_goal, jtime=jtime)

    if verbose:
        print("SALTRO trajOpt succeeded (closed-loop TVLQR, bc2)")
        print(f"N ref={n_ref}, N sim={n_out}")
        print(f"X_ref shape={X_ref.shape}, U_ref shape={U_ref.shape}, K_flat shape={K_flat.shape}")

    return time_hist, state_hist, os_hist, sensor_hist, u_hist, vector_goal_hist, boresight


def compute_vector_error(
    q: np.ndarray,
    vector_goal_hist: np.ndarray,
    boresight_body: np.ndarray,
    time_hist: np.ndarray,
) -> np.ndarray:
    """Compute pointing error between boresight and goal vector."""
    vec_err = np.zeros(len(q), dtype=np.float64)
    for i in range(len(q)):
        q_i = q[i]
        goal_row = vector_goal_hist[i, :]

        target_vec = np.asarray(goal_row[1:4], dtype=np.float64)
        if np.linalg.norm(target_vec) <= 0.0:
            vec_err[i] = np.nan
            continue

        if boresight_body is None:
            vec_err[i] = np.nan
            continue

        if boresight_body.ndim == 1:
            bore_body = boresight_body
        else:
            goal_idx = np.searchsorted(time_hist, time_hist[i], side="right") if len(time_hist) > 1 else 0
            if goal_idx >= boresight_body.shape[1]:
                goal_idx = boresight_body.shape[1] - 1
            bore_body = boresight_body[:, goal_idx]

        if np.linalg.norm(bore_body) <= 0.0:
            vec_err[i] = np.nan
            continue

        bore_unit = bore_body / np.linalg.norm(bore_body)
        target_unit = target_vec / np.linalg.norm(target_vec)
        bore_inertial = _boresight_eci(q_i, bore_unit)
        vec_err[i] = _vec_angle_deg(bore_inertial, target_unit)

    return vec_err


def plot_comparison(
    verbose: bool = False,
    tf: float = 1000.0,
    dt: float = 5.0,
    closed_loop_dt: float = 1.0,
    real_orbit: bool = True,
) -> None:
    """Plot open-loop and closed-loop results for BC2 vector goal."""
    print("Computing open-loop trajectory...")
    time_hist_ol, state_hist_ol, os_hist_ol, sensor_hist_ol, u_hist_ol, vector_goal_hist_ol, boresight_ol = debug_saltro(
        verbose=verbose,
        tf=tf,
        dt=dt,
        real_orbit=real_orbit,
    )

    print("Computing closed-loop trajectory...")
    time_hist_cl, state_hist_cl, os_hist_cl, sensor_hist_cl, u_hist_cl, vector_goal_hist_cl, boresight_cl = _debug_saltro_closed_loop_bc2(
        verbose=verbose,
        tf=tf,
        planner_dt=dt,
        sim_dt=closed_loop_dt,
        real_orbit=real_orbit,
    )

    _ = os_hist_ol
    _ = os_hist_cl
    _ = sensor_hist_ol
    _ = sensor_hist_cl

    q_ol = state_hist_ol[:, 3:7]
    q_cl = state_hist_cl[:, 3:7]
    w_ol = state_hist_ol[:, 0:3]
    w_cl = state_hist_cl[:, 0:3]
    h_ol = state_hist_ol[:, 7:]
    h_cl = state_hist_cl[:, 7:]

    vec_err_ol = compute_vector_error(q_ol, vector_goal_hist_ol, boresight_ol, time_hist_ol)
    vec_err_cl = compute_vector_error(q_cl, vector_goal_hist_cl, boresight_cl, time_hist_cl)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)

    ax = axes[0, 0]
    for i in range(4):
        ax.plot(time_hist_ol, q_ol[:, i], linewidth=1.5, label=f"q{i} (OL)", alpha=0.7)
        ax.plot(time_hist_cl, q_cl[:, i], "--", linewidth=1.2, label=f"q{i} (CL)", alpha=0.7)
    ax.set_title("Quaternion: Open-Loop vs Closed-Loop (BC2)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("q")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, ncol=2)

    ax = axes[0, 1]
    ax.plot(time_hist_ol, vec_err_ol, linewidth=2.0, label="Pointing Error (OL)", alpha=0.7)
    ax.plot(time_hist_cl, vec_err_cl, "--", linewidth=1.5, label="Pointing Error (CL)", alpha=0.7)
    ax.set_title("BC2 Vector Pointing Error")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("deg")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    if h_ol.shape[1] > 0:
        for i in range(h_ol.shape[1]):
            ax.plot(time_hist_ol, h_ol[:, i], linewidth=1.5, label=f"h_rw{i} (OL)", alpha=0.7)
            if i < h_cl.shape[1]:
                ax.plot(time_hist_cl, h_cl[:, i], "--", linewidth=1.2, label=f"h_rw{i} (CL)", alpha=0.7)
    ax.set_title("RW Momentum")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("N*m*s")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    n_u_ol = min(len(time_hist_ol), u_hist_ol.shape[0])
    n_u_cl = min(len(time_hist_cl), u_hist_cl.shape[0])
    ax.plot(
        time_hist_ol[:n_u_ol],
        np.linalg.norm(u_hist_ol[:n_u_ol, :], axis=1),
        linewidth=1.5,
        label="||u|| (OL)",
        alpha=0.7,
    )
    ax.plot(
        time_hist_cl[:n_u_cl],
        np.linalg.norm(u_hist_cl[:n_u_cl, :], axis=1),
        "--",
        linewidth=1.2,
        label="||u|| (CL)",
        alpha=0.7,
    )
    ax.set_title("Control Magnitude")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2, 0]
    ax.plot(time_hist_ol, np.linalg.norm(w_ol, axis=1), linewidth=1.5, label="||w|| (OL)", alpha=0.7)
    ax.plot(time_hist_cl, np.linalg.norm(w_cl, axis=1), "--", linewidth=1.2, label="||w|| (CL)", alpha=0.7)
    ax.set_title("Angular Velocity Magnitude")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("rad/s")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    ax.plot(time_hist_ol, w_ol[:, 0], linewidth=1.5, label="wx (OL)", alpha=0.7)
    ax.plot(time_hist_ol, w_ol[:, 1], linewidth=1.5, label="wy (OL)", alpha=0.7)
    ax.plot(time_hist_ol, w_ol[:, 2], linewidth=1.5, label="wz (OL)", alpha=0.7)
    ax.plot(time_hist_cl, w_cl[:, 0], "--", linewidth=1.2, label="wx (CL)", alpha=0.7)
    ax.plot(time_hist_cl, w_cl[:, 1], "--", linewidth=1.2, label="wy (CL)", alpha=0.7)
    ax.plot(time_hist_cl, w_cl[:, 2], "--", linewidth=1.2, label="wz (CL)", alpha=0.7)
    ax.set_title("Angular Velocity Components")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("rad/s")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, ncol=2)

    fig.suptitle("SALTRO BC2 Tracking: Open-Loop vs Closed-Loop", fontsize=14, fontweight="bold")
    plt.show()
    print("Rendering complete.")


if __name__ == "__main__":
    plot_comparison(verbose=True, tf=1000.0, dt=5.0, closed_loop_dt=1.0, real_orbit=True)
