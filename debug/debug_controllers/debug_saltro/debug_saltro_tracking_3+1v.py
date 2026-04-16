"""
Track and compare open-loop vs closed-loop SALTRO control for 3+1 (vector goal).
"""
import sys
import os
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

# Ensure local SALTRO build is importable when running from Generalized_ADCS.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
saltro_path = os.path.join(parent_dir, "SALTRO", "build")
sys.path.append(saltro_path)

# Dynamic import of debug_saltro_3+1v (module name contains +)
debug_module_path = os.path.join(current_dir, "debug_saltro_3+1v.py")
spec = importlib.util.spec_from_file_location("debug_saltro_3_1v", debug_module_path)
debug_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(debug_module)

debug_saltro = debug_module.debug_saltro
debug_saltro_closed_loop = debug_module.debug_saltro_closed_loop
_plot_debug_run = debug_module._plot_debug_run
_boresight_eci = debug_module._boresight_eci
_vec_angle_deg = debug_module._vec_angle_deg


def compute_vector_error(
    q: np.ndarray,
    vector_goal_hist: np.ndarray,
    boresight_body: np.ndarray,
    time_hist: np.ndarray,
) -> np.ndarray:
    """Compute pointing error between boresight and goal vector."""
    vec_err = np.zeros(len(q), dtype=np.float64)
    for i in range(len(q)):
        q_i = q[i]  # Quaternion: [q0, q1, q2, q3]
        goal_row = vector_goal_hist[i, :]  # [nan, gx, gy, gz]
        
        # Extract components
        target_vec = np.asarray(goal_row[1:4], dtype=np.float64)
        if np.linalg.norm(target_vec) <= 0.0:
            vec_err[i] = np.nan
            continue
        
        # Get boresight vector in body frame
        if boresight_body is None:
            vec_err[i] = np.nan
            continue
            
        if boresight_body.ndim == 1:
            bore_body = boresight_body
        else:
            # Expand boresight to match trajectory (zero-order hold at knot points)
            goal_idx = np.searchsorted(time_hist, time_hist[i], side='right') if len(time_hist) > 1 else 0
            if goal_idx >= boresight_body.shape[1]:
                goal_idx = boresight_body.shape[1] - 1
            bore_body = boresight_body[:, goal_idx]
        
        if np.linalg.norm(bore_body) <= 0.0:
            vec_err[i] = np.nan
            continue
        
        # Transform boresight from body to inertial frame
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
    """Plot open-loop and closed-loop results for 3+1 vector goal."""
    
    print("Computing open-loop trajectory...")
    time_hist_ol, state_hist_ol, os_hist_ol, sensor_hist_ol, u_hist_ol, vector_goal_hist_ol, boresight_ol = debug_saltro(
        verbose=verbose,
        tf=tf,
        dt=dt,
        real_orbit=real_orbit,
    )
    
    print("Computing closed-loop trajectory...")
    time_hist_cl, state_hist_cl, os_hist_cl, sensor_hist_cl, u_hist_cl, vector_goal_hist_cl, boresight_cl = debug_saltro_closed_loop(
        verbose=verbose,
        tf=tf,
        planner_dt=dt,
        sim_dt=closed_loop_dt,
        real_orbit=real_orbit,
    )
    
    # Compute pointing errors
    q_ol = state_hist_ol[:, 3:7]
    q_cl = state_hist_cl[:, 3:7]
    
    vec_err_ol = compute_vector_error(q_ol, vector_goal_hist_ol, boresight_ol, time_hist_ol)
    vec_err_cl = compute_vector_error(q_cl, vector_goal_hist_cl, boresight_cl, time_hist_cl)
    
    # Create comparison plots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    
    # Extract state components
    w_ol = state_hist_ol[:, 0:3]
    h_ol = state_hist_ol[:, 7:]
    
    w_cl = state_hist_cl[:, 0:3]
    h_cl = state_hist_cl[:, 7:]
    
    # Quaternion
    ax = axes[0, 0]
    for i in range(4):
        ax.plot(time_hist_ol, q_ol[:, i], linewidth=1.5, label=f"q{i} (OL)", alpha=0.7)
        ax.plot(time_hist_cl, q_cl[:, i], "--", linewidth=1.2, label=f"q{i} (CL)", alpha=0.7)
    ax.set_title("Quaternion: Open-Loop vs Closed-Loop (Vector Goal)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("q")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    
    # Pointing error
    ax = axes[0, 1]
    ax.plot(time_hist_ol, vec_err_ol, linewidth=2.0, label="Pointing Error (OL)", alpha=0.7)
    ax.plot(time_hist_cl, vec_err_cl, "--", linewidth=1.5, label="Pointing Error (CL)", alpha=0.7)
    ax.set_title("Vector Pointing Error")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("deg")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Wheel momentum
    ax = axes[1, 0]
    if h_ol.shape[1] > 0:
        for i in range(h_ol.shape[1]):
            ax.plot(time_hist_ol, h_ol[:, i], linewidth=1.5, label=f"h_rw{i} (OL)", alpha=0.7)
            if i < h_cl.shape[1]:
                ax.plot(time_hist_cl, h_cl[:, i], "--", linewidth=1.2, label=f"h_rw{i} (CL)", alpha=0.7)
    ax.set_title("RW Momentum")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("N·m·s")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Control magnitude
    ax = axes[1, 1]
    n_u_ol = min(len(time_hist_ol), u_hist_ol.shape[0])
    n_u_cl = min(len(time_hist_cl), u_hist_cl.shape[0])
    ax.plot(time_hist_ol[:n_u_ol], np.linalg.norm(u_hist_ol[:n_u_ol, :], axis=1), 
            linewidth=1.5, label="||u|| (OL)", alpha=0.7)
    ax.plot(time_hist_cl[:n_u_cl], np.linalg.norm(u_hist_cl[:n_u_cl, :], axis=1), 
            "--", linewidth=1.2, label="||u|| (CL)", alpha=0.7)
    ax.set_title("Control Magnitude")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Angular velocity magnitude
    ax = axes[2, 0]
    ax.plot(time_hist_ol, np.linalg.norm(w_ol, axis=1), linewidth=1.5, label="||w|| (OL)", alpha=0.7)
    ax.plot(time_hist_cl, np.linalg.norm(w_cl, axis=1), "--", linewidth=1.2, label="||w|| (CL)", alpha=0.7)
    ax.set_title("Angular Velocity Magnitude")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("rad/s")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Individual angular velocities (CL vs OL at end time)
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
    
    fig.suptitle("SALTRO 3+1 Tracking: Open-Loop vs Closed-Loop (Vector Goal)", 
                 fontsize=14, fontweight="bold")
    plt.show()
    print("Rendering complete.")


if __name__ == "__main__":
    plot_comparison(verbose=True, tf=1000.0, dt=5.0, closed_loop_dt=1.0, real_orbit=True)
