"""
Track and compare open-loop vs closed-loop SALTRO control for 3+1 (quaternion goal).
"""
import sys
import os
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../...")))

# Ensure local SALTRO build is importable when running from Generalized_ADCS.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
saltro_path = os.path.join(parent_dir, "SALTRO", "build")
sys.path.append(saltro_path)

# Dynamic import of debug_saltro_3+1 (module name contains +)
debug_module_path = os.path.join(current_dir, "debug_saltro_3+1.py")
spec = importlib.util.spec_from_file_location("debug_saltro_3_1", debug_module_path)
debug_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(debug_module)

debug_saltro = debug_module.debug_saltro
debug_saltro_closed_loop = debug_module.debug_saltro_closed_loop
_plot_debug_run = debug_module._plot_debug_run


def plot_comparison(
    verbose: bool = False,
    tf: float = 1000.0,
    dt: float = 5.0,
    closed_loop_dt: float = 1.0,
    real_orbit: bool = True,
) -> None:
    """Plot open-loop and closed-loop results for 3+1 quaternion goal."""
    
    print("Computing open-loop trajectory...")
    time_hist_ol, state_hist_ol, os_hist_ol, sensor_hist_ol, u_hist_ol, boresight_hist_ol = debug_saltro(
        verbose=verbose,
        tf=tf,
        dt=dt,
        real_orbit=real_orbit,
    )
    
    print("Computing closed-loop trajectory...")
    time_hist_cl, state_hist_cl, os_hist_cl, sensor_hist_cl, u_hist_cl, boresight_hist_cl = debug_saltro_closed_loop(
        verbose=verbose,
        tf=tf,
        planner_dt=dt,
        sim_dt=closed_loop_dt,
        real_orbit=real_orbit,
    )
    
    # Create comparison plots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    
    # Extract state components
    q_ol = state_hist_ol[:, 3:7]
    w_ol = state_hist_ol[:, 0:3]
    h_ol = state_hist_ol[:, 7:]
    
    q_cl = state_hist_cl[:, 3:7]
    w_cl = state_hist_cl[:, 0:3]
    h_cl = state_hist_cl[:, 7:]
    
    # Quaternion
    ax = axes[0, 0]
    for i in range(4):
        ax.plot(time_hist_ol, q_ol[:, i], linewidth=1.5, label=f"q{i} (OL)", alpha=0.7)
        ax.plot(time_hist_cl, q_cl[:, i], "--", linewidth=1.2, label=f"q{i} (CL)", alpha=0.7)
    ax.set_title("Quaternion: Open-Loop vs Closed-Loop")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("q")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    
    # Angular velocity
    ax = axes[0, 1]
    ax.plot(time_hist_ol, np.linalg.norm(w_ol, axis=1), linewidth=2.0, label="||w|| (OL)", alpha=0.7)
    ax.plot(time_hist_cl, np.linalg.norm(w_cl, axis=1), "--", linewidth=1.5, label="||w|| (CL)", alpha=0.7)
    ax.set_title("Angular Velocity Magnitude")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("rad/s")
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
    
    # Angular velocity components (OL)
    ax = axes[2, 0]
    for i, name in enumerate(["wx", "wy", "wz"]):
        ax.plot(time_hist_ol, w_ol[:, i], linewidth=1.5, label=f"{name} (OL)", alpha=0.7)
    ax.set_title("Angular Velocity Components (Open-Loop)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("rad/s")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Angular velocity components (CL)
    ax = axes[2, 1]
    for i, name in enumerate(["wx", "wy", "wz"]):
        ax.plot(time_hist_cl, w_cl[:, i], linewidth=1.5, label=f"{name} (CL)", alpha=0.7)
    ax.set_title("Angular Velocity Components (Closed-Loop)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("rad/s")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    fig.suptitle("SALTRO 3+1 Tracking: Open-Loop vs Closed-Loop (Quaternion Goal)", 
                 fontsize=14, fontweight="bold")
    plt.show()
    print("Rendering complete.")


if __name__ == "__main__":
    plot_comparison(verbose=True, tf=1000.0, dt=5.0, closed_loop_dt=1.0, real_orbit=True)
