import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import matplotlib.cm as cm

def _rot_mat_vec(q: np.ndarray) -> np.ndarray:
    """
    Vectorized conversion of Scalar-First Quaternions (w, x, y, z) 
    to Rotation Matrices (Body -> Inertial).
    
    Input: q shape (N, 4)
    Output: R shape (N, 3, 3)
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Formula for Rotation Matrix from Quaternion (Hamilton/Scalar First)
    R = np.empty((q.shape[0], 3, 3))
    
    R[:, 0, 0] = 1 - 2*(y**2 + z**2)
    R[:, 0, 1] = 2*(x*y - z*w)
    R[:, 0, 2] = 2*(x*z + y*w)
    
    R[:, 1, 0] = 2*(x*y + z*w)
    R[:, 1, 1] = 1 - 2*(x**2 + z**2)
    R[:, 1, 2] = 2*(y*z - x*w)
    
    R[:, 2, 0] = 2*(x*z - y*w)
    R[:, 2, 1] = 2*(y*z + x*w)
    R[:, 2, 2] = 1 - 2*(x**2 + y**2)
    
    return R
    
def plot_h_tracking_mc(
    full_results: List[Dict[str, Any]],
    body_boresight: np.ndarray = np.array([0, 0, 1]),
    title: str = "Monte Carlo Target Stored Angular Momentum"
) -> None:
    """
    Plots the stored angular momentum for multiple Monte Carlo runs on a single figure.
    """
    
    if not full_results:
        print("[plot_h_tracking_mc] Warning: No results to plot.")
        return

    # Normalize the fixed body vector once
    v_bore_body = body_boresight / np.linalg.norm(body_boresight)
    
    plt.figure(figsize=(5, 3))
    
    # Iterate through every MC run
    colors = []
    for run_idx, res in enumerate(full_results):
        
        # --- Validation Checks ---
        if "state" not in res or "time" not in res:
             # Skip malformed runs or raise error
             continue
        
        state = res["state"]       # Shape (N, 7+)
        goal = res["boresight_goal"] # Shape (N, 3) ECI
        time = res["time"]         # Shape (N,)
        
        # --- Calculation ---
        
        # 1. Extract Quaternions (Columns 3:7 -> w, x, y, z)
        h_hist = np.asarray(state[:, 7:],dtype=float)

        if not colors:
            color_num = h_hist.shape[1]
            cmap = cm.get_cmap('tab10')

            # Generate a list of M colors (RGBA tuples)
            # We select colors evenly spaced across the colormap
            colors = [cmap(i / color_num) for i in range(color_num)]
            
        
        
        
        # --- Plotting ---
        for j in range(color_num):
            plt.plot(time, h_hist[:,j], color=colors[j], alpha=0.1, linewidth=1.5)

    # Add a dummy line for the legend so it's not transparent
    for j in range(color_num):
        plt.plot([], [], color=colors[j], label='MC RWh of RW '+str(j))
    
    plt.xlabel("Time [s]")
    plt.ylabel("Stored Angular Momentum")
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()


def plot_target_tracking_mc(
    full_results: List[Dict[str, Any]],
    body_boresight: np.ndarray = np.array([0, 0, 1]),
    title: str = "Monte Carlo Target Tracking Error"
) -> None:
    """
    Plots the angular tracking error for multiple Monte Carlo runs on a single figure.
    """
    
    if not full_results:
        print("[plot_target_tracking_mc] Warning: No results to plot.")
        return

    # Normalize the fixed body vector once
    v_bore_body = body_boresight / np.linalg.norm(body_boresight)
    
    plt.figure(figsize=(10, 6))
    
    # Iterate through every MC run
    for run_idx, res in enumerate(full_results):
        
        # --- Validation Checks ---
        if "state" not in res or "boresight_goal" not in res or "time" not in res:
             # Skip malformed runs or raise error
             continue
        
        state = res["state"]       # Shape (N, 7+)
        goal = res["boresight_goal"] # Shape (N, 3) ECI
        time = res["time"]         # Shape (N,)
        
        # --- Calculation ---
        
        # 1. Extract Quaternions (Columns 3:7 -> w, x, y, z)
        q_hist = state[:, 3:7] 
        
        # 2. Get Rotation Matrices (Vectorized) -> USES LOCAL HELPER NOW
        R_b2i = _rot_mat_vec(q_hist) 
        
        # 3. Rotate Body Boresight to ECI
        # (N,3,3) @ (3,) -> (N,3)
        v_bore_eci = np.einsum('nij,j->ni', R_b2i, v_bore_body)
        
        # 4. Normalize Vectors (Row-wise)
        v_bore_eci_norm = np.linalg.norm(v_bore_eci, axis=1, keepdims=True)
        v_goal_norm = np.linalg.norm(goal, axis=1, keepdims=True)
        
        v_b = v_bore_eci / v_bore_eci_norm
        v_g = goal / v_goal_norm
        
        # 5. Dot Product & Angle
        dot_prod = np.sum(v_b * v_g, axis=1)
        dot_prod = np.clip(dot_prod, -1.0, 1.0)
        
        error_deg = np.rad2deg(np.arccos(dot_prod))
        
        # --- Plotting ---
        plt.plot(time, error_deg, color='tab:blue', alpha=0.1, linewidth=1.5)

    # Add a dummy line for the legend so it's not transparent
    plt.plot([], [], color='tab:blue', label='MC Runs')
    
    plt.xlabel("Time [s]")
    plt.ylabel("Tracking Error [deg]")
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()


def plot_convergence_histogram_mc(
    full_results: List[Dict[str, Any]],
    body_boresight: np.ndarray = np.array([0.0, 0.0, 1.0]),
    title: str = "Monte Carlo Convergence Error (Final Timestep)",
    bin_width_deg: float = 5.0,
    under_thresh_deg: float = 1.0,
    show_stats_box: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Computes the angular tracking error at the final timestep for each MC run
    and plots a histogram with fixed-degree bins.

    Returns
    -------
    errors_deg : np.ndarray
        Final-timestep errors for each valid run.
    stats : Dict[str, float]
        Summary statistics:
          - pct_under_thresh
          - min
          - max
          - mean
          - median
          - n
    """

    if not full_results:
        print("[plot_convergence_histogram_mc] Warning: No results to plot.")
        return np.array([]), {
            "pct_under_thresh": np.nan,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "n": 0.0,
        }

    v_bore_body = np.asarray(body_boresight, dtype=float)
    nb = np.linalg.norm(v_bore_body)
    if nb == 0:
        raise ValueError("body_boresight must be non-zero.")
    v_bore_body = v_bore_body / nb

    errors = []

    for res in full_results:
        # --- Validation ---
        if ("state" not in res) or ("boresight_goal" not in res):
            continue

        state = res["state"]
        goal = res["boresight_goal"]

        if state is None or goal is None:
            continue
        if len(state) == 0 or len(goal) == 0:
            continue

        # Guard against mismatched lengths
        N = min(state.shape[0], goal.shape[0])
        if N < 1:
            continue

        # Final timestep index
        k = N - 1

        # 1) Quaternion at final step (w,x,y,z)
        q = np.asarray(state[k, 3:7], dtype=float)
        if q.shape[0] != 4:
            continue

        # Optional: normalize quaternion for safety
        nq = np.linalg.norm(q)
        if nq == 0:
            continue
        q = q / nq

        # 2) Rotation matrix body->inertial for this single quaternion
        # Reuse your vectorized helper by wrapping (1,4)
        R_b2i = _rot_mat_vec(q.reshape(1, 4))[0]  # (3,3)

        # 3) Rotate boresight into ECI
        v_b = R_b2i @ v_bore_body

        # 4) Normalize boresight & goal
        v_g = np.asarray(goal[k, :], dtype=float).reshape(3,)
        nb2 = np.linalg.norm(v_b)
        ng2 = np.linalg.norm(v_g)
        if nb2 == 0 or ng2 == 0:
            continue
        v_b = v_b / nb2
        v_g = v_g / ng2

        # 5) Angle error
        dot = float(np.clip(np.dot(v_b, v_g), -1.0, 1.0))
        err_deg = float(np.rad2deg(np.arccos(dot)))
        errors.append(err_deg)

    errors_deg = np.asarray(errors, dtype=float)

    if errors_deg.size == 0:
        print("[plot_convergence_histogram_mc] Warning: No valid runs after filtering.")
        return errors_deg, {
            "pct_under_thresh": np.nan,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "n": 0.0,
        }

    # --- Stats ---
    pct_under = 100.0 * np.mean(errors_deg < under_thresh_deg)
    stats = {
        "pct_under_thresh": float(pct_under),
        "min": float(np.min(errors_deg)),
        "max": float(np.max(errors_deg)),
        "mean": float(np.mean(errors_deg)),
        "median": float(np.median(errors_deg)),
        "n": float(errors_deg.size),
    }

    # --- Histogram bins (5 deg default) ---
    # Ensure bins cover full range, starting at 0
    max_edge = np.ceil(errors_deg.max() / bin_width_deg) * bin_width_deg
    bins = np.arange(0.0, max_edge + bin_width_deg, bin_width_deg)

    plt.figure(figsize=(10, 6))
    plt.hist(errors_deg, bins=bins, edgecolor="black")
    plt.xlabel("Final Tracking Error [deg]")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)

    if show_stats_box:
        txt = (
            f"N = {int(stats['n'])}\n"
            f"% < {under_thresh_deg:.2f}°: {stats['pct_under_thresh']:.2f}%\n"
            f"min: {stats['min']:.3f}°\n"
            f"max: {stats['max']:.3f}°\n"
            f"mean: {stats['mean']:.3f}°\n"
            f"median: {stats['median']:.3f}°"
        )
        plt.gca().text(
            0.98, 0.98, txt,
            transform=plt.gca().transAxes,
            ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    plt.tight_layout()

    return errors_deg, stats