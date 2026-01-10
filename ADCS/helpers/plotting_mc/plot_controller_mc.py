import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

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
    plt.show()