__all__ = [
    "plot_h_tracking_mc",
    "plot_target_tracking_mc",
    "plot_convergence_histogram_mc",
    "plot_single_run",
    "plot_quaternion_error_mc",
    "plot_quaternion_histogram_mc",
    "plot_mc_summary",
    "plot_planned_trajectory",
    "create_planner_diagnostic_callback",
]

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
    r"""
    Plot stored reaction wheel angular momentum for multiple Monte Carlo runs.

    This function overlays reaction wheel stored angular momentum histories
    from a set of Monte Carlo simulations on a single figure. Each Monte Carlo
    run is plotted with low opacity to visualize dispersion and trends across
    the ensemble.

    ======================
    State Assumption
    ======================

    Each Monte Carlo result dictionary is expected to contain:

    * ``"time"`` — time history
    * ``"state"`` — state history with reaction wheel momentum stored from
      index 7 onward

    The reaction wheel momentum vector is assumed to be

    .. math::

        \mathbf{h}_{\text{rw}}(t) \in \mathbb{R}^{N_{\text{rw}}}

    ======================
    Visualization Strategy
    ======================

    * Each reaction wheel component is assigned a consistent color
    * Individual Monte Carlo runs are plotted with transparency
    * A non-transparent legend entry is added for clarity

    :param full_results:
        List of Monte Carlo result dictionaries.
    :type full_results:
        list of dict

    :param body_boresight:
        Fixed boresight direction expressed in the body frame.
        (Included for interface consistency; not used in computation.)
    :type body_boresight:
        numpy.ndarray

    :param title:
        Plot title.
    :type title:
        str

    :return:
        None. The function generates a Matplotlib figure.
    :rtype:
        None

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
            plt.plot(time, h_hist[:,j], color=colors[j], alpha=0.3, linewidth=1.5)

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
    r"""
    Plot angular target tracking error for multiple Monte Carlo runs.

    This function computes and overlays the instantaneous angular tracking
    error between a fixed body-frame boresight and a desired inertial target
    direction across many Monte Carlo simulations.

    ==========================
    Tracking Error Definition
    ==========================

    Let:

    * :math:`\hat{\mathbf{b}}` be the normalized boresight direction in the body frame
    * :math:`\mathbf{R}_{\mathcal{B}\rightarrow\mathcal{I}}` be the attitude rotation matrix
    * :math:`\hat{\mathbf{g}}_i` be the normalized target direction in ECI

    The boresight direction in ECI is

    .. math::

        \hat{\mathbf{b}}_i^{\text{ECI}} =
        \mathbf{R}_{\mathcal{B}\rightarrow\mathcal{I}} \hat{\mathbf{b}}

    The angular tracking error at time :math:`t_i` is

    .. math::

        \theta_i =
        \cos^{-1}\!\left(
        \hat{\mathbf{b}}_i^{\text{ECI}} \cdot \hat{\mathbf{g}}_i
        \right)

    ======================
    Visualization Strategy
    ======================

    * Each Monte Carlo run is plotted with low opacity
    * All runs share a common color for ensemble visualization

    :param full_results:
        List of Monte Carlo result dictionaries.
    :type full_results:
        list of dict

    :param body_boresight:
        Fixed boresight direction expressed in the body frame.
    :type body_boresight:
        numpy.ndarray

    :param title:
        Plot title.
    :type title:
        str

    :return:
        None. The function generates a Matplotlib figure.
    :rtype:
        None

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
        plt.plot(time, error_deg, color='tab:blue', alpha=0.3, linewidth=1.5)

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
    r"""
    Plot and analyze the distribution of final-timestep tracking error
    across Monte Carlo simulations.

    This function computes the angular tracking error at the final timestep
    for each valid Monte Carlo run and visualizes the resulting distribution
    using a histogram with fixed-width angular bins. Summary statistics are
    optionally displayed on the plot.

    ======================
    Final Error Definition
    ======================

    For each Monte Carlo run, the final angular tracking error is defined as

    .. math::

        \theta_f =
        \cos^{-1}\!\left(
        \hat{\mathbf{b}}_f^{\text{ECI}} \cdot \hat{\mathbf{g}}_f
        \right)

    where:

    * :math:`\hat{\mathbf{b}}_f^{\text{ECI}}` is the final boresight direction
    * :math:`\hat{\mathbf{g}}_f` is the final target direction

    ======================
    Statistics Reported
    ======================

    The following summary statistics are computed:

    +----------------------+----------------------------------+
    | Statistic            | Description                      |
    +======================+==================================+
    | ``pct_under_thresh`` | Percent of runs below threshold  |
    +----------------------+----------------------------------+
    | ``min``              | Minimum final error (deg)        |
    +----------------------+----------------------------------+
    | ``max``              | Maximum final error (deg)        |
    +----------------------+----------------------------------+
    | ``mean``             | Mean final error (deg)           |
    +----------------------+----------------------------------+
    | ``median``           | Median final error (deg)         |
    +----------------------+----------------------------------+
    | ``n``                | Number of valid runs             |
    +----------------------+----------------------------------+

    ======================
    Histogram Construction
    ======================

    * Histogram bins start at 0 degrees
    * Bin width is user-defined
    * Bins cover the full range of observed errors

    :param full_results:
        List of Monte Carlo result dictionaries.
    :type full_results:
        list of dict

    :param body_boresight:
        Fixed boresight direction expressed in the body frame.
    :type body_boresight:
        numpy.ndarray

    :param title:
        Plot title.
    :type title:
        str

    :param bin_width_deg:
        Width of histogram bins in degrees.
    :type bin_width_deg:
        float

    :param under_thresh_deg:
        Threshold angle (degrees) used for convergence statistics.
    :type under_thresh_deg:
        float

    :param show_stats_box:
        If True, display a statistics summary box on the plot.
    :type show_stats_box:
        bool

    :return:
        Array of final tracking errors (degrees) and a dictionary of summary statistics.
    :rtype:
        tuple of numpy.ndarray and dict

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


def _compute_quaternion_error(q_current: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
    """
    Compute quaternion error angle in degrees.

    Parameters
    ----------
    q_current : np.ndarray
        Current quaternion(s), shape (N, 4) or (4,), scalar-first [w, x, y, z]
    q_goal : np.ndarray
        Goal quaternion(s), shape (N, 4) or (4,), scalar-first [w, x, y, z]

    Returns
    -------
    np.ndarray
        Error angle in degrees, shape (N,) or scalar
    """
    q_current = np.atleast_2d(q_current)
    q_goal = np.atleast_2d(q_goal)

    # Handle single goal quaternion broadcast
    if q_goal.shape[0] == 1 and q_current.shape[0] > 1:
        q_goal = np.tile(q_goal, (q_current.shape[0], 1))

    # Quaternion dot product gives cos(theta/2)
    dot = np.sum(q_current * q_goal, axis=1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)

    # Error angle = 2 * arccos(|dot|)
    error_rad = 2.0 * np.arccos(dot)
    error_deg = np.rad2deg(error_rad)

    return error_deg.squeeze()


def plot_single_run(
    result: Dict[str, Any],
    body_boresight: np.ndarray = np.array([0.0, 1.0, 0.0]),
    title_prefix: str = "Single Run",
    show: bool = False,
) -> plt.Figure:
    """
    Plot comprehensive results for a single simulation run.

    Creates a 2x3 subplot figure with:
    - Angular velocity components (ω_x, ω_y, ω_z)
    - MTQ control inputs
    - RW control inputs (if present)
    - Attitude error (quaternion or boresight)
    - Quaternion components
    - Angular velocity magnitude

    When trajectory data is present (from planner runs), both the planned
    trajectory (dashed) and actual tracking (solid) are shown.

    Parameters
    ----------
    result : Dict[str, Any]
        Single simulation result dictionary with keys:
        - "time": time history
        - "state": state history (N, 7+)
        - "u": control history (N, n_u)
        - "q_goal" or "boresight_goal": goal specification
        Optional trajectory keys (from planner runs):
        - "traj_time": trajectory time history
        - "traj_state": planned state history
        - "traj_u": planned control history
    body_boresight : np.ndarray
        Boresight direction in body frame
    title_prefix : str
        Prefix for plot titles
    show : bool
        If True, call plt.show() at the end

    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    if not result or "state" not in result or "time" not in result:
        print("[plot_single_run] Warning: Invalid result dictionary.")
        return None

    time = result["time"]
    state = result["state"]
    u = result.get("u", None)

    # Check for trajectory data (planner runs)
    has_traj = "traj_time" in result and "traj_state" in result
    if has_traj:
        traj_time = result["traj_time"]
        traj_state = result["traj_state"]
        traj_u = result.get("traj_u", None)

    # Determine goal type
    has_quat_goal = "q_goal" in result
    has_boresight_goal = "boresight_goal" in result

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Extract state components
    omega = state[:, 0:3]  # rad/s
    quat = state[:, 3:7]   # [w, x, y, z]
    h_rw = state[:, 7:] if state.shape[1] > 7 else None

    # Extract trajectory state components if available
    if has_traj:
        traj_omega = traj_state[:, 0:3]
        traj_quat = traj_state[:, 3:7]
        traj_h_rw = traj_state[:, 7:] if traj_state.shape[1] > 7 else None

    # 1. Angular velocity components
    ax = axes[0, 0]
    omega_deg = np.rad2deg(omega)
    ax.plot(time, omega_deg[:, 0], 'r-', label='ωx (track)', linewidth=1.5)
    ax.plot(time, omega_deg[:, 1], 'g-', label='ωy (track)', linewidth=1.5)
    ax.plot(time, omega_deg[:, 2], 'b-', label='ωz (track)', linewidth=1.5)
    if has_traj:
        traj_omega_deg = np.rad2deg(traj_omega)
        ax.plot(traj_time, traj_omega_deg[:, 0], 'r--', label='ωx (plan)', alpha=0.7, linewidth=1.0)
        ax.plot(traj_time, traj_omega_deg[:, 1], 'g--', label='ωy (plan)', alpha=0.7, linewidth=1.0)
        ax.plot(traj_time, traj_omega_deg[:, 2], 'b--', label='ωz (plan)', alpha=0.7, linewidth=1.0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angular Velocity [°/s]")
    ax.set_title(f"{title_prefix}: Angular Velocity")
    ax.legend(fontsize='small', ncol=2)
    ax.grid(True, linestyle='--', alpha=0.6)

    # 2. MTQ control inputs
    ax = axes[0, 1]
    if u is not None:
        n_mtq = min(3, u.shape[1])
        colors = ['r', 'g', 'b']
        labels = ['MTQ_x', 'MTQ_y', 'MTQ_z']
        for i in range(n_mtq):
            ax.plot(time, u[:, i], colors[i], label=f'{labels[i]} (track)', linewidth=1.5)
        if has_traj and traj_u is not None:
            n_mtq_traj = min(3, traj_u.shape[1])
            traj_ctrl_time = traj_time[:traj_u.shape[0]]
            for i in range(n_mtq_traj):
                ax.plot(traj_ctrl_time, traj_u[:, i], colors[i], linestyle='--',
                       label=f'{labels[i]} (plan)', alpha=0.7, linewidth=1.0)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("MTQ Dipole [A·m²]")
        ax.set_title(f"{title_prefix}: MTQ Controls")
        ax.legend(fontsize='small', ncol=2)
        ax.grid(True, linestyle='--', alpha=0.6)
    else:
        ax.text(0.5, 0.5, "No control data", ha='center', va='center', transform=ax.transAxes)

    # 3. RW control inputs (or RW momentum if no RW controls)
    ax = axes[0, 2]
    if u is not None and u.shape[1] > 3:
        n_rw = u.shape[1] - 3
        for i in range(n_rw):
            ax.plot(time, u[:, 3 + i], label=f'RW_{i} (track)', linewidth=1.5)
        if has_traj and traj_u is not None and traj_u.shape[1] > 3:
            n_rw_traj = traj_u.shape[1] - 3
            traj_ctrl_time = traj_time[:traj_u.shape[0]]
            for i in range(n_rw_traj):
                ax.plot(traj_ctrl_time, traj_u[:, 3 + i], linestyle='--',
                       label=f'RW_{i} (plan)', alpha=0.7, linewidth=1.0)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("RW Torque [N·m]")
        ax.set_title(f"{title_prefix}: RW Controls")
        ax.legend(fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.6)
    elif h_rw is not None and h_rw.shape[1] > 0:
        for i in range(h_rw.shape[1]):
            ax.plot(time, h_rw[:, i], label=f'h_RW{i} (track)', linewidth=1.5)
        if has_traj and traj_h_rw is not None and traj_h_rw.shape[1] > 0:
            for i in range(traj_h_rw.shape[1]):
                ax.plot(traj_time, traj_h_rw[:, i], linestyle='--',
                       label=f'h_RW{i} (plan)', alpha=0.7, linewidth=1.0)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("RW Momentum [N·m·s]")
        ax.set_title(f"{title_prefix}: RW Momentum")
        ax.legend(fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.6)
    else:
        ax.text(0.5, 0.5, "No RW data", ha='center', va='center', transform=ax.transAxes)

    # 4. Attitude error
    ax = axes[1, 0]
    if has_quat_goal:
        q_goal = result["q_goal"]
        # Handle constant vs time-varying goal
        if q_goal.ndim == 1:
            q_goal_arr = np.tile(q_goal, (len(time), 1))
        else:
            q_goal_arr = q_goal
        error_deg = _compute_quaternion_error(quat, q_goal_arr)
        ax.plot(time, error_deg, 'b-', linewidth=1.5, label='Track')
        if has_traj:
            if q_goal.ndim == 1:
                traj_q_goal_arr = np.tile(q_goal, (len(traj_time), 1))
            else:
                # Interpolate goal to trajectory time
                traj_q_goal_arr = np.tile(q_goal[0], (len(traj_time), 1))
            traj_error_deg = _compute_quaternion_error(traj_quat, traj_q_goal_arr)
            ax.plot(traj_time, traj_error_deg, 'b--', linewidth=1.0, alpha=0.7, label='Plan')
        ax.set_ylabel("Quaternion Error [°]")
        ax.set_title(f"{title_prefix}: Attitude Error")
    elif has_boresight_goal:
        goal = result["boresight_goal"]
        v_bore_body = body_boresight / np.linalg.norm(body_boresight)
        R_b2i = _rot_mat_vec(quat)
        v_bore_eci = np.einsum('nij,j->ni', R_b2i, v_bore_body)
        v_bore_eci_norm = np.linalg.norm(v_bore_eci, axis=1, keepdims=True)
        v_goal_norm = np.linalg.norm(goal, axis=1, keepdims=True)
        # No_Goal periods have zero goal vector — set to NaN to avoid div-by-zero
        no_goal_mask = v_goal_norm.ravel() < 1e-10
        v_goal_safe = goal.copy()
        v_goal_safe[no_goal_mask] = 1.0  # placeholder to avoid div-by-zero
        v_b = v_bore_eci / v_bore_eci_norm
        v_g = v_goal_safe / np.linalg.norm(v_goal_safe, axis=1, keepdims=True)
        dot_prod = np.sum(v_b * v_g, axis=1)
        dot_prod = np.clip(dot_prod, -1.0, 1.0)
        error_deg = np.rad2deg(np.arccos(dot_prod))
        error_deg[no_goal_mask] = np.nan
        ax.plot(time, error_deg, 'b-', linewidth=1.5, label='Track')
        if has_traj:
            # Interpolate per-timestep goal to trajectory times
            from scipy.interpolate import interp1d
            goal_interp = interp1d(time, goal, axis=0, kind='nearest',
                                   bounds_error=False, fill_value='extrapolate')
            traj_goal = goal_interp(traj_time)
            traj_goal_norm = np.linalg.norm(traj_goal, axis=1, keepdims=True)
            traj_no_goal = traj_goal_norm.ravel() < 1e-10
            traj_goal_safe = traj_goal.copy()
            traj_goal_safe[traj_no_goal] = 1.0
            traj_R_b2i = _rot_mat_vec(traj_quat)
            traj_v_bore_eci = np.einsum('nij,j->ni', traj_R_b2i, v_bore_body)
            traj_v_bore_eci_norm = np.linalg.norm(traj_v_bore_eci, axis=1, keepdims=True)
            traj_v_b = traj_v_bore_eci / traj_v_bore_eci_norm
            traj_v_g = traj_goal_safe / np.linalg.norm(traj_goal_safe, axis=1, keepdims=True)
            traj_dot_prod = np.sum(traj_v_b * traj_v_g, axis=1)
            traj_dot_prod = np.clip(traj_dot_prod, -1.0, 1.0)
            traj_error_deg = np.rad2deg(np.arccos(traj_dot_prod))
            traj_error_deg[traj_no_goal] = np.nan
            ax.plot(traj_time, traj_error_deg, 'b--', linewidth=1.0, alpha=0.7, label='Plan')
        ax.set_ylabel("Boresight Error [°]")
        ax.set_title(f"{title_prefix}: Tracking Error")
    else:
        ax.text(0.5, 0.5, "No goal data", ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel("Time [s]")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='1°')
    ax.axhline(y=5.0, color='orange', linestyle='--', alpha=0.5, label='5°')
    ax.legend(fontsize='small')

    # 5. Quaternion components
    ax = axes[1, 1]
    ax.plot(time, quat[:, 0], 'k-', label='q_w (track)', linewidth=1.5)
    ax.plot(time, quat[:, 1], 'r-', label='q_x (track)', linewidth=1.5)
    ax.plot(time, quat[:, 2], 'g-', label='q_y (track)', linewidth=1.5)
    ax.plot(time, quat[:, 3], 'b-', label='q_z (track)', linewidth=1.5)
    if has_traj:
        ax.plot(traj_time, traj_quat[:, 0], 'k--', alpha=0.7, linewidth=1.0, label='q_w (plan)')
        ax.plot(traj_time, traj_quat[:, 1], 'r--', alpha=0.7, linewidth=1.0, label='q_x (plan)')
        ax.plot(traj_time, traj_quat[:, 2], 'g--', alpha=0.7, linewidth=1.0, label='q_y (plan)')
        ax.plot(traj_time, traj_quat[:, 3], 'b--', alpha=0.7, linewidth=1.0, label='q_z (plan)')
    if has_quat_goal:
        q_g = result["q_goal"]
        if q_g.ndim == 1:
            ax.axhline(y=q_g[0], color='k', linestyle=':', alpha=0.5)
            ax.axhline(y=q_g[1], color='r', linestyle=':', alpha=0.5)
            ax.axhline(y=q_g[2], color='g', linestyle=':', alpha=0.5)
            ax.axhline(y=q_g[3], color='b', linestyle=':', alpha=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Quaternion")
    ax.set_title(f"{title_prefix}: Quaternion Components")
    ax.legend(fontsize='x-small', ncol=2)
    ax.grid(True, linestyle='--', alpha=0.6)

    # 6. Stored momentum (if RW present) or angular velocity magnitude (fallback)
    ax = axes[1, 2]
    if h_rw is not None and h_rw.shape[1] > 0:
        for i in range(h_rw.shape[1]):
            ax.plot(time, h_rw[:, i], label=f'h_RW{i} (track)', linewidth=1.5)
        if has_traj and traj_h_rw is not None and traj_h_rw.shape[1] > 0:
            for i in range(traj_h_rw.shape[1]):
                ax.plot(traj_time, traj_h_rw[:, i], linestyle='--',
                       label=f'h_RW{i} (plan)', alpha=0.7, linewidth=1.0)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Stored Momentum [N·m·s]")
        ax.set_title(f"{title_prefix}: RW Stored Momentum")
        ax.legend(fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.6)
    else:
        omega_mag = np.linalg.norm(omega_deg, axis=1)
        ax.plot(time, omega_mag, 'b-', linewidth=1.5, label='Track')
        if has_traj:
            traj_omega_mag = np.linalg.norm(np.rad2deg(traj_omega), axis=1)
            ax.plot(traj_time, traj_omega_mag, 'b--', linewidth=1.0, alpha=0.7, label='Plan')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("|ω| [°/s]")
        ax.set_title(f"{title_prefix}: Angular Velocity Magnitude")
        ax.legend(fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_planned_trajectory(
    traj,
    config: Dict[str, Any],
    body_boresight: np.ndarray,
    title_prefix: str = "Planned Trajectory",
    save_path: str = "/tmp/planned_trajectory.png",
    goals=None,
) -> None:
    """
    Plot the planned trajectory non-blocking before the tracking simulation starts.

    Auto-detects goal type from config keys and adapts subplot 4 to show
    RW stored momentum (if present) or angular velocity magnitude (fallback).

    Parameters
    ----------
    traj : Trajectory
        Trajectory object with .states, .controls, .times attributes.
    config : Dict[str, Any]
        Simulation config with goal information.
    body_boresight : np.ndarray
        Boresight direction in body frame.
    title_prefix : str
        Title prefix for the figure.
    save_path : str
        Path to save the figure.
    goals : GoalList, optional
        Time-varying goal list. When provided, computes boresight error against
        the active goal at each timestep (NaN during No_Goal periods).
    """
    from ADCS.CONOPS.goals import No_Goal

    Xset = traj.states     # (n_state, N)
    Uset = traj.controls   # (n_ctrl, N-1)
    times = traj.times
    N_traj = Xset.shape[1]
    traj_times = (times - times[0]) * 36525 * 24 * 3600  # J2000 centuries to seconds

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Angular velocity
    omega_deg = np.degrees(Xset[0:3, :])
    for i, label in enumerate([r'$\omega_x$', r'$\omega_y$', r'$\omega_z$']):
        axes[0, 0].plot(traj_times, omega_deg[i, :], label=label)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Angular Velocity (°/s)')
    axes[0, 0].set_title('Planned Angular Velocity')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. Angle from goal (auto-detect type)
    body_bore = body_boresight / np.linalg.norm(body_boresight)
    if "q_goal" in config:
        q_goal = np.asarray(config["q_goal"], dtype=float)
        angles_deg = np.zeros(N_traj)
        for k in range(N_traj):
            dot = np.abs(np.dot(Xset[3:7, k], q_goal))
            angles_deg[k] = np.degrees(2 * np.arccos(np.clip(dot, 0, 1)))
        angle_label = "Quaternion Error"
    elif goals is not None:
        # Time-varying goals: compute error against active goal at each timestep
        angles_deg = np.full(N_traj, np.nan)
        for k in range(N_traj):
            active_goal = goals.get_active_goal(times[k])
            if isinstance(active_goal, No_Goal):
                continue  # NaN gap during No_Goal periods
            goal_vec = active_goal.eci_vector
            goal_norm = goal_vec / np.linalg.norm(goal_vec)
            w, x, y, z = Xset[3:7, k]
            R = np.array([
                [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
            ])
            bore_eci = R @ body_bore
            dot = np.clip(np.dot(bore_eci, goal_norm), -1.0, 1.0)
            angles_deg[k] = np.degrees(np.arccos(dot))
        # Mark goal transitions with vertical lines
        t0 = times[0]
        sec2cent = 1.0 / (36525 * 24 * 3600)
        for i, gt in enumerate(goals.times):
            t_sec = (gt - t0) / sec2cent  # Convert to seconds from start
            if 0 < t_sec < traj_times[-1]:
                goal_obj = goals.goals[i]
                color = '0.5' if isinstance(goal_obj, No_Goal) else 'green'
                style = ':' if isinstance(goal_obj, No_Goal) else '--'
                axes[0, 1].axvline(x=t_sec, color=color, linestyle=style, alpha=0.6)
        angle_label = "Boresight Error"
    else:
        if "goal_eci_vec" in config:
            goal_vec = np.asarray(config["goal_eci_vec"], dtype=float)
        elif "goal1" in config:
            goal_vec = np.asarray(config["goal1"], dtype=float)
        else:
            goal_vec = np.array([0.0, 0.0, 1.0])
        goal_norm = goal_vec / np.linalg.norm(goal_vec)
        angles_deg = np.zeros(N_traj)
        for k in range(N_traj):
            w, x, y, z = Xset[3:7, k]
            R = np.array([
                [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
            ])
            bore_eci = R @ body_bore
            dot = np.clip(np.dot(bore_eci, goal_norm), -1.0, 1.0)
            angles_deg[k] = np.degrees(np.arccos(dot))
        angle_label = "Boresight Error"

    axes[0, 1].plot(traj_times, angles_deg)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel(f'{angle_label} (°)')
    axes[0, 1].set_title(f'Planned {angle_label}')
    axes[0, 1].grid(True)
    axes[0, 1].axhline(y=0, color='g', linestyle='--', alpha=0.5)

    # 3. Controls
    ctrl_times = traj_times[:Uset.shape[1]]
    n_ctrl = Uset.shape[0]
    if n_ctrl == 4:
        ctrl_labels = ['MTQ_x', 'MTQ_y', 'MTQ_z', 'RW']
    elif n_ctrl == 3:
        ctrl_labels = ['MTQ_x', 'MTQ_y', 'MTQ_z']
    else:
        ctrl_labels = [f'u{i}' for i in range(n_ctrl)]
    for i in range(n_ctrl):
        axes[1, 0].plot(ctrl_times, Uset[i, :], label=ctrl_labels[i])
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Control')
    axes[1, 0].set_title('Planned Controls')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 4. Stored momentum (if RW present) or angular velocity magnitude
    n_state = Xset.shape[0]
    if n_state > 7:
        h_rw = Xset[7:, :]
        for i in range(h_rw.shape[0]):
            axes[1, 1].plot(traj_times, h_rw[i, :], label=f'h_RW{i}')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Stored Momentum [N·m·s]')
        axes[1, 1].set_title('Planned RW Stored Momentum')
        axes[1, 1].legend()
    else:
        omega_mag = np.linalg.norm(omega_deg, axis=0)
        axes[1, 1].plot(traj_times, omega_mag)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('|ω| (°/s)')
        axes[1, 1].set_title('Planned Angular Velocity Magnitude')
    axes[1, 1].grid(True)

    # Handle NaN from No_Goal periods in suptitle
    start_angle = angles_deg[0] if not np.isnan(angles_deg[0]) else np.nanmin(angles_deg[:10]) if np.any(~np.isnan(angles_deg[:10])) else float('nan')
    end_angle = angles_deg[-1] if not np.isnan(angles_deg[-1]) else np.nanmin(angles_deg[-10:]) if np.any(~np.isnan(angles_deg[-10:])) else float('nan')
    plt.suptitle(f'{title_prefix}\nStart: {start_angle:.1f}°  End: {end_angle:.1f}°')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved planned trajectory plot to {save_path}")
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show(block=False)
    plt.pause(0.5)


def create_planner_diagnostic_callback(
    config: Dict[str, Any],
    body_boresight: np.ndarray,
    tf: float,
) -> callable:
    """
    Create a diagnostic callback for PythonALILQR planner visualization.

    Auto-detects goal type from config keys:
    - "q_goal" -> quaternion error
    - "goal_eci_vec" -> boresight error
    - "goal1" -> boresight error (multi-goal, uses first goal)

    Parameters
    ----------
    config : Dict[str, Any]
        Simulation config with goal information
    body_boresight : np.ndarray
        Boresight direction in body frame
    tf : float
        Total simulation time in seconds

    Returns
    -------
    callable
        Callback function compatible with PythonALILQR.set_iteration_callback()
    """
    iter_count = [0]

    # Determine angle computation method based on goal type
    if "q_goal" in config:
        q_goal = np.asarray(config["q_goal"], dtype=float)
        q_goal_inv = np.array([q_goal[0], -q_goal[1], -q_goal[2], -q_goal[3]])
        error_label = "Quaternion Error"

        def compute_angles(Xset):
            N_pts = Xset.shape[1]
            angles = np.zeros(N_pts)
            for k in range(N_pts):
                qk = Xset[3:7, k]
                qerr_w = q_goal_inv[0]*qk[0] - np.dot(q_goal_inv[1:], qk[1:])
                angles[k] = np.degrees(2 * np.arccos(np.clip(np.abs(qerr_w), 0, 1)))
            return angles
    else:
        if "goal_eci_vec" in config:
            goal_vec = np.asarray(config["goal_eci_vec"], dtype=float)
        elif "goal1" in config:
            goal_vec = np.asarray(config["goal1"], dtype=float)
        else:
            goal_vec = np.array([0.0, 0.0, 1.0])
        goal_norm = goal_vec / np.linalg.norm(goal_vec)
        body_bore = body_boresight / np.linalg.norm(body_boresight)
        error_label = "Boresight Error"

        def compute_angles(Xset):
            N_pts = Xset.shape[1]
            angles = np.zeros(N_pts)
            for k in range(N_pts):
                w, x, y, z = Xset[3:7, k]
                R = np.array([
                    [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
                    [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
                    [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
                ])
                bore_eci = R @ body_bore
                dot = np.clip(np.dot(bore_eci, goal_norm), -1.0, 1.0)
                angles[k] = np.degrees(np.arccos(dot))
            return angles

    def callback(iter_data):
        Xset = iter_data.Xset
        N_pts = Xset.shape[1]
        angles = compute_angles(Xset)

        max_angle = np.max(angles)
        max_idx = np.argmax(angles)
        half_N = N_pts // 2
        second_half_angles = angles[half_N:]
        max_2nd_half = np.max(second_half_angles)
        max_2nd_idx = half_N + np.argmax(second_half_angles)

        print(f"  [{iter_data.pass_label}] O:{iter_data.outer_iter} I:{iter_data.inner_iter} "
              f"Cost:{iter_data.LA:.2e} Cmax:{iter_data.cmax:.2e} rho:{iter_data.rho:.1e} "
              f"Angle[start:{angles[0]:.0f}\u00b0 max:{max_angle:.0f}\u00b0@{max_idx} "
              f"spike:{max_2nd_half:.0f}\u00b0@{max_2nd_idx} mean:{np.mean(angles):.0f}\u00b0 "
              f"end:{angles[-1]:.0f}\u00b0]")

        iter_count[0] += 1
        save_iters = [1, 5, 10, 20, 34, 50, 70, 100, 150, 200, 250, 300]
        if iter_count[0] in save_iters:
            fig_diag, ax_diag = plt.subplots(2, 2, figsize=(12, 8))
            times = np.arange(N_pts) * (tf / N_pts)

            ax_diag[0, 0].plot(times, angles, 'b-', linewidth=1.5)
            ax_diag[0, 0].axhline(90, color='r', linestyle='--', alpha=0.5, label='90\u00b0')
            ax_diag[0, 0].set_xlabel('Time (s)')
            ax_diag[0, 0].set_ylabel(f'{error_label} (deg)')
            ax_diag[0, 0].set_title(f'{error_label} - Iter {iter_count[0]} [{iter_data.pass_label}]')
            ax_diag[0, 0].set_ylim(0, 200)
            ax_diag[0, 0].grid(True)
            ax_diag[0, 0].legend()

            ax_diag[0, 1].plot(times, Xset[3, :], label='q0')
            ax_diag[0, 1].plot(times, Xset[4, :], label='q1')
            ax_diag[0, 1].plot(times, Xset[5, :], label='q2')
            ax_diag[0, 1].plot(times, Xset[6, :], label='q3')
            ax_diag[0, 1].set_xlabel('Time (s)')
            ax_diag[0, 1].set_ylabel('Quaternion')
            ax_diag[0, 1].set_title('Quaternion Components')
            ax_diag[0, 1].legend()
            ax_diag[0, 1].grid(True)

            ax_diag[1, 0].plot(times, np.degrees(Xset[0, :]), label='\u03c9x')
            ax_diag[1, 0].plot(times, np.degrees(Xset[1, :]), label='\u03c9y')
            ax_diag[1, 0].plot(times, np.degrees(Xset[2, :]), label='\u03c9z')
            ax_diag[1, 0].set_xlabel('Time (s)')
            ax_diag[1, 0].set_ylabel('Angular Velocity (deg/s)')
            ax_diag[1, 0].set_title('Angular Velocity')
            ax_diag[1, 0].legend()
            ax_diag[1, 0].grid(True)

            Uset = iter_data.Uset
            ctrl_times = times[:Uset.shape[1]]
            for i in range(Uset.shape[0]):
                ax_diag[1, 1].plot(ctrl_times, Uset[i, :], label=f'u{i}')
            ax_diag[1, 1].set_xlabel('Time (s)')
            ax_diag[1, 1].set_ylabel('Control')
            ax_diag[1, 1].set_title('Control Inputs')
            ax_diag[1, 1].legend()
            ax_diag[1, 1].grid(True)

            plt.tight_layout()
            plt.savefig(f'/tmp/planner_iter_{iter_count[0]:03d}.png', dpi=100)
            plt.close(fig_diag)
            print(f"    -> Saved /tmp/planner_iter_{iter_count[0]:03d}.png")

    return callback


def plot_quaternion_error_mc(
    full_results: List[Dict[str, Any]],
    title: str = "Monte Carlo Quaternion Error",
    alpha: float = 0.5,
) -> None:
    """
    Plot quaternion attitude error traces for multiple Monte Carlo runs.

    Parameters
    ----------
    full_results : List[Dict[str, Any]]
        List of MC result dictionaries with "time", "state", and "q_goal"
    title : str
        Plot title
    alpha : float
        Transparency for individual traces
    """
    if not full_results:
        print("[plot_quaternion_error_mc] Warning: No results to plot.")
        return

    plt.figure(figsize=(10, 6))

    for res in full_results:
        if "state" not in res or "q_goal" not in res or "time" not in res:
            continue

        time = res["time"]
        state = res["state"]
        q_goal = res["q_goal"]

        quat = state[:, 3:7]

        # Handle constant vs time-varying goal
        if q_goal.ndim == 1:
            q_goal_arr = np.tile(q_goal, (len(time), 1))
        else:
            q_goal_arr = q_goal

        error_deg = _compute_quaternion_error(quat, q_goal_arr)
        plt.plot(time, error_deg, color='tab:blue', alpha=alpha, linewidth=1.0)

    # Add dummy line for legend
    plt.plot([], [], color='tab:blue', label='MC Runs')

    plt.xlabel("Time [s]")
    plt.ylabel("Quaternion Error [°]")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()


def plot_quaternion_histogram_mc(
    full_results: List[Dict[str, Any]],
    title: str = "Monte Carlo Final Quaternion Error",
    bin_width_deg: float = 5.0,
    under_thresh_deg: float = 1.0,
    show_stats_box: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Plot histogram of final quaternion errors for MC runs.

    Parameters
    ----------
    full_results : List[Dict[str, Any]]
        List of MC result dictionaries
    title : str
        Plot title
    bin_width_deg : float
        Histogram bin width in degrees
    under_thresh_deg : float
        Threshold for computing % converged
    show_stats_box : bool
        Whether to show statistics box

    Returns
    -------
    Tuple[np.ndarray, Dict[str, float]]
        Array of final errors and statistics dictionary
    """
    if not full_results:
        print("[plot_quaternion_histogram_mc] Warning: No results to plot.")
        return np.array([]), {"pct_under_thresh": np.nan, "min": np.nan, "max": np.nan,
                             "mean": np.nan, "median": np.nan, "n": 0.0}

    errors = []

    for res in full_results:
        if "state" not in res or "q_goal" not in res:
            continue

        state = res["state"]
        q_goal = res["q_goal"]

        if state is None or len(state) == 0:
            continue

        # Final quaternion
        q_final = state[-1, 3:7]

        # Goal quaternion (handle time-varying)
        if q_goal.ndim == 1:
            q_g = q_goal
        else:
            q_g = q_goal[-1]

        err = _compute_quaternion_error(q_final, q_g)
        errors.append(float(err))

    errors_deg = np.asarray(errors, dtype=float)

    if errors_deg.size == 0:
        print("[plot_quaternion_histogram_mc] Warning: No valid runs.")
        return errors_deg, {"pct_under_thresh": np.nan, "min": np.nan, "max": np.nan,
                           "mean": np.nan, "median": np.nan, "n": 0.0}

    # Statistics
    pct_under = 100.0 * np.mean(errors_deg < under_thresh_deg)
    stats = {
        "pct_under_thresh": float(pct_under),
        "min": float(np.min(errors_deg)),
        "max": float(np.max(errors_deg)),
        "mean": float(np.mean(errors_deg)),
        "median": float(np.median(errors_deg)),
        "n": float(errors_deg.size),
    }

    # Histogram
    max_edge = max(np.ceil(errors_deg.max() / bin_width_deg) * bin_width_deg, bin_width_deg)
    bins = np.arange(0.0, max_edge + bin_width_deg, bin_width_deg)

    plt.figure(figsize=(10, 6))
    plt.hist(errors_deg, bins=bins, edgecolor="black")
    plt.xlabel("Final Quaternion Error [°]")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)

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


def plot_mc_summary(
    full_results: List[Dict[str, Any]],
    body_boresight: np.ndarray = np.array([0.0, 1.0, 0.0]),
    title_prefix: str = "MC",
    show: bool = False,
) -> plt.Figure:
    """
    Plot comprehensive MC summary: error traces and histogram.

    Automatically detects goal type (quaternion or boresight) and uses
    appropriate plotting functions.

    Parameters
    ----------
    full_results : List[Dict[str, Any]]
        List of MC result dictionaries
    body_boresight : np.ndarray
        Boresight direction in body frame (for boresight goals)
    title_prefix : str
        Prefix for plot titles
    show : bool
        If True, call plt.show() at the end

    Returns
    -------
    plt.Figure
        The matplotlib figure object (for the time series plot)
    """
    if not full_results:
        print("[plot_mc_summary] Warning: No results to plot.")
        return None

    # Detect goal type from first valid result
    goal_type = None
    for res in full_results:
        if "q_goal" in res:
            goal_type = "full_attitude"
            break
        elif "boresight_goal" in res:
            goal_type = "reduced_attitude"
            break

    if goal_type is None:
        print("[plot_mc_summary] Warning: Could not detect goal type.")
        return None

    # Plot based on goal type
    if goal_type == "full_attitude":
        plot_quaternion_error_mc(full_results, title=f"{title_prefix}: Quaternion Error Traces")
        fig = plt.gcf()
        plot_quaternion_histogram_mc(full_results, title=f"{title_prefix}: Final Error Distribution")
    else:
        plot_target_tracking_mc(full_results, body_boresight=body_boresight,
                               title=f"{title_prefix}: Boresight Error Traces")
        fig = plt.gcf()
        plot_convergence_histogram_mc(full_results, body_boresight=body_boresight,
                                      title=f"{title_prefix}: Final Error Distribution")

    if show:
        plt.show()

    return fig