__all__ = [
    "plot_h_tracking_mc_compare",
    "plot_target_tracking_mc_compare",
    "plot_convergence_histogram_mc_compare",
]

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import matplotlib.cm as cm

def _rot_mat_vec(q: np.ndarray) -> np.ndarray:
    """
    Vectorized conversion of Scalar-First Quaternions (w, x, y, z) 
    to Rotation Matrices (Body -> Inertial).
    
    Input: q shape (N, 4) OR (4,)
    Output: R shape (N, 3, 3) OR (3, 3)
    """
    q = np.asarray(q)
    
    # Handle single quaternion input case: (4,) -> (1, 4)
    if q.ndim == 1:
        q_in = q[np.newaxis, :]
        is_single = True
    else:
        q_in = q
        is_single = False
        
    w, x, y, z = q_in[:, 0], q_in[:, 1], q_in[:, 2], q_in[:, 3]
    
    # Formula for Rotation Matrix from Quaternion (Hamilton/Scalar First)
    R = np.empty((q_in.shape[0], 3, 3))
    
    R[:, 0, 0] = 1 - 2*(y**2 + z**2)
    R[:, 0, 1] = 2*(x*y - z*w)
    R[:, 0, 2] = 2*(x*z + y*w)
    
    R[:, 1, 0] = 2*(x*y + z*w)
    R[:, 1, 1] = 1 - 2*(x**2 + z**2)
    R[:, 1, 2] = 2*(y*z - x*w)
    
    R[:, 2, 0] = 2*(x*z - y*w)
    R[:, 2, 1] = 2*(y*z + x*w)
    R[:, 2, 2] = 1 - 2*(x**2 + y**2)
    
    # If input was (4,), return (3, 3) instead of (1, 3, 3)
    if is_single:
        return R[0]
        
    return R

def plot_h_tracking_mc_compare(
    full_results_A: List[Dict[str, Any]],
    full_results_B: List[Dict[str, Any]],
    title: str = "Monte Carlo Stored Angular Momentum (Comparison)",
    label_A: str = "Case A",
    label_B: str = "Case B",
) -> None:
    r"""
    Compare stored reaction wheel angular momentum across two Monte Carlo cases.

    This function overlays stored angular momentum time histories from two sets
    of Monte Carlo simulation results, enabling qualitative comparison of
    momentum buildup, dispersion, and actuator usage.

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

    * Each Monte Carlo run is plotted with low opacity
    * Each wheel component uses a consistent color within a case
    * Separate colormaps are used for Case A and Case B

    :param full_results_A:
        Monte Carlo result set for Case A.
    :type full_results_A:
        list of dict

    :param full_results_B:
        Monte Carlo result set for Case B.
    :type full_results_B:
        list of dict

    :param title:
        Plot title.
    :type title:
        str

    :param label_A:
        Legend label for Case A.
    :type label_A:
        str

    :param label_B:
        Legend label for Case B.
    :type label_B:
        str

    :return:
        None. The function generates a Matplotlib figure.
    :rtype:
        None

    """

    if not full_results_A and not full_results_B:
        print("[plot_h_tracking_mc_compare] Warning: No results to plot.")
        return

    plt.figure(figsize=(10, 6))

    # Color maps
    cmap_A = cm.get_cmap("tab10")
    cmap_B = cm.get_cmap("Dark2")

    def _plot_set(results, cmap, alpha, label_prefix):
        colors = None
        color_num = None

        for res in results:
            if "state" not in res or "time" not in res:
                continue

            state = np.asarray(res["state"])
            time = np.asarray(res["time"])

            if state.shape[1] <= 7:
                continue

            h_hist = state[:, 7:]

            if colors is None:
                color_num = h_hist.shape[1]
                colors = [cmap(i / max(color_num - 1, 1)) for i in range(color_num)]

            for j in range(color_num):
                plt.plot(
                    time,
                    h_hist[:, j],
                    color=colors[j],
                    alpha=alpha,
                    linewidth=1.5,
                )

        # Legend handles
        if colors is not None:
            for j in range(color_num):
                plt.plot(
                    [],
                    [],
                    color=colors[j],
                    label=f"{label_prefix} RW {j}",
                )

    _plot_set(full_results_A, cmap_A, alpha=0.15, label_prefix=label_A)
    _plot_set(full_results_B, cmap_B, alpha=0.15, label_prefix=label_B)

    plt.xlabel("Time [s]")
    plt.ylabel("Stored Angular Momentum")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()


def plot_target_tracking_mc_compare(
    full_results_A: List[Dict[str, Any]],
    full_results_B: List[Dict[str, Any]],
    body_boresight: np.ndarray = np.array([0.0, 0.0, 1.0]),
    title: str = "Monte Carlo Target Tracking Error (Comparison)",
    label_A: str = "Case A",
    label_B: str = "Case B",
) -> None:
    r"""
    Compare target tracking angular error histories for two Monte Carlo cases.

    This function computes and overlays the instantaneous angular tracking
    error between a body-fixed boresight and an inertial target direction for
    two sets of Monte Carlo simulations.

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

    The angular tracking error is

    .. math::

        \theta_i =
        \cos^{-1}\!\left(
        \hat{\mathbf{b}}_i^{\text{ECI}} \cdot \hat{\mathbf{g}}_i
        \right)

    ======================
    Visualization Strategy
    ======================

    * Each Monte Carlo run is plotted with low opacity
    * Case A and Case B are distinguished by color

    :param full_results_A:
        Monte Carlo result set for Case A.
    :type full_results_A:
        list of dict

    :param full_results_B:
        Monte Carlo result set for Case B.
    :type full_results_B:
        list of dict

    :param body_boresight:
        Fixed boresight direction expressed in the body frame.
    :type body_boresight:
        numpy.ndarray

    :param title:
        Plot title.
    :type title:
        str

    :param label_A:
        Legend label for Case A.
    :type label_A:
        str

    :param label_B:
        Legend label for Case B.
    :type label_B:
        str

    :return:
        None. The function generates a Matplotlib figure.
    :rtype:
        None

    """

    v_bore_body = np.asarray(body_boresight, dtype=float)
    v_bore_body /= np.linalg.norm(v_bore_body)

    plt.figure(figsize=(10, 6))

    def _plot_set(results, color, alpha):
        for res in results:
            if (
                "state" not in res
                or "boresight_goal" not in res
                or "time" not in res
            ):
                continue

            state = np.asarray(res["state"])
            goal = np.asarray(res["boresight_goal"])
            time = np.asarray(res["time"])

            N = min(len(state), len(goal), len(time))
            if N == 0:
                continue

            q_hist = state[:N, 3:7]

            # Rotate boresight using _rot_mat_vec
            v_bore_eci = np.zeros((N, 3))
            for k in range(N):
                R_b2i = _rot_mat_vec(q_hist[k])
                v_bore_eci[k] = R_b2i @ v_bore_body

            # Normalize
            v_b = v_bore_eci / np.linalg.norm(v_bore_eci, axis=1, keepdims=True)
            v_g = goal[:N] / np.linalg.norm(goal[:N], axis=1, keepdims=True)

            dot = np.sum(v_b * v_g, axis=1)
            dot = np.clip(dot, -1.0, 1.0)

            err_deg = np.rad2deg(np.arccos(dot))

            plt.plot(time[:N], err_deg, color=color, alpha=alpha, linewidth=1.5)

    _plot_set(full_results_A, color="tab:blue", alpha=0.15)
    _plot_set(full_results_B, color="tab:orange", alpha=0.15)

    plt.plot([], [], color="tab:blue", label=label_A)
    plt.plot([], [], color="tab:orange", label=label_B)

    plt.xlabel("Time [s]")
    plt.ylabel("Tracking Error [deg]")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()


def plot_convergence_histogram_mc_compare(
    full_results_A: List[Dict[str, Any]],
    full_results_B: List[Dict[str, Any]],
    body_boresight: np.ndarray = np.array([0.0, 0.0, 1.0]),
    title: str = "Final Tracking Error Distribution (Comparison)",
    bin_width_deg: float = 5.0,
    under_thresh_deg: float = 1.0,
    label_A: str = "Case A",
    label_B: str = "Case B",
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Compare final-timestep tracking error distributions for two Monte Carlo cases.

    This function computes the final angular tracking error for each Monte Carlo
    run and visualizes the resulting distributions using overlaid histograms.
    Summary statistics are displayed directly on the plot.

    ======================
    Final Error Definition
    ======================

    For each Monte Carlo run, the final tracking error is computed as

    .. math::

        \theta_f =
        \cos^{-1}\!\left(
        \hat{\mathbf{b}}_f^{\text{ECI}} \cdot \hat{\mathbf{g}}_f
        \right)

    where:

    * :math:`\hat{\mathbf{b}}_f^{\text{ECI}}` is the final boresight direction
    * :math:`\hat{\mathbf{g}}_f` is the final target direction

    ======================
    Histogram Construction
    ======================

    * Bin width is user-defined in degrees
    * Overlapping histograms are shown with transparency
    * A statistics box reports:
        - Sample count
        - Percentage below a threshold
        - Mean and maximum error

    ======================
    Return Values
    ======================

    The raw error arrays are returned to support further statistical analysis.

    :param full_results_A:
        Monte Carlo result set for Case A.
    :type full_results_A:
        list of dict

    :param full_results_B:
        Monte Carlo result set for Case B.
    :type full_results_B:
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
        Histogram bin width in degrees.
    :type bin_width_deg:
        float

    :param under_thresh_deg:
        Threshold angle (degrees) used for convergence statistics.
    :type under_thresh_deg:
        float

    :param label_A:
        Legend label for Case A.
    :type label_A:
        str

    :param label_B:
        Legend label for Case B.
    :type label_B:
        str

    :return:
        Tuple containing final tracking error arrays for Case A and Case B.
    :rtype:
        tuple of numpy.ndarray

    """

    v_bore_body = np.asarray(body_boresight, dtype=float)
    v_bore_body /= np.linalg.norm(v_bore_body)

    def _extract_errors(results):
        errors = []
        for res in results:
            if "state" not in res or "boresight_goal" not in res:
                continue

            state = np.asarray(res["state"])
            goal = np.asarray(res["boresight_goal"])

            N = min(len(state), len(goal))
            if N == 0:
                continue

            # Extract final quaternion
            q = state[N - 1, 3:7]
            
            # Robust call: reshape to (1,4) to satisfy vectorized helper, then take [0]
            R_b2i = _rot_mat_vec(q.reshape(1, 4))[0]

            v_b = R_b2i @ v_bore_body
            v_g = goal[N - 1]

            nb = np.linalg.norm(v_b)
            ng = np.linalg.norm(v_g)
            if nb == 0 or ng == 0:
                continue

            v_b /= nb
            v_g /= ng

            dot = np.clip(np.dot(v_b, v_g), -1.0, 1.0)
            err = np.rad2deg(np.arccos(dot))
            errors.append(err)

        return np.asarray(errors)

    def _get_stats_str(name, errs, thresh):
        if errs.size == 0:
            return f"{name}: No Data"
        
        pct_under = 100.0 * np.mean(errs < thresh)
        return (
            f"{name} (N={errs.size})\n"
            f"  < {thresh:.1f}°: {pct_under:.1f}%\n"
            f"  Mean: {np.mean(errs):.2f}°\n"
            f"  Max:  {np.max(errs):.2f}°"
        )

    # 1. Extract Errors
    err_A = _extract_errors(full_results_A)
    err_B = _extract_errors(full_results_B)

    # 2. Determine Histogram Bins
    max_err = max(
        err_A.max() if err_A.size else 0.0,
        err_B.max() if err_B.size else 0.0,
    )
    max_edge = np.ceil(max_err / bin_width_deg) * bin_width_deg
    # Ensure at least one bin if everything is 0
    if max_edge == 0: max_edge = bin_width_deg 
    bins = np.arange(0.0, max_edge + bin_width_deg, bin_width_deg)

    # 3. Plot
    plt.figure(figsize=(10, 6)) # Slightly wider to accommodate text
    
    # Use 'stepfilled' or 'bar' with transparency for overlapping histograms
    plt.hist(err_A, bins=bins, alpha=0.5, label=label_A, edgecolor="black", linewidth=1)
    plt.hist(err_B, bins=bins, alpha=0.5, label=label_B, edgecolor="black", linewidth=1, linestyle='--')

    plt.xlabel("Final Tracking Error [deg]")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc='upper left')

    # 4. Add Stats Box
    stats_str_A = _get_stats_str(label_A, err_A, under_thresh_deg)
    stats_str_B = _get_stats_str(label_B, err_B, under_thresh_deg)
    full_stats_txt = f"{stats_str_A}\n\n{stats_str_B}"

    plt.gca().text(
        0.98, 0.98, full_stats_txt,
        transform=plt.gca().transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
    )

    plt.tight_layout()

    return err_A, err_B
