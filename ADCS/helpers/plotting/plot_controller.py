__all__ = [
    "plot_control",
    "plot_rw_momentum",
    "plot_target_tracking",
]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons
from numpy.linalg import norm
from typing import Callable, Tuple, List
from ADCS.helpers.math_helpers import quat_to_euler
from ADCS.state import State

def plot_control(
    time: np.ndarray,
    u_hist: np.ndarray,
    title: str = "Control Effort Over Time",
    units: str = "Command"
) -> None:
    r"""
    Plot time histories of spacecraft control commands.

    This function visualizes actuator or control-command outputs as a function
    of time. Each control channel is plotted as a separate curve, enabling
    direct comparison of magnitudes, trends, and relative activity among
    actuators.

    ======================
    Mathematical Context
    ======================

    Let the control input vector be

    .. math::

        \mathbf{u}(t) =
        \begin{bmatrix}
        u_1(t) & u_2(t) & \dots & u_M(t)
        \end{bmatrix}^T

    where :math:`M` is the number of control channels (e.g., reaction wheels,
    magnetorquers, thrusters).

    Given discrete time samples :math:`t_i`, the function plots

    .. math::

        u_j(t_i), \qquad i = 1, \dots, N,\; j = 1, \dots, M

    on a shared set of axes.

    ======================
    Visualization Details
    ======================

    * Each control channel is labeled as :math:`u_1, u_2, \dots, u_M`
    * The y-axis units are user-configurable
    * A grid and legend are automatically enabled

    :param time:
        One-dimensional array of time stamps in seconds.
    :type time:
        numpy.ndarray

    :param u_hist:
        Control command history. Each column corresponds to one control channel.
    :type u_hist:
        numpy.ndarray

    :param title:
        Title displayed at the top of the plot.
    :type title:
        str

    :param units:
        Units of the control command shown on the y-axis.
    :type units:
        str

    :return:
        None. The function creates a Matplotlib figure.
    :rtype:
        None

    """
    time = np.asarray(time)
    u_hist = np.asarray(u_hist)

    if u_hist.ndim == 1:
        u_hist = u_hist.reshape(-1, 1)

    N, M = u_hist.shape

    fig, ax = plt.subplots(figsize=(10, 4))

    # Default labels: u₁, u₂, ...
    base_labels = [f"$u_{i+1}$" for i in range(M)]

    for i in range(M):
        ax.plot(time, u_hist[:, i], label=base_labels[i])

    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"Control ({units})")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()


def plot_rw_momentum(
    time: np.ndarray,
    state_hist: np.ndarray,
    title: str = "Reaction Wheel Stored Momentum",
    units: str = "N·m·s"
) -> None:
    r"""
    Plot reaction wheel stored angular momentum over time.

    This function extracts reaction wheel momentum states from the spacecraft
    state history and visualizes their evolution. It is commonly used to
    evaluate momentum buildup, desaturation events, and long-term actuator
    usage.

    =========================
    State Vector Assumption
    =========================

    The spacecraft state vector is assumed to be ordered as

    .. math::

        \mathbf{x} =
        \begin{bmatrix}
        \boldsymbol{\omega} \\
        \mathbf{q} \\
        \mathbf{h}_{\text{rw}}
        \end{bmatrix}

    where:

    * :math:`\boldsymbol{\omega} \in \mathbb{R}^3` is body angular rate
    * :math:`\mathbf{q} \in \mathbb{R}^4` is the attitude quaternion
    * :math:`\mathbf{h}_{\text{rw}} \in \mathbb{R}^{N_{\text{rw}}}` is the
      reaction wheel stored angular momentum

    The reaction wheel momentum components are assumed to begin at index 7.

    ======================
    Mathematical Meaning
    ======================

    For each reaction wheel :math:`k`, the plotted quantity is

    .. math::

        h_k(t_i), \qquad i = 1, \dots, N

    expressed in units of angular momentum.

    :param time:
        One-dimensional array of time stamps in seconds.
    :type time:
        numpy.ndarray

    :param state_hist:
        Spacecraft state history containing reaction wheel momentum states.
    :type state_hist:
        numpy.ndarray

    :param title:
        Title displayed at the top of the plot.
    :type title:
        str

    :param units:
        Units of stored momentum shown on the y-axis.
    :type units:
        str

    :return:
        None. The function creates a Matplotlib figure.
    :rtype:
        None

    """
    time = np.asarray(time)
    state_hist = State.stack(state_hist)

    if state_hist.ndim != 2:
        raise ValueError(
            f"state_hist must be 2D (time_steps, state_dim). Got shape {state_hist.shape}"
        )

    N, state_dim = state_hist.shape

    if state_dim <= 7:
        raise ValueError(
            f"state_hist has no RW states: state_dim={state_dim}. Expected >7."
        )

    # Extract h_rw history
    h_hist = state_hist[:, 7:]  # shape: (N, N_rw)

    N, M = h_hist.shape  # N timesteps, M wheels

    fig, ax = plt.subplots(figsize=(10, 4))

    base_labels = [f"$h_{{{i+1}}}$" for i in range(M)]

    for i in range(M):
        ax.plot(time, h_hist[:, i], label=base_labels[i])

    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"Momentum ({units})")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()


import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from ADCS.helpers.math_helpers import rot_mat


def _normalize_quat(q: np.ndarray) -> Optional[np.ndarray]:
    q = np.asarray(q, dtype=float).reshape(4)
    if not np.all(np.isfinite(q)):
        return None
    n = np.linalg.norm(q)
    if n < 1e-12:
        return None
    return q / n


def _quat_attitude_error_deg(q: np.ndarray, q_ref: np.ndarray) -> float:
    """
    Minimal rotation angle between q and q_ref (scalar-first, Hamilton).
    """
    q = _normalize_quat(q)
    q_ref = _normalize_quat(q_ref)
    if q is None or q_ref is None:
        return np.nan

    # q_err = q^{-1} ⊗ q_ref
    q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
    w0, x0, y0, z0 = q_inv
    w1, x1, y1, z1 = q_ref

    q_err = np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
    ])

    w = float(np.clip(q_err[0], -1.0, 1.0))
    return float(np.rad2deg(2.0 * np.arccos(w)))


def plot_target_tracking(
    state_hist: np.ndarray,
    boresight_hist: np.ndarray,
    body_boresight: np.ndarray,
    time: Optional[np.ndarray] = None,
) -> None:
    Th = np.asarray(boresight_hist, dtype=float)
    X = State.stack(state_hist)

    # Normalize fixed body boresight
    v_bore_body = np.asarray(body_boresight, dtype=float).reshape(3)
    nb = np.linalg.norm(v_bore_body)
    if nb < 1e-12:
        raise ValueError("body_boresight must be a non-zero 3-vector")
    v_bore_body /= nb

    N = min(len(X), len(Th))
    if N <= 0:
        plt.figure(figsize=(10, 5))
        plt.title("Target Tracking Error (no data)")
        plt.show()
        return

    error_angle_deg = np.full(N, np.nan, dtype=float)

    for i in range(N):
        q = X[i, 3:7]
        row = np.asarray(Th[i], dtype=float).reshape(-1)

        # ---- Legacy vector-only target ----
        if row.size == 3:
            R_b2i = rot_mat(q)
            v_bore_eci = R_b2i @ v_bore_body
            if np.linalg.norm(v_bore_eci) < 1e-12:
                continue
            v_bore_eci /= np.linalg.norm(v_bore_eci)

            v_goal = row
            if not np.all(np.isfinite(v_goal)) or np.linalg.norm(v_goal) < 1e-12:
                continue
            v_goal /= np.linalg.norm(v_goal)

            dot = np.clip(np.dot(v_bore_eci, v_goal), -1.0, 1.0)
            error_angle_deg[i] = np.rad2deg(np.arccos(dot))

        # ---- Mixed Nx4 target ----
        elif row.size == 4:
            if np.isnan(row[0]):
                # Vector goal: [nan, gx, gy, gz]
                R_b2i = rot_mat(q)
                v_bore_eci = R_b2i @ v_bore_body
                if np.linalg.norm(v_bore_eci) < 1e-12:
                    continue
                v_bore_eci /= np.linalg.norm(v_bore_eci)

                v_goal = row[1:4]
                if not np.all(np.isfinite(v_goal)) or np.linalg.norm(v_goal) < 1e-12:
                    continue
                v_goal /= np.linalg.norm(v_goal)

                dot = np.clip(np.dot(v_bore_eci, v_goal), -1.0, 1.0)
                error_angle_deg[i] = np.rad2deg(np.arccos(dot))

            else:
                # Quaternion goal: [q0,q1,q2,q3]
                error_angle_deg[i] = _quat_attitude_error_deg(q, row)

        # ---- Anything else: skip ----
        else:
            continue

    # ---- Plot ----
    plt.figure(figsize=(10, 5))
    if time is not None:
        plt.plot(np.asarray(time)[:N], error_angle_deg)
        plt.xlabel("Time [s]")
    else:
        plt.plot(error_angle_deg)
        plt.xlabel("Sample")

    plt.ylabel("Tracking Error [deg]")
    plt.title("Target Tracking Error")
    plt.grid(True)
    plt.tight_layout()
