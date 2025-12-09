import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons
from numpy.linalg import norm
from typing import Callable, Tuple, List
from ADCS.helpers.math_helpers import quat_to_euler

def plot_control(
    time: np.ndarray,
    u_hist: np.ndarray,
    title: str = "Control Effort Over Time",
    units: str = "Command"
) -> None:
    """
    Plot control command time series.

    Parameters
    ----------
    time : np.ndarray
        1D array of time stamps [s], shape (N,).
    u_hist : np.ndarray
        Control history, shape (N, M) where M is the number of actuators /
        control channels.
    title : str, optional
        Title of the plot.
    units : str, optional
        Units of the control command for the y-label (e.g. "N·m").
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
    """
    Plot reaction wheel stored angular momentum over time using state history.

    Assumes state vector structure:
        x = [w(3), q(4), h_rw(N_rw)]

    i.e. RW momentum terms begin at index 7 onward.

    Parameters
    ----------
    time : np.ndarray
        Time array of shape (N,).
    state_hist : np.ndarray
        State history of shape (N, 7 + N_rw), where N_rw >= 1.
        Momentum components must be stored from index 7 onward.
    title : str
        Plot title.
    units : str
        Units for y-label (default: "N·m·s").
    """
    time = np.asarray(time)
    state_hist = np.asarray(state_hist)

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


def plot_target_tracking(
    state_hist: np.ndarray,
    boresight_hist: np.ndarray,
    body_boresight: np.ndarray,
    time: Optional[np.ndarray] = None
) -> None:
    """
    Plot angular tracking error between body boresight and ECI target vector.

    Parameters
    ----------
    state_hist : np.ndarray
        True state history including quaternions (cols 3-7), shape (N, >7).
        Quaternion assumed to be Hamilton, body -> ECI (same as animate_attitude).
    boresight_hist : np.ndarray
        Target boresight vector in ECI frame, shape (N, 3).
    body_boresight : np.ndarray
        Fixed boresight vector in BODY frame, shape (3,).
    time : np.ndarray, optional
        Time vector for x-axis. If None, index is used.
    """

    N = min(len(state_hist), len(boresight_hist))

    # Normalize fixed body boresight
    v_bore_body = body_boresight / np.linalg.norm(body_boresight)

    error_angle = np.zeros(N)

    for i in range(N):
        q = state_hist[i, 3:7]
        R_b2i = rot_mat(q)  # Body -> ECI

        # Rotate body boresight into ECI
        v_bore_eci = R_b2i @ v_bore_body
        v_bore_eci /= np.linalg.norm(v_bore_eci)

        # Normalize ECI goal vector
        v_goal = boresight_hist[i]
        v_goal /= np.linalg.norm(v_goal)

        # Angle error via dot product
        dot = np.clip(np.dot(v_bore_eci, v_goal), -1.0, 1.0)
        error_angle[i] = np.arccos(dot)  # radians

    # Convert to degrees
    error_angle_deg = np.rad2deg(error_angle)

    # ---- Plot ----
    plt.figure(figsize=(10, 5))

    if time is not None:
        plt.plot(time[:N], error_angle_deg)
        plt.xlabel("Time [s]")
    else:
        plt.plot(error_angle_deg)
        plt.xlabel("Sample")

    plt.ylabel("Tracking Error [deg]")
    plt.title("Target Tracking Error (Boresight vs ECI Target)")
    plt.grid(True)
    plt.tight_layout()