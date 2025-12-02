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