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