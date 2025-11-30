import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons
from numpy.linalg import norm
from typing import Callable, Tuple, List, Optional
from ADCS.helpers.math_helpers import quat_to_euler


def plot_state_comparison(
    time: np.ndarray,
    state_hist: np.ndarray,
    est_state_hist: Optional[np.ndarray] = None
) -> None:
    """
    Plot angular velocity and Euler angles.
    If est_state_hist is provided, also overlay estimated values.
    """
    euler_real = np.array([quat_to_euler(q) for q in state_hist[:, 3:7]])

    if est_state_hist is not None:
        euler_est = np.array([quat_to_euler(q) for q in est_state_hist[:, 3:7]])

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    axs = axs.flatten()

    state_labels = ["ω₁", "ω₂", "ω₃"]
    euler_labels = ["Roll [deg]", "Pitch [deg]", "Yaw [deg]"]

    # Angular velocity
    for i in range(3):
        axs[i].plot(time, state_hist[:, i], label="Real")

        if est_state_hist is not None:
            axs[i].plot(time, est_state_hist[:, i], "--", label="Estimated")

        axs[i].set_title(state_labels[i])
        axs[i].grid(True)

    # Euler angles
    for i in range(3):
        axs[i+3].plot(time, euler_real[:, i], label="Real")

        if est_state_hist is not None:
            axs[i+3].plot(time, euler_est[:, i], "--", label="Estimated")

        axs[i+3].set_title(euler_labels[i])
        axs[i+3].grid(True)
        axs[i+3].set_xlabel("Time [s]")

    # Only show legend if we have estimates too
    if est_state_hist is not None:
        axs[0].legend()

    title = "State Time Series (ω and Euler Angles)"
    if est_state_hist is not None:
        title = "Real vs Estimated States (ω and Euler Angles)"

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])


def plot_error_and_sun(
    time: np.ndarray,
    state_hist: np.ndarray,
    est_state_hist: np.ndarray,
    os_hist: List
) -> None:
    """
    Compute and plot quaternion error, angular velocity error, and sunlit state.

    Parameters
    ----------
    time : np.ndarray
        1D array of time stamps [s].
    state_hist : np.ndarray
        Real (truth) state history. Expected shape (N, >=7):
        [ω1, ω2, ω3, q0, q1, q2, q3, ...].
    est_state_hist : np.ndarray
        Estimated state history with same layout as state_hist.
    os_hist : List
        List of orbit/spacecraft state objects, each with:
            - .is_sunlit() -> bool
    """
    # Allocate error arrays
    quat_err = np.zeros_like(time, dtype=float)
    omega_err = np.zeros_like(time, dtype=float)

    # Compute errors
    for i in range(len(time)):
        q_hat = est_state_hist[i, 3:7]
        q = state_hist[i, 3:7]

        # dot product, clipped for numerical safety
        qdot = float(np.clip(np.dot(q_hat, q), -1.0, 1.0))

        # standard angle between quaternions (deg)
        quat_err[i] = (180.0 / np.pi) * np.arccos(-1.0 + 2.0 * qdot**2.0)

        # angular velocity error norm (convert rad/s to deg/s)
        omega_err[i] = norm(est_state_hist[i, 0:3] - state_hist[i, 0:3]) * 180.0 / np.pi

    # Sunlit state: 0 / 1 over time
    sun_state = np.array([int(os.is_sunlit()) for os in os_hist], dtype=int)

    # ---- Plotting ----
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 9))

    ax1.plot(time, quat_err)
    ax1.set_ylabel("Quat Err [deg]")
    ax1.grid(True)

    ax2.plot(time, omega_err)
    ax2.set_ylabel("ω Error [deg/s]")
    ax2.grid(True)

    ax3.step(time, sun_state, where="post", color="orange")
    ax3.set_ylim([-0.2, 1.2])
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["Dark", "Sunlit"])
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Sun")
    ax3.grid(True)

    fig.suptitle("Quaternion Error, Angular Velocity Error, and Sunlight State")
    fig.tight_layout(rect=[0, 0, 1, 0.96])


def plot_sensor_data(
    time: np.ndarray,
    sensor_hist: np.ndarray,
    clean_sensor_hist: np.ndarray
) -> None:
    """Plot measured vs clean sensor readings."""
    fig, axs = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
    axs = axs.flatten()

    sensor_names = [
        "MTM X", "MTM Y", "MTM Z",
        "GYR X", "GYR Y", "GYR Z",
        "SUN X", "SUN Y", "SUN Z"
    ]

    for i in range(9):
        axs[i].plot(time, sensor_hist[:, i], label="Measured")
        axs[i].plot(time, clean_sensor_hist[:, i], '--', label="Clean")
        axs[i].set_title(sensor_names[i])
        axs[i].grid(True)
        if i >= 6:
            axs[i].set_xlabel("Time [s]")

    axs[0].legend()
    fig.suptitle("Measured Sensor Readings vs Clean Sensor Values")
    fig.tight_layout(rect=[0, 0, 1, 0.96])



def plot_bias_comparison(
    time: np.ndarray,
    real_bias: np.ndarray,
    est_bias: np.ndarray,
    title: str,
    units: str
) -> None:
    """Generic function for plotting 3 bias components."""
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["Bias X", "Bias Y", "Bias Z"]

    for i in range(3):
        axs[i].plot(time, real_bias[:, i], label="Real Bias")
        axs[i].plot(time, est_bias[:, i], "--", label="Estimated Bias")
        axs[i].set_ylabel(f"{labels[i]} [{units}]")
        axs[i].grid(True)
        if i == 0:
            axs[i].legend()

    axs[2].set_xlabel("Time [s]")
    fig.suptitle(title)
    fig.tight_layout()
