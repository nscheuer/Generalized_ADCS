__all__ = [
    "plot_state_comparison",
    "plot_error_and_sun",
    "plot_sensor_data",
    "plot_bias_comparison"
]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons
from numpy.linalg import norm
from typing import Callable, Tuple, List, Optional
from ADCS.helpers.math_helpers import quat_to_euler
from ADCS.state import State


def plot_state_comparison(
    time: np.ndarray,
    state_hist: np.ndarray,
    est_state_hist: Optional[np.ndarray] = None
) -> None:
    r"""
    Plot angular velocity and Euler angle time histories, with optional estimation overlay.

    This function visualizes the rotational state of a spacecraft by plotting:

    * Body angular velocity components :math:`\boldsymbol{\omega} = [\omega_1, \omega_2, \omega_3]^T`
    * Euler attitude angles (roll, pitch, yaw) derived from attitude quaternions

    If an estimated state history is provided, estimated values are overlaid
    for direct comparison against the true (simulated) states.

    ========================
    Mathematical Background
    ========================

    The spacecraft attitude is represented by a quaternion

    .. math::

        \mathbf{q} =
        \begin{bmatrix}
        q_0 & q_1 & q_2 & q_3
        \end{bmatrix}^T

    which is converted to Euler angles

    .. math::

        (\phi, \theta, \psi)

    using :func:`~ADCS.helpers.math_helpers.quat_to_euler`. The angular velocity
    components are assumed to be stored directly in the state vector as

    .. math::

        \boldsymbol{\omega}(t_i) = [\omega_1, \omega_2, \omega_3]^T

    ======================
    Visualization Layout
    ======================

    The figure consists of six subplots arranged in a 3×2 grid:

    * Top three: angular velocity components
    * Bottom three: Euler angles (roll, pitch, yaw)

    ======================
    Usage Notes
    ======================

    * Euler angles are plotted in degrees
    * Angular velocities are plotted in their native units (typically rad/s)
    * Legends are shown only when estimated states are provided

    :param time:
        One-dimensional array of time stamps in seconds.
    :type time:
        numpy.ndarray

    :param state_hist:
        True spacecraft state history. Columns ``[0:3]`` contain angular velocity
        and columns ``[3:7]`` contain attitude quaternions.
    :type state_hist:
        numpy.ndarray

    :param est_state_hist:
        Optional estimated spacecraft state history with the same layout as
        ``state_hist``.
    :type est_state_hist:
        numpy.ndarray or None

    :return:
        None. The function generates a Matplotlib figure.
    :rtype:
        None

    """
    state_hist = State.stack(state_hist)
    if est_state_hist is not None:
        est_state_hist = State.stack(est_state_hist)
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
    r"""
    Plot attitude error, angular velocity error, and sunlight state over time.

    This function evaluates estimation performance by computing and plotting:

    * Quaternion attitude error magnitude
    * Angular velocity estimation error norm
    * Binary sunlight (sunlit / eclipse) state

    All quantities are plotted against time in a shared x-axis layout.

    ========================
    Quaternion Error Metric
    ========================

    Let the true and estimated quaternions be :math:`\mathbf{q}` and
    :math:`\hat{\mathbf{q}}`, respectively. The scalar quaternion alignment
    measure is computed via the dot product:

    .. math::

        d = \hat{\mathbf{q}}^T \mathbf{q}

    The equivalent rotation angle error is

    .. math::

        \theta_q =
        \cos^{-1}\!\left(-1 + 2 d^2\right)

    which represents the smallest angular rotation between the two attitudes.
    The result is reported in degrees.

    ========================
    Angular Velocity Error
    ========================

    The angular velocity error is computed as the Euclidean norm of the
    difference between estimated and true angular velocity vectors:

    .. math::

        e_\omega =
        \left\| \hat{\boldsymbol{\omega}} - \boldsymbol{\omega} \right\|

    and converted from rad/s to deg/s.

    ======================
    Sunlight State
    ======================

    The sunlight state is obtained from each orbital state object via
    ``is_sunlit()`` and plotted as a binary signal:

    * 1 — Sunlit
    * 0 — Eclipse

    ======================
    Visualization Layout
    ======================

    The figure consists of three stacked subplots:

    #. Quaternion attitude error
    #. Angular velocity error
    #. Sunlight state (step plot)

    :param time:
        One-dimensional array of time stamps in seconds.
    :type time:
        numpy.ndarray

    :param state_hist:
        True spacecraft state history.
    :type state_hist:
        numpy.ndarray

    :param est_state_hist:
        Estimated spacecraft state history.
    :type est_state_hist:
        numpy.ndarray

    :param os_hist:
        List of orbital state objects providing sunlight information.
    :type os_hist:
        list

    :return:
        None. The function generates a Matplotlib figure.
    :rtype:
        None

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
    r"""
    Plot measured versus clean sensor data for multiple sensor channels.

    This function visualizes raw (measured) sensor outputs alongside their
    corresponding clean or noise-free values, enabling assessment of sensor
    noise, bias, and filtering performance.

    ======================
    Data Organization
    ======================

    Let the sensor measurement matrix be

    .. math::

        \mathbf{Y}(t) \in \mathbb{R}^{N \times M}

    where:

    * :math:`N` is the number of time samples
    * :math:`M` is the number of sensor channels

    Each sensor channel :math:`j` is plotted as:

    .. math::

        y_j^{\text{meas}}(t_i), \quad
        y_j^{\text{clean}}(t_i)

    ======================
    Visualization Layout
    ======================

    * Subplots are arranged in a near-square grid
    * Each subplot corresponds to one sensor channel
    * Unused subplot axes are hidden automatically

    :param time:
        One-dimensional array of time stamps in seconds.
    :type time:
        numpy.ndarray

    :param sensor_hist:
        Measured sensor data history.
    :type sensor_hist:
        numpy.ndarray

    :param clean_sensor_hist:
        Clean or reference sensor data history.
    :type clean_sensor_hist:
        numpy.ndarray

    :return:
        None. The function generates a Matplotlib figure.
    :rtype:
        None

    """

    n = sensor_hist.shape[1]  # number of sensor channels

    # Choose a near-square layout
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharex=True)
    axs = np.atleast_1d(axs).flatten()

    # Generate generic labels if none are provided
    sensor_names = [f"Sensor {i}" for i in range(n)]

    for i in range(n):
        axs[i].plot(time, sensor_hist[:, i], label="Measured")
        axs[i].plot(time, clean_sensor_hist[:, i], "--", label="Clean")
        axs[i].set_title(sensor_names[i])
        axs[i].grid(True)

        # Only bottom row gets x labels
        if i >= (nrows - 1) * ncols:
            axs[i].set_xlabel("Time [s]")

    # Hide unused subplots (if n is not a perfect grid)
    for j in range(n, len(axs)):
        axs[j].set_visible(False)

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
    r"""
    Plot true versus estimated bias components over time.

    This function provides a generic visualization for comparing real and
    estimated bias states, such as gyro bias, accelerometer bias, or sensor
    offsets.

    ======================
    Mathematical Context
    ======================

    Let the bias vector be

    .. math::

        \mathbf{b}(t) =
        \begin{bmatrix}
        b_1(t) & b_2(t) & \dots & b_M(t)
        \end{bmatrix}^T

    with corresponding estimates :math:`\hat{\mathbf{b}}(t)`. The function
    plots each component:

    .. math::

        b_k(t_i), \quad \hat{b}_k(t_i)

    ======================
    Visualization Layout
    ======================

    * One subplot per bias component
    * Shared time axis
    * Real bias shown as solid line
    * Estimated bias shown as dashed line

    :param time:
        One-dimensional array of time stamps in seconds.
    :type time:
        numpy.ndarray

    :param real_bias:
        True bias history. Can be one- or two-dimensional.
    :type real_bias:
        numpy.ndarray

    :param est_bias:
        Estimated bias history. Must match ``real_bias`` in shape.
    :type est_bias:
        numpy.ndarray

    :param title:
        Title displayed at the top of the figure.
    :type title:
        str

    :param units:
        Units of the bias values shown on the y-axis.
    :type units:
        str

    :return:
        None. The function generates a Matplotlib figure.
    :rtype:
        None

    """
    real_bias = np.atleast_2d(real_bias).reshape(len(real_bias), -1)
    est_bias  = np.atleast_2d(est_bias).reshape(len(est_bias), -1)

    n = real_bias.shape[1]  # number of bias components

    fig, axs = plt.subplots(n, 1, figsize=(10, 3*n), sharex=True)

    # Ensure axs is always iterable (matplotlib returns a single Axes if n == 1)
    if n == 1:
        axs = [axs]

    for i in range(n):
        axs[i].plot(time, real_bias[:, i], label="Real Bias")
        axs[i].plot(time, est_bias[:, i], "--", label="Estimated Bias")
        axs[i].set_ylabel(f"Bias {i} [{units}]")
        axs[i].grid(True)

        if i == 0:
            axs[i].legend()

    axs[-1].set_xlabel("Time [s]")
    fig.suptitle(title)
    fig.tight_layout()
