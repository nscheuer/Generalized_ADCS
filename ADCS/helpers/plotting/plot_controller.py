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
    r"""
    Plot angular tracking error between a body-fixed boresight and an inertial target.

    This function computes and visualizes the angular separation between a
    spacecraft-fixed boresight vector and a desired target direction expressed
    in the Earth-Centered Inertial (ECI) frame. It is commonly used to assess
    pointing performance for payloads, antennas, or sensors.

    ======================
    Attitude and Geometry
    ======================

    Let the spacecraft attitude be represented by a unit quaternion

    .. math::

        \mathbf{q} =
        \begin{bmatrix}
        q_0 & q_1 & q_2 & q_3
        \end{bmatrix}^T

    mapping vectors from the body frame to the ECI frame via the rotation matrix

    .. math::

        \mathbf{R}_{\mathcal{B}\rightarrow\mathcal{I}}(\mathbf{q})

    computed using :func:`~ADCS.helpers.math_helpers.rot_mat`.

    ============================
    Tracking Error Computation
    ============================

    Let:

    * :math:`\hat{\mathbf{b}} \in \mathbb{R}^3` be the normalized boresight
      direction expressed in the body frame
    * :math:`\hat{\mathbf{g}}_i \in \mathbb{R}^3` be the normalized target
      direction in ECI at time step :math:`i`

    The boresight direction in ECI is

    .. math::

        \hat{\mathbf{b}}_i^{\text{ECI}} =
        \mathbf{R}_{\mathcal{B}\rightarrow\mathcal{I}}(\mathbf{q}_i)\,
        \hat{\mathbf{b}}

    The instantaneous pointing error angle is computed using the dot product:

    .. math::

        \theta_i =
        \cos^{-1}\!\left(
        \hat{\mathbf{b}}_i^{\text{ECI}} \cdot \hat{\mathbf{g}}_i
        \right)

    The resulting angle is converted to degrees for visualization.

    ========================
    Visualization Options
    ========================

    * If a time vector is provided, the error is plotted versus time
    * Otherwise, the error is plotted versus sample index

    :param state_hist:
        True spacecraft state history containing attitude quaternions in
        columns ``[3:7]``.
    :type state_hist:
        numpy.ndarray

    :param boresight_hist:
        Desired target boresight vectors expressed in the ECI frame.
    :type boresight_hist:
        numpy.ndarray

    :param body_boresight:
        Fixed boresight direction expressed in the spacecraft body frame.
    :type body_boresight:
        numpy.ndarray

    :param time:
        Optional time array for the x-axis.
    :type time:
        numpy.ndarray or None

    :return:
        None. The function creates a Matplotlib figure.
    :rtype:
        None

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