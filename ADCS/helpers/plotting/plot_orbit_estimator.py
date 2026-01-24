__all__ = [
    "plot_gps_error"
]

import matplotlib.pyplot as plt
import numpy as np

def plot_gps_error(time_hist, true_hist, est_hist):
    r"""
    Plot true, estimated, and error time histories for GPS-based position and velocity.

    This function generates a 3×2 grid of subplots comparing true and estimated
    orbital position and velocity components over time. In each subplot, the
    estimation error (estimated minus true) is also shown, enabling direct
    assessment of navigation performance.

    ======================
    State Decomposition
    ======================

    The true and estimated orbital state histories are assumed to be structured as

    .. math::

        \mathbf{x} =
        \begin{bmatrix}
        x & y & z & \dot{x} & \dot{y} & \dot{z}
        \end{bmatrix}^T

    where:

    * :math:`(x, y, z)` are Cartesian position components
    * :math:`(\dot{x}, \dot{y}, \dot{z})` are Cartesian velocity components

    Position components are converted from meters to kilometers for plotting,
    while velocity components are assumed to already be expressed in km/s.

    ======================
    Error Definition
    ======================

    For each state component :math:`s_i`, the estimation error is defined as

    .. math::

        e_i(t) = \hat{s}_i(t) - s_i(t)

    where :math:`\hat{s}_i` is the estimated value and :math:`s_i` is the true
    value.

    ======================
    Visualization Layout
    ======================

    The six subplots are arranged as follows:

    +-------------------+-------------------+
    | Position X        | Position Y        |
    +-------------------+-------------------+
    | Position Z        | Velocity X        |
    +-------------------+-------------------+
    | Velocity Y        | Velocity Z        |
    +-------------------+-------------------+

    Each subplot includes:
    * True state (solid line)
    * Estimated state (dashed line)
    * Estimation error (dotted line)

    ======================
    Intended Use
    ======================

    This visualization is particularly useful for evaluating GPS navigation
    filters, orbit determination algorithms, and estimator convergence
    behavior.

    :param time_hist:
        One-dimensional array of time stamps in seconds.
    :type time_hist:
        numpy.ndarray

    :param true_hist:
        True orbital state history with columns
        ``[x, y, z, vx, vy, vz]``.
    :type true_hist:
        numpy.ndarray

    :param est_hist:
        Estimated orbital state history with the same layout as ``true_hist``.
    :type est_hist:
        numpy.ndarray

    :return:
        None. The function generates a Matplotlib figure.
    :rtype:
        None

    """

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    labels = [
        "Position X (km)",
        "Position Y (km)",
        "Position Z (km)",
        "Velocity X (km/s)",
        "Velocity Y (km/s)",
        "Velocity Z (km/s)",
    ]

    # Convert m to km for nicer plotting
    pos_true = true_hist[:, 0:3] / 1000.0
    pos_est  = est_hist[:, 0:3]  / 1000.0
    vel_true = true_hist[:, 3:6]
    vel_est  = est_hist[:, 3:6]

    data_pairs = [
        (pos_true[:,0], pos_est[:,0]),   # X
        (pos_true[:,1], pos_est[:,1]),   # Y
        (pos_true[:,2], pos_est[:,2]),   # Z
        (vel_true[:,0], vel_est[:,0]),   # Vx
        (vel_true[:,1], vel_est[:,1]),   # Vy
        (vel_true[:,2], vel_est[:,2]),   # Vz
    ]

    for i, ax in enumerate(axes):
        true_i, est_i = data_pairs[i]
        ax.plot(time_hist, true_i, label="True", linewidth=2)
        ax.plot(time_hist, est_i, "--", label="Estimated", linewidth=2)

        # Error
        ax.plot(time_hist, est_i - true_i, "k:", alpha=0.6, label="Error")

        ax.set_title(labels[i])
        ax.set_xlabel("Time (s)")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()