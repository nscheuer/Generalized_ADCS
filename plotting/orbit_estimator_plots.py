import matplotlib.pyplot as plt
import numpy as np

def plot_gps_error(time_hist, true_hist, est_hist):
    """
    Creates 6 subplots showing position (3) and velocity (3)
    for both the true and estimated orbital states.
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