import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons
from typing import List
from ADCS.helpers.math_helpers import rot_mat


def animate_attitude(
    time: np.ndarray,
    state_hist: np.ndarray,
    est_state_hist: np.ndarray,
    os_hist: List
) -> None:
    """
    3D animation of body axes, magnetic field vector, and sun vector.

    Parameters
    ----------
    time : np.ndarray
        1D array of time stamps [s].
    q_true : np.ndarray
        True quaternion history, shape (N, 4).
    q_est : np.ndarray
        Estimated quaternion history, shape (N, 4).
    os_hist : List
        List of orbit/spacecraft state objects, each with:
            - .B : np.ndarray (magnetic field vector)
            - .S : np.ndarray (sun vector)
            - .is_sunlit() -> bool
    """
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    q_true = state_hist[:, 3:7]
    q_est  = est_state_hist[:, 3:7]

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Attitude + Magnetic Field + Sun Vector")

    body_axes = np.eye(3)
    true_lines = [ax.plot([], [], [], lw=2)[0] for _ in range(3)]
    est_lines  = [ax.plot([], [], [], lw=1, linestyle="--")[0] for _ in range(3)]

    # Initial empty quivers
    B_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='magenta')
    S_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='yellow')

    frame = [0]
    play = [True]
    speed = [1.0]

    def init_anim():
        return true_lines + est_lines + [B_arrow, S_arrow]

    def update(_):
        nonlocal B_arrow, S_arrow

        if not play[0]:
            return true_lines + est_lines + [B_arrow, S_arrow]

        frame[0] = (frame[0] + speed[0]) % len(time)
        i = int(frame[0])

        Rt = rot_mat(q_true[i])
        Re = rot_mat(q_est[i])

        true_ax = Rt @ body_axes
        est_ax  = Re @ body_axes

        # Body axes
        for k in range(3):
            true_lines[k].set_data([0, true_ax[0, k]], [0, true_ax[1, k]])
            true_lines[k].set_3d_properties([0, true_ax[2, k]])

            est_lines[k].set_data([0, est_ax[0, k]], [0, est_ax[1, k]])
            est_lines[k].set_3d_properties([0, est_ax[2, k]])

        # B vector
        B_arrow.remove()
        B = os_hist[i].B / np.linalg.norm(os_hist[i].B)
        B_arrow = ax.quiver(0, 0, 0, B[0], B[1], B[2], color='magenta')

        # Sun vector
        S_arrow.remove()
        if os_hist[i].is_sunlit():
            S = os_hist[i].S / np.linalg.norm(os_hist[i].S)
            S_arrow = ax.quiver(0, 0, 0, S[0], S[1], S[2], color='yellow')
        else:
            S_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='yellow', linewidth=0)

        return true_lines + est_lines + [B_arrow, S_arrow]

    ani = FuncAnimation(fig, update, init_func=init_anim, interval=50)

    # UI Controls
    ax_pause = plt.axes([0.75, 0.02, 0.15, 0.05])
    btn_pause = Button(ax_pause, "Pause / Play")
    btn_pause.on_clicked(lambda e: play.__setitem__(0, not play[0]))

    ax_speed = plt.axes([0.02, 0.02, 0.20, 0.15])
    speed_buttons = RadioButtons(ax_speed, ["0.25×", "0.5×", "1×", "2×", "4×"], active=2)
    speed_buttons.on_clicked(
        lambda label: speed.__setitem__(0, {
            "0.25×": 0.25,
            "0.5×": 0.5,
            "1×": 1.0,
            "2×": 2.0,
            "4×": 4.0
        }[label])
    )
    fig.animation = ani
    plt.show(block=False)