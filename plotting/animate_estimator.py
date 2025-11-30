import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons
from typing import List, Optional
from ADCS.helpers.math_helpers import rot_mat


def animate_attitude(
    time: np.ndarray,
    state_hist: Optional[np.ndarray] = None,
    est_state_hist: Optional[np.ndarray] = None,
    os_hist: Optional[List] = None
) -> None:
    """
    3D animation of body axes, magnetic field vector, and sun vector.
    
    Can selectively plot true attitude, estimated attitude, and environmental vectors
    based on which history arrays are provided.

    Parameters
    ----------
    time : np.ndarray
        1D array of time stamps [s].
    state_hist : np.ndarray, optional
        True state history including quaternions (cols 3-7), shape (N, >7).
    est_state_hist : np.ndarray, optional
        Estimated state history including quaternions (cols 3-7), shape (N, >7).
    os_hist : List, optional
        List of orbit/spacecraft state objects, each with:
            - .B : np.ndarray (magnetic field vector)
            - .S : np.ndarray (sun vector)
            - .is_sunlit() -> bool
    """
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    title_parts = []
    if state_hist is not None: title_parts.append("True Att")
    if est_state_hist is not None: title_parts.append("Est Att")
    if os_hist is not None: title_parts.append("Env Vectors")
    ax.set_title(" + ".join(title_parts) if title_parts else "Empty Plot")

    body_axes = np.eye(3)
    
    # --- Initialize Artists ---
    true_lines = []
    if state_hist is not None:
        # RGB for Body X, Y, Z
        colors = ['r', 'g', 'b'] 
        true_lines = [ax.plot([], [], [], lw=2, color=colors[k], label=f'True {axis}')[0] 
                      for k, axis in enumerate(['X', 'Y', 'Z'])]

    est_lines = []
    if est_state_hist is not None:
        colors = ['salmon', 'lightgreen', 'lightblue']
        est_lines = [ax.plot([], [], [], lw=1, linestyle="--", color=colors[k])[0] for k in range(3)]

    B_arrow = None
    S_arrow = None
    
    # We need a container for quiver artists because they are tricky to update in 3D
    # We will remove() and re-add them every frame if they exist
    quiver_artists = []

    # Initial dummy quivers if needed
    if os_hist is not None:
        B_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='magenta', label='B-Field')
        S_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='orange', label='Sun')
        quiver_artists = [B_arrow, S_arrow]

    # Add legend if we have things to show
    if true_lines or est_lines or os_hist:
        # Create proxy artists for legend
        proxies = []
        labels = []
        if state_hist is not None:
            proxies.append(plt.Line2D([0], [0], color='r', lw=2))
            labels.append("True Body X")
        if est_state_hist is not None:
            proxies.append(plt.Line2D([0], [0], color='salmon', lw=1, linestyle="--"))
            labels.append("Est Body X")
        if os_hist is not None:
            proxies.extend([plt.Line2D([0], [0], color='magenta', lw=2), 
                           plt.Line2D([0], [0], color='orange', lw=2)])
            labels.extend(["Magnetic Field", "Sun Vector"])
        ax.legend(proxies, labels, loc='upper left')

    frame = [0]
    play = [True]
    speed = [1.0]

    def init_anim():
        artists = []
        if true_lines: artists.extend(true_lines)
        if est_lines: artists.extend(est_lines)
        if quiver_artists: artists.extend(quiver_artists)
        return artists

    def update(_):
        nonlocal B_arrow, S_arrow

        # Return existing artists if paused
        current_artists = []
        if true_lines: current_artists.extend(true_lines)
        if est_lines: current_artists.extend(est_lines)
        if B_arrow: current_artists.append(B_arrow)
        if S_arrow: current_artists.append(S_arrow)

        if not play[0]:
            return current_artists

        # Update Frame Index
        frame[0] = (frame[0] + speed[0]) % len(time)
        i = int(frame[0])

        # --- Update True Attitude ---
        if state_hist is not None:
            q_true = state_hist[i, 3:7]
            Rt = rot_mat(q_true)
            true_ax = Rt @ body_axes
            for k in range(3):
                true_lines[k].set_data([0, true_ax[0, k]], [0, true_ax[1, k]])
                true_lines[k].set_3d_properties([0, true_ax[2, k]])

        # --- Update Estimated Attitude ---
        if est_state_hist is not None:
            q_est = est_state_hist[i, 3:7]
            Re = rot_mat(q_est)
            est_ax = Re @ body_axes
            for k in range(3):
                est_lines[k].set_data([0, est_ax[0, k]], [0, est_ax[1, k]])
                est_lines[k].set_3d_properties([0, est_ax[2, k]])

        # --- Update Environment Vectors ---
        if os_hist is not None:
            # Remove old arrows
            if B_arrow: B_arrow.remove()
            if S_arrow: S_arrow.remove()

            # Magnetic Field
            if hasattr(os_hist[i], 'B') and os_hist[i].B is not None:
                norm_B = np.linalg.norm(os_hist[i].B)
                if norm_B > 1e-9:
                    B = os_hist[i].B / norm_B
                    B_arrow = ax.quiver(0, 0, 0, B[0], B[1], B[2], color='magenta')
                else:
                    B_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='magenta', alpha=0)

            # Sun Vector
            if hasattr(os_hist[i], 'S') and os_hist[i].S is not None:
                is_lit = True
                if hasattr(os_hist[i], 'is_sunlit'):
                    is_lit = os_hist[i].is_sunlit()
                
                if is_lit:
                    norm_S = np.linalg.norm(os_hist[i].S)
                    if norm_S > 1e-9:
                        S = os_hist[i].S / norm_S
                        S_arrow = ax.quiver(0, 0, 0, S[0], S[1], S[2], color='orange')
                    else:
                        S_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='orange', alpha=0)
                else:
                    # Shadow: plot invisible or distinct style
                    S_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='grey', alpha=0)
            
            # Update the return list
            current_artists = []
            if true_lines: current_artists.extend(true_lines)
            if est_lines: current_artists.extend(est_lines)
            if B_arrow: current_artists.append(B_arrow)
            if S_arrow: current_artists.append(S_arrow)

        return current_artists

    ani = FuncAnimation(fig, update, init_func=init_anim, interval=50)

    # UI Controls
    ax_pause = plt.axes([0.75, 0.02, 0.15, 0.05])
    btn_pause = Button(ax_pause, "Pause / Play")
    btn_pause.on_clicked(lambda e: play.__setitem__(0, not play[0]))

    ax_speed = plt.axes([0.02, 0.02, 0.20, 0.15])
    speed_buttons = RadioButtons(ax_speed, ["0.25x", "0.5x", "1x", "2x", "4x"], active=2)
    
    def set_speed(label):
        val = float(label.replace('x', ''))
        speed[0] = val
        
    speed_buttons.on_clicked(set_speed)
    
    # Store animation to prevent garbage collection
    fig.animation = ani