import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List

from ADCS.helpers.math_helpers import rot_mat

def animate_attitude(
    time: np.ndarray,
    state_hist: Optional[np.ndarray] = None,
    est_state_hist: Optional[np.ndarray] = None,
    os_hist: Optional[List] = None,
    boresight_goal_hist: Optional[np.ndarray] = None
) -> None:
    """
    3D animation of body axes, magnetic field vector, sun vector, and goal vector.

    Parameters
    ----------
    time : np.ndarray
        1D array of time stamps [s].
    state_hist : np.ndarray, optional
        True state history including quaternions (cols 3-7), shape (N, >7).
    est_state_hist : np.ndarray, optional
        Estimated state history including quaternions (cols 3-7), shape (N, >7).
    os_hist : List, optional
        List of orbit/spacecraft state objects.
    boresight_goal_hist : np.ndarray, optional
        History of the target boresight vector in ECI frame. Shape (N, 3).
        Plotted as a dotted line.
    """
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xlabel("X (ECI)")
    ax.set_ylabel("Y (ECI)")
    ax.set_zlabel("Z (ECI)")
    
    title_parts = []
    if state_hist is not None: title_parts.append("True Att")
    if est_state_hist is not None: title_parts.append("Est Att")
    if os_hist is not None: title_parts.append("Env Vectors")
    if boresight_goal_hist is not None: title_parts.append("Goal")
    
    ax.set_title(" + ".join(title_parts) if title_parts else "Empty Plot")

    body_axes = np.eye(3)
    
    # --- Initialize Artists ---
    
    # 1. True Attitude Lines
    true_lines = []
    if state_hist is not None:
        colors = ['r', 'g', 'b'] 
        true_lines = [ax.plot([], [], [], lw=2, color=colors[k], label=f'True {axis}')[0] 
                      for k, axis in enumerate(['X', 'Y', 'Z'])]

    # 2. Estimated Attitude Lines
    est_lines = []
    if est_state_hist is not None:
        colors = ['salmon', 'lightgreen', 'lightblue']
        est_lines = [ax.plot([], [], [], lw=1, linestyle="--", color=colors[k])[0] for k in range(3)]

    # 3. Boresight Goal Line
    goal_line = None
    if boresight_goal_hist is not None:
        # Using cyan to differentiate from the Body-Z (Blue)
        goal_line = ax.plot([], [], [], lw=2, linestyle=":", color='cyan', label="Goal Vector")[0]

    # 4. Environment Quivers
    B_arrow = None
    S_arrow = None
    quiver_artists = []

    if os_hist is not None:
        B_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='magenta', label='B-Field')
        S_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='orange', label='Sun')
        quiver_artists = [B_arrow, S_arrow]

    # --- Legend ---
    if true_lines or est_lines or os_hist or goal_line:
        proxies = []
        labels = []
        
        if state_hist is not None:
            proxies.append(plt.Line2D([0], [0], color='r', lw=2))
            labels.append("True Body X")
        
        if est_state_hist is not None:
            proxies.append(plt.Line2D([0], [0], color='salmon', lw=1, linestyle="--"))
            labels.append("Est Body X")
            
        if boresight_goal_hist is not None:
            proxies.append(plt.Line2D([0], [0], color='cyan', lw=2, linestyle=":"))
            labels.append("Goal Vector")

        if os_hist is not None:
            proxies.extend([plt.Line2D([0], [0], color='magenta', lw=2), 
                            plt.Line2D([0], [0], color='orange', lw=2)])
            labels.extend(["Magnetic Field", "Sun Vector"])
            
        ax.legend(proxies, labels, loc='upper left')

    # --- Animation Control Variables ---
    frame = [0]
    play = [True]
    speed = [1.0]

    def init_anim():
        artists = []
        if true_lines: artists.extend(true_lines)
        if est_lines: artists.extend(est_lines)
        if goal_line: artists.append(goal_line)
        if quiver_artists: artists.extend(quiver_artists)
        return artists

    def update(_):
        nonlocal B_arrow, S_arrow

        current_artists = []
        if true_lines: current_artists.extend(true_lines)
        if est_lines: current_artists.extend(est_lines)
        if goal_line: current_artists.append(goal_line)
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
            Rt = rot_mat(q_true) # Body to ECI
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

        # --- Update Boresight Goal ---
        if goal_line is not None:
            g_vec = boresight_goal_hist[i]
            norm_g = np.linalg.norm(g_vec)
            if norm_g > 1e-9:
                g_vec = g_vec / norm_g
                goal_line.set_data([0, g_vec[0]], [0, g_vec[1]])
                goal_line.set_3d_properties([0, g_vec[2]])
            else:
                goal_line.set_data([], [])
                goal_line.set_3d_properties([])

        # --- Update Environment Vectors ---
        if os_hist is not None:
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
                    S_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='grey', alpha=0)
            
            # Re-add to return list (important because objects were re-created)
            current_artists = []
            if true_lines: current_artists.extend(true_lines)
            if est_lines: current_artists.extend(est_lines)
            if goal_line: current_artists.append(goal_line)
            if B_arrow: current_artists.append(B_arrow)
            if S_arrow: current_artists.append(S_arrow)

        return current_artists

    ani = FuncAnimation(fig, update, init_func=init_anim, interval=50)

    # --- UI Controls ---
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