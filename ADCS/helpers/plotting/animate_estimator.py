import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons
from typing import Optional, List, Sequence

from ADCS.helpers.math_helpers import rot_mat
from ADCS.state import State

__all__ = ["animate_attitude"]


def animate_attitude(
    time: np.ndarray,
    state_hist: Optional[Sequence[State]] = None,
    est_state_hist: Optional[Sequence[State]] = None,
    os_hist: Optional[List] = None,
    boresight_goal_hist: Optional[np.ndarray] = None,
) -> None:
    r"""
    Animate spacecraft attitude, estimated attitude, environment vectors, and boresight goals in 3D.

    UPDATED boresight_goal_hist:
      - Nx3 legacy: [gx, gy, gz] (ECI) -> draw vector
      - Nx4 mixed:
          * [nan, gx, gy, gz] -> draw vector
          * [q0, q1, q2, q3]  -> draw target body axes (Body->ECI)
    """
    if state_hist is not None:
        state_hist = State.stack(state_hist)
    if est_state_hist is not None:
        est_state_hist = State.stack(est_state_hist)
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xlabel("X (ECI)")
    ax.set_ylabel("Y (ECI)")
    ax.set_zlabel("Z (ECI)")

    title_parts = []
    if state_hist is not None:
        title_parts.append("True Att")
    if est_state_hist is not None:
        title_parts.append("Est Att")
    if os_hist is not None:
        title_parts.append("Env Vectors")
    if boresight_goal_hist is not None:
        title_parts.append("Goal (vec/quat)")

    ax.set_title(" + ".join(title_parts) if title_parts else "Empty Plot")

    body_axes = np.eye(3, dtype=float)

    def _safe_unit(v: np.ndarray) -> Optional[np.ndarray]:
        v = np.asarray(v, dtype=float).reshape(-1)
        n = np.linalg.norm(v)
        if n <= 1e-12:
            return None
        return v / n

    def _clear_line(line):
        line.set_data([], [])
        line.set_3d_properties([])

    # --- Initialize Artists ---

    # 1. True Attitude Lines
    true_lines = []
    if state_hist is not None:
        colors = ["r", "g", "b"]
        true_lines = [
            ax.plot([], [], [], lw=2, color=colors[k], label=f"True {axis}")[0]
            for k, axis in enumerate(["X", "Y", "Z"])
        ]

    # 2. Estimated Attitude Lines
    est_lines = []
    if est_state_hist is not None:
        colors = ["salmon", "lightgreen", "lightblue"]
        est_lines = [ax.plot([], [], [], lw=1, linestyle="--", color=colors[k])[0] for k in range(3)]

    # 3. Goal: vector line OR target axes (cyan dotted)
    goal_vec_line = None
    goal_axes_lines = []
    goal_hist = None
    if boresight_goal_hist is not None:
        goal_hist = np.asarray(boresight_goal_hist, dtype=float)
        # Create both; we toggle visibility by clearing whichever isn't active.
        goal_vec_line = ax.plot([], [], [], lw=2, linestyle=":", color="cyan", label="Goal")[0]
        goal_axes_lines = [ax.plot([], [], [], lw=2, linestyle=":", color="cyan")[0] for _ in range(3)]

    # 4. Environment Quivers
    B_arrow = None
    S_arrow = None
    quiver_artists = []
    if os_hist is not None:
        B_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color="magenta", label="B-Field")
        S_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color="orange", label="Sun")
        quiver_artists = [B_arrow, S_arrow]

    # --- Legend (keep your “proxy legend” approach) ---
    if true_lines or est_lines or os_hist or goal_vec_line:
        proxies = []
        labels = []

        if state_hist is not None:
            proxies.append(plt.Line2D([0], [0], color="r", lw=2))
            labels.append("True Body X")  # matches your original style (only X shown)

        if est_state_hist is not None:
            proxies.append(plt.Line2D([0], [0], color="salmon", lw=1, linestyle="--"))
            labels.append("Est Body X")

        if boresight_goal_hist is not None:
            proxies.append(plt.Line2D([0], [0], color="cyan", lw=2, linestyle=":"))
            labels.append("Goal (Vec/Axes)")

        if os_hist is not None:
            proxies.extend(
                [plt.Line2D([0], [0], color="magenta", lw=2), plt.Line2D([0], [0], color="orange", lw=2)]
            )
            labels.extend(["Magnetic Field", "Sun Vector"])

        ax.legend(proxies, labels, loc="upper left")

    # --- Animation Control Variables ---
    frame = [0.0]
    play = [True]
    speed = [1.0]

    def init_anim():
        artists = []
        if true_lines:
            artists.extend(true_lines)
        if est_lines:
            artists.extend(est_lines)
        if goal_vec_line is not None:
            artists.append(goal_vec_line)
        if goal_axes_lines:
            artists.extend(goal_axes_lines)
        if quiver_artists:
            artists.extend(quiver_artists)
        return artists

    def update(_):
        nonlocal B_arrow, S_arrow

        # Build list of current artists (IMPORTANT: return even when paused)
        current_artists = []
        if true_lines:
            current_artists.extend(true_lines)
        if est_lines:
            current_artists.extend(est_lines)
        if goal_vec_line is not None:
            current_artists.append(goal_vec_line)
        if goal_axes_lines:
            current_artists.extend(goal_axes_lines)
        if B_arrow is not None:
            current_artists.append(B_arrow)
        if S_arrow is not None:
            current_artists.append(S_arrow)

        if not play[0]:
            return current_artists

        # Update Frame Index
        frame[0] = (frame[0] + speed[0]) % len(time)
        i = int(frame[0])

        # --- Update True Attitude ---
        if state_hist is not None:
            q_true = state_hist[i, 3:7]
            Rt = rot_mat(q_true)  # Body to ECI
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

        # --- Update Goal (vector OR axes) ---
        if goal_hist is not None:
            row = np.asarray(goal_hist[i], dtype=float).reshape(-1)

            # Legacy Nx3 -> vector goal
            if row.shape[0] == 3:
                g = _safe_unit(row)
                if g is not None and goal_vec_line is not None:
                    goal_vec_line.set_data([0, g[0]], [0, g[1]])
                    goal_vec_line.set_3d_properties([0, g[2]])
                elif goal_vec_line is not None:
                    _clear_line(goal_vec_line)

                for ln in goal_axes_lines:
                    _clear_line(ln)

            # Mixed Nx4
            elif row.shape[0] == 4:
                if np.isnan(row[0]):
                    # Vector goal: [nan, gx, gy, gz]
                    g = _safe_unit(row[1:4])
                    if g is not None and goal_vec_line is not None:
                        goal_vec_line.set_data([0, g[0]], [0, g[1]])
                        goal_vec_line.set_3d_properties([0, g[2]])
                    elif goal_vec_line is not None:
                        _clear_line(goal_vec_line)

                    # Hide axes
                    for ln in goal_axes_lines:
                        _clear_line(ln)

                else:
                    # Quaternion goal: [q0,q1,q2,q3] (Body->ECI)
                    qg = _safe_unit(row)  # normalize
                    # Hide vector
                    if goal_vec_line is not None:
                        _clear_line(goal_vec_line)

                    if qg is None:
                        for ln in goal_axes_lines:
                            _clear_line(ln)
                    else:
                        Rg = rot_mat(qg)
                        goal_ax = Rg @ body_axes
                        for k in range(3):
                            goal_axes_lines[k].set_data([0, goal_ax[0, k]], [0, goal_ax[1, k]])
                            goal_axes_lines[k].set_3d_properties([0, goal_ax[2, k]])
            else:
                # Unexpected shape: clear both
                if goal_vec_line is not None:
                    _clear_line(goal_vec_line)
                for ln in goal_axes_lines:
                    _clear_line(ln)

        # --- Update Environment Vectors ---
        if os_hist is not None:
            if B_arrow is not None:
                B_arrow.remove()
            if S_arrow is not None:
                S_arrow.remove()

            # Magnetic Field
            if hasattr(os_hist[i], "B") and os_hist[i].B is not None:
                norm_B = np.linalg.norm(os_hist[i].B)
                if norm_B > 1e-9:
                    B = os_hist[i].B / norm_B
                    B_arrow = ax.quiver(0, 0, 0, B[0], B[1], B[2], color="magenta")
                else:
                    B_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color="magenta", alpha=0)

            # Sun Vector
            if hasattr(os_hist[i], "S") and os_hist[i].S is not None:
                is_lit = True
                if hasattr(os_hist[i], "is_sunlit"):
                    is_lit = os_hist[i].is_sunlit()

                if is_lit:
                    norm_S = np.linalg.norm(os_hist[i].S)
                    if norm_S > 1e-9:
                        S = os_hist[i].S / norm_S
                        S_arrow = ax.quiver(0, 0, 0, S[0], S[1], S[2], color="orange")
                    else:
                        S_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color="orange", alpha=0)
                else:
                    S_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color="grey", alpha=0)

            # Rebuild current_artists because quivers were recreated
            current_artists = []
            if true_lines:
                current_artists.extend(true_lines)
            if est_lines:
                current_artists.extend(est_lines)
            if goal_vec_line is not None:
                current_artists.append(goal_vec_line)
            if goal_axes_lines:
                current_artists.extend(goal_axes_lines)
            if B_arrow is not None:
                current_artists.append(B_arrow)
            if S_arrow is not None:
                current_artists.append(S_arrow)

        return current_artists

    ani = FuncAnimation(fig, update, init_func=init_anim, interval=50)

    # --- UI Controls ---
    ax_pause = plt.axes([0.75, 0.02, 0.15, 0.05])
    btn_pause = Button(ax_pause, "Pause / Play")
    btn_pause.on_clicked(lambda e: play.__setitem__(0, not play[0]))

    ax_speed = plt.axes([0.02, 0.02, 0.20, 0.15])
    speed_buttons = RadioButtons(ax_speed, ["0.25x", "0.5x", "1x", "2x", "4x"], active=2)

    def set_speed(label):
        speed[0] = float(label.replace("x", ""))

    speed_buttons.on_clicked(set_speed)

    # Store animation to prevent garbage collection
    fig.animation = ani
    # (Your original didn't call plt.show(); keep/omit as your application expects)
