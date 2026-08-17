__all__ = ["animate_orbit"]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection
from typing import List, Optional

from ADCS.CONOPS.goals import Goal, ECI_Goal, Coordinate_Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants
from ADCS.helpers.math_helpers import rot_mat
from ADCS.state import State

def animate_orbit(
    time_hist: np.ndarray,
    state_hist: List[State],
    os_hist: List[Orbital_State],
    est_state_hist: Optional[List[State]] = None,
    est_os_hist: Optional[List[Orbital_State]] = None,
    boresight_goal_hist: Optional[np.ndarray] = None,
    coord_goal: Optional[Coordinate_Goal]=None,  # Expected: Coordinate_Goal instance (has .target_ecef)
) -> None:
    r"""
    Animate a spacecraft orbit, attitude, environment vectors, and mission goals in 3D.

    This function produces an interactive 3D Matplotlib animation in the
    Earth-Centered Inertial (ECI) frame, visualizing orbital motion, spacecraft
    attitude, environmental vectors, and optional mission goals. It is intended
    for post-simulation analysis and debugging of Guidance, Navigation, and
    Control (GNC) and Attitude Determination and Control System (ADCS) behavior.

    The animation integrates information from orbital dynamics, spacecraft
    attitude quaternions, and mission goal definitions, all synchronized over a
    common discrete time history.

    ======================
    Reference Frames
    ======================

    The following frames are used throughout the visualization:

    +------------------+--------------------------------------------------+
    | Frame            | Description                                      |
    +==================+==================================================+
    | ECI              | Earth-Centered Inertial reference frame          |
    +------------------+--------------------------------------------------+
    | ECEF             | Earth-Centered Earth-Fixed rotating frame        |
    +------------------+--------------------------------------------------+
    | Body             | Spacecraft body-fixed frame                      |
    +------------------+--------------------------------------------------+

    All vectors and trajectories are ultimately rendered in the ECI frame.
    Earth-fixed quantities are transformed to ECI using orbital state methods.

    ======================
    Orbit Visualization
    ======================

    The spacecraft position in ECI is extracted from each
    :class:`~ADCS.orbits.orbital_state.Orbital_State` object:

    .. math::

        \mathbf{r}_{\text{ECI}}(t_i) \in \mathbb{R}^3

    The true orbit trajectory is drawn as a solid line, while the estimated
    orbit (if provided) is drawn as a dashed line for comparison.

    ======================
    Earth Rendering
    ======================

    The Earth is modeled as a sphere of radius

    .. math::

        R_e = \text{EarthConstants.R\_e}

    defined in the Earth-Centered Earth-Fixed (ECEF) frame. A longitude–latitude
    grid is constructed as

    .. math::

        \begin{aligned}
        x &= R_e \cos\phi \cos\lambda \\
        y &= R_e \cos\phi \sin\lambda \\
        z &= R_e \sin\phi
        \end{aligned}

    where :math:`\lambda` is longitude and :math:`\phi` is latitude.

    At each time step, the Earth grid is rotated into the ECI frame using

    .. math::

        \mathbf{r}_{\text{ECI}} =
        \mathcal{T}_{\text{ECEF}\rightarrow\text{ECI}}(\mathbf{r}_{\text{ECEF}})

    via :meth:`~ADCS.orbits.orbital_state.Orbital_State.ecef_to_eci`.

    =========================
    Attitude Representation
    =========================

    Spacecraft attitude is represented by a unit quaternion

    .. math::

        \mathbf{q} =
        \begin{bmatrix}
        q_0 & q_1 & q_2 & q_3
        \end{bmatrix}^T

    stored in columns ``[3:7]`` of the state history. The quaternion is converted
    to a direction cosine matrix using
    :func:`~ADCS.helpers.math_helpers.rot_mat`:

    .. math::

        \mathbf{R}_{\mathcal{B}\rightarrow\mathcal{I}}(\mathbf{q})

    The body-frame basis vectors

    .. math::

        \mathbf{I}_3 =
        \begin{bmatrix}
        1 & 0 & 0 \\
        0 & 1 & 0 \\
        0 & 0 & 1
        \end{bmatrix}

    are mapped into ECI as

    .. math::

        \mathbf{A}_i =
        \mathbf{R}_{\mathcal{B}\rightarrow\mathcal{I}}(\mathbf{q}_i)\,\mathbf{I}_3

    yielding the inertial directions of the body X, Y, and Z axes. These axes are
    rendered as colored line segments originating at the spacecraft position.

    ======================
    Environmental Vectors
    ======================

    If present in the orbital state, the following inertial vectors are shown:

    * Magnetic field vector :math:`\mathbf{B}`
    * Sun direction vector :math:`\mathbf{S}`

    Each vector is normalized and rendered as an arrow:

    .. math::

        \hat{\mathbf{v}} =
        \frac{\mathbf{v}}{\|\mathbf{v}\|}

    ==================================
    Boresight and Coordinate Goals
    ==================================

    **Boresight Goal**

    A time history of desired line-of-sight vectors
    :math:`\mathbf{g}_i^{\text{ECI}}` may be provided. Each vector is normalized
    and drawn from the spacecraft position:

    .. math::

        \hat{\mathbf{g}}_i =
        \frac{\mathbf{g}_i}{\|\mathbf{g}_i\|}

    **Coordinate Goal**

    A fixed Earth-surface target defined by
    :class:`~ADCS.CONOPS.goals.vector_goals.coordinate_goal.Coordinate_Goal`
    is rendered as a large sphere attached to the Earth. Its ECEF position

    .. math::

        \mathbf{r}_{\text{goal}}^{\text{ECEF}}

    is rotated into ECI each frame using

    .. math::

        \mathbf{r}_{\text{goal}}^{\text{ECI}}(t) =
        \mathcal{T}_{\text{ECEF}\rightarrow\text{ECI}}
        \left(\mathbf{r}_{\text{goal}}^{\text{ECEF}}\right)

    ======================
    User Interaction
    ======================

    The animation includes interactive controls:

    * Pause / play toggle
    * Playback speed selection:
      :math:`\{0.25\times, 0.5\times, 1\times, 2\times, 4\times\}`

    These controls affect visualization timing only and do not alter the data.

    :param time_hist:
        One-dimensional array of simulation time stamps in seconds.
    :type time_hist:
        numpy.ndarray

    :param state_hist:
        True spacecraft state history. Quaternion attitude must be stored in
        columns ``[3:7]``.
    :type state_hist:
        numpy.ndarray

    :param os_hist:
        List of true orbital state objects providing position, Earth rotation,
        and environmental vectors.
    :type os_hist:
        list of :class:`~ADCS.orbits.orbital_state.Orbital_State`

    :param est_state_hist:
        Optional estimated spacecraft state history with the same layout as
        ``state_hist``.
    :type est_state_hist:
        numpy.ndarray or None

    :param est_os_hist:
        Optional estimated orbital state history.
    :type est_os_hist:
        list of :class:`~ADCS.orbits.orbital_state.Orbital_State` or None

    :param boresight_goal_hist:
        Optional time history of desired boresight direction vectors in ECI.
    :type boresight_goal_hist:
        numpy.ndarray or None

    :param coord_goal:
        Optional Earth-fixed coordinate goal defining a surface target.
    :type coord_goal:
        :class:`~ADCS.CONOPS.goals.vector_goals.coordinate_goal.Coordinate_Goal`
        or None

    :return:
        None. The function creates and displays an interactive animation.
    :rtype:
        None

    """
    time_hist = np.asarray(time_hist)
    state_hist = State.stack(state_hist)
    if est_state_hist is not None:
        est_state_hist = State.stack(est_state_hist)
    N = len(time_hist)

    if len(os_hist) != N:
        raise ValueError("os_hist length must match time_hist length.")
    if est_state_hist is not None and len(est_state_hist) != N:
        raise ValueError("est_state_hist length must match time_hist length.")
    if est_os_hist is not None and len(est_os_hist) != N:
        raise ValueError("est_os_hist length must match time_hist length.")
    if boresight_goal_hist is not None:
        boresight_goal_hist = np.asarray(boresight_goal_hist)
        if boresight_goal_hist.shape[0] != N or boresight_goal_hist.shape[1] != 3:
            raise ValueError(
                "boresight_goal_hist must have shape (N, 3), where N = len(time_hist)."
            )

    # Earth texture (assumed to exist at this path)
    earth_img = plt.imread("plotting/textures/2k_earth_daymap.jpg")

    # ---- Extract positions in ECI ----
    R_true = np.array([os.R for os in os_hist])
    R_est = None
    if est_os_hist is not None:
        R_est = np.array([os.R for os in est_os_hist])

    # -----------------------------
    # Figure & Axes
    # -----------------------------
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Scaling from all orbit points
    all_R = [R_true]
    if R_est is not None:
        all_R.append(R_est)
    all_R = np.vstack(all_R)
    max_r = np.max(np.linalg.norm(all_R, axis=1))
    lim = max(max_r * 1.1, EarthConstants.R_e * 1.1)

    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass

    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.set_title("Orbit Visualization")

    # ============================================================
    # EARTH SPHERE (built once in ECEF, rotated each frame to ECI)
    # ============================================================
    R_e = EarthConstants.R_e

    # Sphere grid in ECEF: lon λ ∈ [0, 2π], lat φ ∈ [-π/2, π/2]
    n_lon = 72
    n_lat = 36
    lon = np.linspace(0.0, 2.0 * np.pi, n_lon)
    lat = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_lat)
    Lon, Lat = np.meshgrid(lon, lat)

    X_ecef = R_e * np.cos(Lat) * np.cos(Lon)
    Y_ecef = R_e * np.cos(Lat) * np.sin(Lon)
    Z_ecef = R_e * np.sin(Lat)

    # Optional texture: build facecolors via equirectangular mapping
    facecolors = None
    if earth_img is not None:
        img = np.asarray(earth_img)
        H, W = img.shape[0], img.shape[1]

        lon_norm = (Lon + np.pi) / (2.0 * np.pi)
        lat_norm = (Lat + 0.5 * np.pi) / np.pi
        lat_norm = 1.0 - lat_norm  # flip so row 0 is north

        x_idx = np.clip((lon_norm * (W - 1)).astype(int), 0, W - 1)
        y_idx = np.clip((lat_norm * (H - 1)).astype(int), 0, H - 1)

        facecolors = (
            img[y_idx, x_idx] / 255.0 if img.max() > 1.0 else img[y_idx, x_idx]
        )

    # Helper: rotate entire ECEF sphere grid to ECI for a given orbital state
    def ecef_grid_to_eci(os: Orbital_State):
        pts = np.stack(
            [X_ecef.ravel(), Y_ecef.ravel(), Z_ecef.ravel()],
            axis=1,
        )
        pts_eci = np.array([os.ecef_to_eci(p) for p in pts])
        X_eci = pts_eci[:, 0].reshape(X_ecef.shape)
        Y_eci = pts_eci[:, 1].reshape(Y_ecef.shape)
        Z_eci = pts_eci[:, 2].reshape(Z_ecef.shape)
        return X_eci, Y_eci, Z_eci

    # Initial Earth orientation (frame 0)
    X0, Y0, Z0 = ecef_grid_to_eci(os_hist[0])

    # Draw Earth sphere
    if facecolors is not None:
        earth_surf = ax.plot_surface(
            X0,
            Y0,
            Z0,
            rstride=1,
            cstride=1,
            facecolors=facecolors,
            linewidth=0,
            antialiased=False,
            zorder=0,
            alpha=0.9,
        )
    else:
        earth_surf = ax.plot_surface(
            X0,
            Y0,
            Z0,
            rstride=1,
            cstride=1,
            alpha=0.3,
            edgecolor="none",
            color="lightblue",
            zorder=0,
        )

    # ============================================================
    # GOAL SPHERE (fixed in ECEF, rotated each frame to ECI)
    # ============================================================
    goal_surf = None
    X_goal_local = Y_goal_local = Z_goal_local = None
    if coord_goal is not None:
        # "Large" sphere relative to Earth radius; tweak as you like
        goal_radius = 0.2 * R_e

        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 15)
        U, V = np.meshgrid(u, v)

        X_goal_local = goal_radius * np.cos(U) * np.sin(V)
        Y_goal_local = goal_radius * np.sin(U) * np.sin(V)
        Z_goal_local = goal_radius * np.cos(V)
        # We will translate this local sphere by the goal center in ECI each frame

    # ------------------------------------
    # Static orbit lines
    # ------------------------------------
    true_orbit_line, = ax.plot(
        R_true[:, 0],
        R_true[:, 1],
        R_true[:, 2],
        lw=1.5,
        color="tab:red",
        label="True Orbit",
        zorder=5,
    )

    est_orbit_line = None
    if R_est is not None:
        est_orbit_line, = ax.plot(
            R_est[:, 0],
            R_est[:, 1],
            R_est[:, 2],
            lw=1.0,
            linestyle="--",
            color="tab:orange",
            label="Est Orbit",
            zorder=5,
        )

    # ------------------------------------
    # Dynamic spacecraft position markers
    # ------------------------------------
    true_pos_marker, = ax.plot(
        [],
        [],
        [],
        "o",
        color="blue",
        label="True Position",
        zorder=6,
    )

    est_pos_marker = None
    if R_est is not None:
        est_pos_marker, = ax.plot(
            [],
            [],
            [],
            "o",
            color="orange",
            label="Est Position",
            zorder=6,
        )

    # ------------------------------------
    # Body axes
    # ------------------------------------
    body_axes = np.eye(3)
    axis_scale = 0.1 * max_r

    true_axes_lines = []
    colors_true = ["r", "g", "b"]
    for k in range(3):
        ln, = ax.plot(
            [],
            [],
            [],
            lw=2,
            color=colors_true[k],
            zorder=6,
        )
        true_axes_lines.append(ln)

    est_axes_lines = []
    if est_state_hist is not None:
        colors_est = ["salmon", "lightgreen", "lightblue"]
        for k in range(3):
            ln, = ax.plot(
                [],
                [],
                [],
                lw=1,
                linestyle="--",
                color=colors_est[k],
                zorder=6,
            )
            est_axes_lines.append(ln)

    # ------------------------------------
    # Boresight goal LOS line
    # ------------------------------------
    boresight_goal_line = None
    if boresight_goal_hist is not None:
        boresight_goal_line, = ax.plot(
            [],
            [],
            [],
            lw=2,
            linestyle=":",
            color="cyan",
            zorder=7,
            label="Boresight Goal",
        )

    # ------------------------------------
    # Magnetic Field & Sun Vectors
    # ------------------------------------
    B_arrow = ax.quiver(
        0,
        0,
        0,
        0,
        0,
        0,
        color="magenta",
        zorder=7,
    )
    S_arrow = ax.quiver(
        0,
        0,
        0,
        0,
        0,
        0,
        color="yellow",
        zorder=7,
    )

    # ------------------------------------
    # Legend
    # ------------------------------------
    legend_handles = [true_orbit_line, true_pos_marker]
    legend_labels = ["True Orbit", "True Position"]

    if est_orbit_line is not None:
        legend_handles.append(est_orbit_line)
        legend_labels.append("Est Orbit")
    if est_pos_marker is not None:
        legend_handles.append(est_pos_marker)
        legend_labels.append("Est Position")

    legend_handles.append(plt.Line2D([0], [0], color="r", lw=2))
    legend_labels.append("True Body Axes")

    if est_axes_lines:
        legend_handles.append(
            plt.Line2D([0], [0], color="salmon", lw=1, linestyle="--")
        )
        legend_labels.append("Est Body Axes")

    if boresight_goal_line is not None:
        legend_handles.append(boresight_goal_line)
        legend_labels.append("Boresight Goal")

    legend_handles.append(plt.Line2D([0], [0], color="magenta", lw=2))
    legend_labels.append("Magnetic Field")

    legend_handles.append(plt.Line2D([0], [0], color="yellow", lw=2))
    legend_labels.append("Sun Vector")

    if coord_goal is not None:
        # Dummy handle for goal sphere
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color="lightblue", linestyle="", markersize=8)
        )
        legend_labels.append("Coordinate Goal")

    ax.legend(legend_handles, legend_labels, loc="upper right")

    # ------------------------------------
    # Animation State
    # ------------------------------------
    frame_idx = [0.0]
    play_state = [True]
    speed_factor = [1.0]

    def init_anim():
        artists = [true_pos_marker]
        if est_pos_marker is not None:
            artists.append(est_pos_marker)
        artists.extend(true_axes_lines)
        artists.extend(est_axes_lines)
        artists.append(earth_surf)
        if boresight_goal_line is not None:
            artists.append(boresight_goal_line)
        artists.append(B_arrow)
        artists.append(S_arrow)
        # goal_surf will be created in update, so nothing to add yet
        return artists

    # ------------------------------------
    # Update Function
    # ------------------------------------
    def update(_):
        nonlocal earth_surf, B_arrow, S_arrow, goal_surf

        if not play_state[0]:
            return init_anim()

        # Advance frame index
        frame_idx[0] = (frame_idx[0] + speed_factor[0]) % N
        i = int(frame_idx[0])

        os_i = os_hist[i]
        pos_true = R_true[i]

        artists = []

        # --- Update Earth orientation ---
        earth_surf.remove()
        Xk, Yk, Zk = ecef_grid_to_eci(os_i)
        if facecolors is not None:
            earth_surf = ax.plot_surface(
                Xk,
                Yk,
                Zk,
                rstride=1,
                cstride=1,
                facecolors=facecolors,
                linewidth=0,
                antialiased=False,
                zorder=0,
                alpha=0.9,
            )
        else:
            earth_surf = ax.plot_surface(
                Xk,
                Yk,
                Zk,
                rstride=1,
                cstride=1,
                alpha=0.3,
                edgecolor="none",
                color="lightblue",
                zorder=0,
            )
        artists.append(earth_surf)

        # --- Goal sphere (rotates with Earth via ecef_to_eci) ---
        if coord_goal is not None and X_goal_local is not None:
            if goal_surf is not None:
                goal_surf.remove()
            # Goal center in ECI at current time
            goal_center_eci = os_i.ecef_to_eci(coord_goal.target_ecef)
            Xg = X_goal_local + goal_center_eci[0]
            Yg = Y_goal_local + goal_center_eci[1]
            Zg = Z_goal_local + goal_center_eci[2]
            goal_surf = ax.plot_surface(
                Xg,
                Yg,
                Zg,
                rstride=1,
                cstride=1,
                alpha=0.6,
                edgecolor="none",
                color="lightblue",
                zorder=8,
            )
            artists.append(goal_surf)

        # --- True position marker ---
        true_pos_marker.set_data([pos_true[0]], [pos_true[1]])
        true_pos_marker.set_3d_properties([pos_true[2]])
        artists.append(true_pos_marker)

        # --- Estimated position marker ---
        if R_est is not None and est_pos_marker is not None:
            pos_est = R_est[i]
            est_pos_marker.set_data([pos_est[0]], [pos_est[1]])
            est_pos_marker.set_3d_properties([pos_est[2]])
            artists.append(est_pos_marker)

        # --- True body axes ---
        q_true = state_hist[i, 3:7]
        Rt = rot_mat(q_true)  # body -> ECI
        axes_true = Rt @ body_axes
        for k in range(3):
            end = pos_true + axis_scale * axes_true[:, k]
            true_axes_lines[k].set_data(
                [pos_true[0], end[0]],
                [pos_true[1], end[1]],
            )
            true_axes_lines[k].set_3d_properties([pos_true[2], end[2]])
            artists.append(true_axes_lines[k])

        # --- Estimated body axes ---
        if est_state_hist is not None and est_axes_lines:
            q_est = est_state_hist[i, 3:7]
            Re = rot_mat(q_est)
            axes_est = Re @ body_axes
            for k in range(3):
                end = pos_true + axis_scale * axes_est[:, k]
                est_axes_lines[k].set_data(
                    [pos_true[0], end[0]],
                    [pos_true[1], end[1]],
                )
                est_axes_lines[k].set_3d_properties([pos_true[2], end[2]])
                artists.append(est_axes_lines[k])

        # --- Boresight goal LOS ---
        if boresight_goal_hist is not None and boresight_goal_line is not None:
            v_goal = boresight_goal_hist[i]
            if np.all(np.isfinite(v_goal)) and np.linalg.norm(v_goal) > 1e-12:
                v_goal_n = v_goal / np.linalg.norm(v_goal)
                end = pos_true + axis_scale * v_goal_n
                boresight_goal_line.set_data(
                    [pos_true[0], end[0]],
                    [pos_true[1], end[1]],
                )
                boresight_goal_line.set_3d_properties(
                    [pos_true[2], end[2]]
                )
            else:
                boresight_goal_line.set_data([], [])
                boresight_goal_line.set_3d_properties([])
            artists.append(boresight_goal_line)

        # --- Magnetic Field Vector ---
        B_vec = getattr(os_i, "B", None)
        if B_vec is not None:
            B_arrow.remove()
            if np.linalg.norm(B_vec) > 1e-12:
                Bn = B_vec / np.linalg.norm(B_vec)
                B_arrow = ax.quiver(
                    pos_true[0],
                    pos_true[1],
                    pos_true[2],
                    Bn[0] * 0.3 * max_r,
                    Bn[1] * 0.3 * max_r,
                    Bn[2] * 0.3 * max_r,
                    color="magenta",
                    zorder=7,
                )
                artists.append(B_arrow)

        # --- Sun Vector ---
        S_vec = getattr(os_i, "S", None)
        if S_vec is not None:
            S_arrow.remove()
            if np.linalg.norm(S_vec) > 1e-12:
                Sn = S_vec / np.linalg.norm(S_vec)
                S_arrow = ax.quiver(
                    pos_true[0],
                    pos_true[1],
                    pos_true[2],
                    Sn[0] * 0.3 * max_r,
                    Sn[1] * 0.3 * max_r,
                    Sn[2] * 0.3 * max_r,
                    color="yellow",
                    zorder=7,
                )
                artists.append(S_arrow)

        return artists

    ani = FuncAnimation(
        fig,
        update,
        init_func=init_anim,
        interval=50,
        blit=False,
    )
    fig.animation = ani  # prevent GC

    # ------------------------------------
    # UI Controls
    # ------------------------------------
    ax_pause = plt.axes([0.75, 0.02, 0.15, 0.05])
    btn_pause = Button(ax_pause, "Pause / Play")
    btn_pause.on_clicked(lambda e: play_state.__setitem__(0, not play_state[0]))

    ax_speed = plt.axes([0.02, 0.02, 0.20, 0.15])
    speed_buttons = RadioButtons(
        ax_speed,
        ["0.25x", "0.5x", "1x", "2x", "4x"],
        active=2,
    )

    def set_speed(label: str):
        mapping = {
            "0.25x": 0.25,
            "0.5x": 0.5,
            "1x": 1.0,
            "2x": 2.0,
            "4x": 4.0,
        }
        speed_factor[0] = mapping.get(label, 1.0)

    speed_buttons.on_clicked(set_speed)
