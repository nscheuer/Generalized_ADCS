import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection
from typing import List, Optional
from ADCS.orbits.orbital_state import Orbital_State

from ADCS.orbits.universal_constants import EarthConstants
from ADCS.helpers.math_helpers import rot_mat


def animate_orbit(
    time_hist: np.ndarray,
    state_hist: np.ndarray,
    os_hist: List[Orbital_State],
    est_state_hist: Optional[np.ndarray] = None,
    est_os_hist: Optional[List[Orbital_State]] = None
) -> None:
    """
    3D interactive orbit visualization.

    Shows:
      - True orbit trajectory (os_hist.R)
      - Optional estimated orbit trajectory (est_os_hist.R, dashed)
      - True spacecraft position marker
      - Optional estimated spacecraft position marker
      - True body axes at spacecraft position (from state_hist quaternions)
      - Optional estimated body axes (from est_state_hist)
      - Magnetic field vector at current position (os_hist[i].B, in ECI)
      - Sun vector at current position (os_hist[i].S, in ECI)
      - Earth sphere with correct radius and optional texture map
        rotated according to each state's time (J2000)

    Parameters
    ----------
    time_hist : np.ndarray
        1D array of time stamps [s], length N.
    state_hist : np.ndarray
        True state history, shape (N, >= 7). Columns 3:7 must be quaternion [q0..q3].
    os_hist : List[Orbital_State]
        List of true Orbital_State objects (length N).
    est_state_hist : np.ndarray or None
        Estimated state history (same shape/layout as state_hist), or None.
    est_os_hist : List[Orbital_State] or None
        List of estimated Orbital_State objects, or None.
    """
    time_hist = np.asarray(time_hist)
    N = len(time_hist)

    if len(os_hist) != N:
        raise ValueError("os_hist length must match time_hist length.")
    if est_state_hist is not None and len(est_state_hist) != N:
        raise ValueError("est_state_hist length must match time_hist length.")
    if est_os_hist is not None and len(est_os_hist) != N:
        raise ValueError("est_os_hist length must match time_hist length.")#
    
    earth_img = plt.imread("plotting/textures/2k_earth_daymap.jpg")

    # ---- Extract positions in ECI ----
    R_true = np.array([os.R for os in os_hist])
    R_est = None
    if est_os_hist is not None:
        R_est = np.array([os.R for os in est_os_hist])

    # ---- Figure & Axes ----
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

        # Normalize lon, lat to [0,1] texture coordinates
        # x: 0 -> lon=0, 1 -> lon=2π
        lon_norm = Lon / (2.0 * np.pi)
        # y: 0 -> north pole, 1 -> south pole
        lat_norm = (Lat + 0.5 * np.pi) / np.pi  # 0 at -π/2, 1 at +π/2 (south->north)
        # But images usually have row 0 at north, so flip:
        lat_norm = 1.0 - lat_norm

        x_idx = np.clip((lon_norm * (W - 1)).astype(int), 0, W - 1)
        y_idx = np.clip((lat_norm * (H - 1)).astype(int), 0, H - 1)

        facecolors = img[y_idx, x_idx] / 255.0 if img.max() > 1.0 else img[y_idx, x_idx]

    # Helper: rotate entire ECEF sphere grid to ECI for a given orbital state
    def ecef_grid_to_eci(os: Orbital_State):
        pts = np.stack([X_ecef.ravel(), Y_ecef.ravel(), Z_ecef.ravel()], axis=1)
        pts_eci = np.array([os.ecef_to_eci(p) for p in pts])
        X_eci = pts_eci[:, 0].reshape(X_ecef.shape)
        Y_eci = pts_eci[:, 1].reshape(Y_ecef.shape)
        Z_eci = pts_eci[:, 2].reshape(Z_ecef.shape)
        return X_eci, Y_eci, Z_eci

    # Initial Earth orientation (frame 0)
    X0, Y0, Z0 = ecef_grid_to_eci(os_hist[0])

    if facecolors is not None:
        earth_surf = ax.plot_surface(
            X0, Y0, Z0,
            rstride=1, cstride=1,
            facecolors=facecolors,
            linewidth=0,
            antialiased=False,
        )
    else:
        earth_surf = ax.plot_surface(
            X0, Y0, Z0,
            rstride=1, cstride=1,
            alpha=0.2,
            edgecolor="none",
            color="lightblue",
        )

    # ------------------------------------
    # Static orbit lines
    # ------------------------------------
    true_orbit_line, = ax.plot(
        R_true[:, 0], R_true[:, 1], R_true[:, 2],
        lw=1.5, color="tab:blue", label="True Orbit"
    )

    est_orbit_line = None
    if R_est is not None:
        est_orbit_line, = ax.plot(
            R_est[:, 0], R_est[:, 1], R_est[:, 2],
            lw=1.0, linestyle="--", color="tab:orange", label="Est Orbit"
        )

    # ------------------------------------
    # Dynamic spacecraft position markers
    # ------------------------------------
    true_pos_marker, = ax.plot([], [], [], "o", color="blue", label="True Position")
    est_pos_marker = None
    if R_est is not None:
        est_pos_marker, = ax.plot([], [], [], "o", color="orange", label="Est Position")

    # ------------------------------------
    # Body axes
    # ------------------------------------
    body_axes = np.eye(3)
    axis_scale = 0.1 * max_r

    true_axes_lines = []
    colors_true = ["r", "g", "b"]
    for k in range(3):
        ln, = ax.plot([], [], [], lw=2, color=colors_true[k])
        true_axes_lines.append(ln)

    est_axes_lines = []
    if est_state_hist is not None:
        colors_est = ["salmon", "lightgreen", "lightblue"]
        for k in range(3):
            ln, = ax.plot([], [], [], lw=1, linestyle="--", color=colors_est[k])
            est_axes_lines.append(ln)

    # ------------------------------------
    # Magnetic Field & Sun Vectors
    # ------------------------------------
    B_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color="magenta")
    S_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color="yellow")

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
        legend_handles.append(plt.Line2D([0], [0], color="salmon", lw=1, linestyle="--"))
        legend_labels.append("Est Body Axes")

    legend_handles.append(plt.Line2D([0], [0], color="magenta", lw=2))
    legend_labels.append("Magnetic Field")

    legend_handles.append(plt.Line2D([0], [0], color="yellow", lw=2))
    legend_labels.append("Sun Vector")

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
        artists.append(B_arrow)
        artists.append(S_arrow)
        return artists

    # ------------------------------------
    # Update Function
    # ------------------------------------
    def update(_):
        nonlocal earth_surf, B_arrow, S_arrow

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
                Xk, Yk, Zk,
                rstride=1, cstride=1,
                facecolors=facecolors,
                linewidth=0,
                antialiased=False,
            )
        else:
            earth_surf = ax.plot_surface(
                Xk, Yk, Zk,
                rstride=1, cstride=1,
                alpha=0.2,
                edgecolor="none",
                color="lightblue",
            )
        artists.append(earth_surf)

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
            true_axes_lines[k].set_data([pos_true[0], end[0]], [pos_true[1], end[1]])
            true_axes_lines[k].set_3d_properties([pos_true[2], end[2]])
            artists.append(true_axes_lines[k])

        # --- Estimated body axes ---
        if est_state_hist is not None and est_axes_lines:
            q_est = est_state_hist[i, 3:7]
            Re = rot_mat(q_est)
            axes_est = Re @ body_axes
            for k in range(3):
                end = pos_true + axis_scale * axes_est[:, k]
                est_axes_lines[k].set_data([pos_true[0], end[0]],
                                           [pos_true[1], end[1]])
                est_axes_lines[k].set_3d_properties([pos_true[2], end[2]])
                artists.append(est_axes_lines[k])

        # --- Magnetic Field Vector ---
        B_vec = getattr(os_i, "B", None)
        if B_vec is not None:
            B_arrow.remove()
            if np.linalg.norm(B_vec) > 1e-12:
                Bn = B_vec / np.linalg.norm(B_vec)
                B_arrow = ax.quiver(
                    pos_true[0], pos_true[1], pos_true[2],
                    Bn[0] * 0.3 * max_r,
                    Bn[1] * 0.3 * max_r,
                    Bn[2] * 0.3 * max_r,
                    color="magenta",
                )
                artists.append(B_arrow)

        # --- Sun Vector ---
        S_vec = getattr(os_i, "S", None)
        if S_vec is not None:
            S_arrow.remove()
            if np.linalg.norm(S_vec) > 1e-12:
                Sn = S_vec / np.linalg.norm(S_vec)
                S_arrow = ax.quiver(
                    pos_true[0], pos_true[1], pos_true[2],
                    Sn[0] * 0.3 * max_r,
                    Sn[1] * 0.3 * max_r,
                    Sn[2] * 0.3 * max_r,
                    color="yellow",
                )
                artists.append(S_arrow)

        return artists

    ani = FuncAnimation(fig, update, init_func=init_anim, interval=50, blit=False)
    fig.animation = ani  # prevent GC

    # ------------------------------------
    # UI Controls
    # ------------------------------------
    ax_pause = plt.axes([0.75, 0.02, 0.15, 0.05])
    btn_pause = Button(ax_pause, "Pause / Play")
    btn_pause.on_clicked(lambda e: play_state.__setitem__(0, not play_state[0]))

    ax_speed = plt.axes([0.02, 0.02, 0.20, 0.15])
    speed_buttons = RadioButtons(ax_speed, ["0.25x", "0.5x", "1x", "2x", "4x"], active=2)

    def set_speed(label: str):
        mapping = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "2x": 2.0, "4x": 4.0}
        speed_factor[0] = mapping.get(label, 1.0)

    speed_buttons.on_clicked(set_speed)
