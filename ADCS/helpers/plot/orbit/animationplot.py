__all__ = ["AnimationPlot"]

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from ..subplot import Subplot

# Keep PyVista happy in headless / CI (safe even on desktop)
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def _get_rotation_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping vec1 -> vec2 (both in R^3)."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < 1e-12 or norm2 < 1e-12:
        return np.eye(3)

    a = vec1 / norm1
    b = vec2 / norm2
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)

    if s < 1e-6:
        return np.eye(3) if c > 0 else -np.eye(3)

    kmat = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=float,
    )
    return np.eye(3) + kmat + (kmat @ kmat) * ((1.0 - c) / (s**2))


def _unwrap_os(os_obj):
    """Unwrap EstimatedOrbital_State-like objects that store `.os`."""
    return os_obj.os if hasattr(os_obj, "os") else os_obj


def _pick_goal_history(sim):
    """
    Prefer sim.eci_target_hist (your sim_results.record uses eci_target=... each step),
    but fall back to a few common names.
    """
    for name in ("eci_target_hist", "eci_target", "boresight_goal_hist"):
        v = getattr(sim, name, None)
        if v is None:
            continue
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0:
            return v, name
    return None, None


class AnimationPlot(Subplot):
    """
    Subplot-compatible wrapper that launches the PyVista orbit/attitude animation.

    - Opens its own PyVista render window (cannot draw into matplotlib Axes).
    - Always uses the goal history from `sim` (defaults to `sim.eci_target_hist`).
    - Optional coordinate goal marker is plotted only if it is a Coordinate_Goal.
    """

    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "3D Animation (PyVista)",
        texture_path: Optional[str | Path] = None,
        texture_alignment_angle_deg: float = -180.0,
        smooth_factor: int = 4,
        min_smooth_N: int = 1000,
        show_true_orbit: bool = True,
        show_est_orbit: bool = True,
        show_body_axes: bool = True,
        show_env_vectors: bool = True,
        goal=None,  # Optional Goal object; if it's Coordinate_Goal, draw surface marker
        window_size=(1200, 900),
        axis_scale_body: float = 0.3,
        axis_scale_sun: float = 0.8,
        axis_scale_mag: float = 0.6,
        axis_scale_goal: float = 1.0,
        axis_scale_base_mult: float = 0.5,  # multiplied by Earth radius
    ):
        self.time = time
        self.title = title
        self.texture_path = texture_path
        self.texture_alignment_angle_deg = float(texture_alignment_angle_deg)
        self.smooth_factor = int(smooth_factor)
        self.min_smooth_N = int(min_smooth_N)

        self.show_true_orbit = bool(show_true_orbit)
        self.show_est_orbit = bool(show_est_orbit)
        self.show_body_axes = bool(show_body_axes)
        self.show_env_vectors = bool(show_env_vectors)

        self.goal = goal
        self.window_size = tuple(window_size)

        self.axis_scale_body = float(axis_scale_body)
        self.axis_scale_sun = float(axis_scale_sun)
        self.axis_scale_mag = float(axis_scale_mag)
        self.axis_scale_goal = float(axis_scale_goal)
        self.axis_scale_base_mult = float(axis_scale_base_mult)

    def plot(self, ax, sim) -> None:
        ax.axis("off")
        ax.set_title(self.title, loc="left", pad=10)
        ax.text(
            0.5,
            0.5,
            "Launching PyVista animation window…\n(close it to continue)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        self._run_pyvista(sim)

    def _run_pyvista(self, sim) -> None:
        # Keep imports local so importing plotting package doesn't hard-require these deps.
        import pyvista as pv
        from pyvista import examples
        import matplotlib.pyplot as plt
        from scipy.interpolate import interp1d
        from scipy.spatial.transform import Rotation as R_scipy
        from scipy.spatial.transform import Slerp

        # Earth radius (try sim's constants if present; otherwise fall back)
        try:
            from ADCS.orbits.universal_constants import EarthConstants

            R_e = float(EarthConstants.R_e)
        except Exception:
            R_e = 6378.137  # km fallback

        # Coordinate_Goal type check (optional dependency)
        try:
            from ADCS.CONOPS.goals import Coordinate_Goal
        except Exception:
            Coordinate_Goal = None

        # ---- Pull required histories from sim ----
        t_orig = np.asarray(getattr(sim, self.time, None))
        if t_orig is None or t_orig.size == 0:
            raise ValueError(f"sim.{self.time} missing/empty")

        os_hist = getattr(sim, "os_hist", None)
        if os_hist is None or len(os_hist) == 0:
            raise ValueError("sim.os_hist missing/empty (required for animation)")

        state_hist = getattr(sim, "state_hist", None)
        if state_hist is None or len(state_hist) == 0:
            raise ValueError("sim.state_hist missing/empty (required for animation)")

        state_hist = np.asarray(state_hist)
        if state_hist.ndim != 2 or state_hist.shape[1] < 7:
            raise ValueError("sim.state_hist must be (N, >=7) with quaternion in columns [3:7]")

        # Optional estimated orbit for polyline display
        est_os_hist = getattr(sim, "est_os_hist", None)

        # Always use goal history from sim
        G_hist, G_name = _pick_goal_history(sim)

        # ---- Smooth time grid ----
        original_N = int(len(t_orig))
        target_N = max(original_N * self.smooth_factor, self.min_smooth_N)
        t_new = np.linspace(float(t_orig[0]), float(t_orig[-1]), target_N)

        def interp_arr(arr: np.ndarray) -> np.ndarray:
            f = interp1d(t_orig, arr, axis=0, kind="linear", fill_value="extrapolate")
            return f(t_new)

        # ---- Physics (true) ----
        R_true_orig = np.array([np.asarray(os.R, dtype=float).reshape(3) for os in os_hist])
        R_true_smooth = interp_arr(R_true_orig)

        q_orig = np.asarray(state_hist[:, 3:7], dtype=float)
        q_smooth = interp_arr(q_orig)
        q_smooth = q_smooth / np.linalg.norm(q_smooth, axis=1, keepdims=True)

        # Environment vectors (if present on os)
        S_orig = np.array(
            [np.asarray(getattr(os, "S", [0, 0, 0]), dtype=float).reshape(3) for os in os_hist]
        )
        B_orig = np.array(
            [np.asarray(getattr(os, "B", [0, 0, 0]), dtype=float).reshape(3) for os in os_hist]
        )
        S_smooth = interp_arr(S_orig)
        B_smooth = interp_arr(B_orig)

        # Goal direction: use sim.eci_target_hist by default.
        # If it's a direction already, great. If it's a target *position*, convert to direction (target - sat_pos).
        G_smooth = None
        if G_hist is not None:
            G_arr = np.asarray(G_hist, dtype=float)
            if G_arr.ndim == 1 and G_arr.size == 3:
                G_arr = np.repeat(G_arr.reshape(1, 3), len(t_orig), axis=0)
            G_arr = G_arr.reshape(-1, 3)

            # Heuristic: if magnitudes look like "position" (km scale), convert to direction from sat to target.
            med_norm = float(np.nanmedian(np.linalg.norm(G_arr, axis=1)))
            if med_norm > 10.0:  # likely position in km (not a unit vector)
                Nmin = min(len(G_arr), len(R_true_orig))
                G_arr = G_arr[:Nmin] - R_true_orig[:Nmin]

            # Interp to smooth timeline
            G_smooth = interp_arr(G_arr)

        # Estimated orbit for static polyline display
        R_est_static = None
        if est_os_hist is not None and len(est_os_hist) > 0:
            rows = []
            for os in est_os_hist:
                if os is None:
                    continue
                base = _unwrap_os(os)
                rows.append(np.asarray(base.R, dtype=float).reshape(3))
            if rows:
                R_est_static = np.vstack(rows)

        # ---- Earth rotation (ECEF->ECI) from Orbital_State.ecef_to_eci ----
        basis_x = np.array([1.0, 0.0, 0.0])
        basis_y = np.array([0.0, 1.0, 0.0])
        basis_z = np.array([0.0, 0.0, 1.0])

        earth_rot_mats = []
        for os_item in os_hist:
            col0 = np.asarray(os_item.ecef_to_eci(basis_x), dtype=float).reshape(3)
            col1 = np.asarray(os_item.ecef_to_eci(basis_y), dtype=float).reshape(3)
            col2 = np.asarray(os_item.ecef_to_eci(basis_z), dtype=float).reshape(3)
            earth_rot_mats.append(np.column_stack((col0, col1, col2)))

        rot_obj = R_scipy.from_matrix(np.asarray(earth_rot_mats))
        earth_slerp = Slerp(t_orig, rot_obj)
        earth_rot_smooth = earth_slerp(t_new)

        # ---- PyVista scene ----
        pv.global_theme.multi_samples = 0
        pl = pv.Plotter(window_size=list(self.window_size), lighting="three lights")
        pl.set_background("black")

        earth_mesh = pv.Sphere(radius=R_e, theta_resolution=120, phi_resolution=120)

        # Texture
        default_texture = Path(__file__).resolve().parent / "textures" / "2k_earth_daymap.jpg"
        texture_path = (
            Path(self.texture_path).expanduser().resolve()
            if self.texture_path is not None
            else default_texture
        )

        try:
            if texture_path.exists():
                img_data = plt.imread(texture_path)
                img_data = np.flipud(img_data)
                tex = pv.numpy_to_texture(img_data)
            else:
                raise FileNotFoundError(texture_path)
        except Exception:
            tex = examples.planets.download_earth_2k()

        earth_mesh.texture_map_to_sphere(inplace=True, prevent_seam=False)
        earth_mesh.rotate_z(self.texture_alignment_angle_deg, inplace=True)
        earth_actor = pl.add_mesh(earth_mesh, texture=tex, smooth_shading=True, specular=0.2)

        # Orbits
        if self.show_true_orbit:
            pl.add_mesh(pv.lines_from_points(R_true_orig), color="red", line_width=2, label="True")
        if self.show_est_orbit and R_est_static is not None:
            pl.add_mesh(pv.lines_from_points(R_est_static), color="orange", line_width=1, label="Est")

        # Satellite marker
        sat_actor = pl.add_mesh(pv.Sphere(radius=R_e * 0.005), color="cyan")

        # Coordinate goal marker (only if goal is Coordinate_Goal)
        goal_actor = None
        goal_ecef_pos = None
        if Coordinate_Goal is not None and isinstance(self.goal, Coordinate_Goal):
            if hasattr(self.goal, "target_ecef"):
                goal_ecef_pos = np.asarray(self.goal.target_ecef, dtype=float).reshape(3)
                goal_mesh = pv.Sphere(radius=0.1 * R_e, theta_resolution=30)
                goal_actor = pl.add_mesh(goal_mesh, color="cyan", opacity=0.6)

        # Arrows
        base_arrow = pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0), scale=1.0)

        def create_arrow_actor(color, opacity=0.5):
            return pl.add_mesh(base_arrow.copy(), color=color, opacity=opacity)

        actors = {
            "body_x": create_arrow_actor("red", opacity=0.5),
            "body_y": create_arrow_actor("green", opacity=0.5),
            "body_z": create_arrow_actor("blue", opacity=0.5),
            "sun": create_arrow_actor("yellow", opacity=0.5),
            "mag": create_arrow_actor("magenta", opacity=0.5),
            "goal": create_arrow_actor("cyan", opacity=0.5),
        }

        pl.camera_position = "iso"
        pl.show(interactive_update=True)

        scale_base = R_e * self.axis_scale_base_mult

        def update_arrow(actor_key: str, pos: np.ndarray, direction: Optional[np.ndarray], scale_mult: float):
            actor = actors[actor_key]
            if direction is None or np.linalg.norm(direction) < 1e-9:
                actor.SetVisibility(False)
                return
            actor.SetVisibility(True)
            d = direction / np.linalg.norm(direction)
            R_align = _get_rotation_from_vectors(np.array([1.0, 0.0, 0.0]), d)

            S_mat = np.diag([scale_base * scale_mult] * 3 + [1.0])
            R_mat = np.eye(4)
            R_mat[:3, :3] = R_align
            T_mat = np.eye(4)
            T_mat[:3, 3] = pos

            actor.user_matrix = T_mat @ R_mat @ S_mat

        idx = 0
        while not pl.render_window.GetInteractor().GetDone():
            pos = R_true_smooth[idx]
            q_curr = q_smooth[idx]

            # SciPy expects [x, y, z, w]. If your q is [q0,q1,q2,q3] with q0 scalar, roll.
            q_scipy = np.roll(q_curr, -1)

            # Earth transform (ECEF -> ECI)
            R_mat_3x3 = earth_rot_smooth[idx].as_matrix()
            phys_transform = np.eye(4)
            phys_transform[:3, :3] = R_mat_3x3
            earth_actor.user_matrix = phys_transform

            # Coordinate goal marker update (fixed on Earth)
            if goal_actor is not None and goal_ecef_pos is not None:
                T_local = np.eye(4)
                T_local[:3, 3] = goal_ecef_pos
                goal_actor.user_matrix = phys_transform @ T_local

            # Satellite marker transform
            sat_mat = np.eye(4)
            sat_mat[:3, 3] = pos
            sat_actor.user_matrix = sat_mat

            # Body axes
            if self.show_body_axes:
                R_body = R_scipy.from_quat(q_scipy).as_matrix()
                update_arrow("body_x", pos, R_body[:, 0], self.axis_scale_body)
                update_arrow("body_y", pos, R_body[:, 1], self.axis_scale_body)
                update_arrow("body_z", pos, R_body[:, 2], self.axis_scale_body)
            else:
                actors["body_x"].SetVisibility(False)
                actors["body_y"].SetVisibility(False)
                actors["body_z"].SetVisibility(False)

            # Environment
            if self.show_env_vectors:
                update_arrow("sun", pos, S_smooth[idx], self.axis_scale_sun)
                update_arrow("mag", pos, B_smooth[idx], self.axis_scale_mag)
            else:
                actors["sun"].SetVisibility(False)
                actors["mag"].SetVisibility(False)

            # Goal direction arrow (always from sim if available)
            if G_smooth is not None:
                update_arrow("goal", pos, G_smooth[idx], self.axis_scale_goal)
            else:
                actors["goal"].SetVisibility(False)

            pl.update()
            idx = (idx + 1) % target_N

        pl.close()
