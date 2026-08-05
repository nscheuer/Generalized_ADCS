__all__ = ["AnimationPlot"]

import os
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..subplot import Subplot

# Keep PyVista happy in headless / CI
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
    Prefer mixed-goal history in sim.target_hist (Nx4), but fall back to a few common names.
    Legacy vector-only histories (Nx3) are also supported.
    """
    for name in ("target_hist", "target", "eci_target_hist", "eci_target", "boresight_goal_hist"):
        v = getattr(sim, name, None)
        if v is None:
            continue
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0:
            return v, name
    return None, None


def _nearest_indices(t_orig: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """
    For each t_new, return the index of the nearest sample in t_orig.
    Assumes t_orig is sorted ascending.
    """
    t_orig = np.asarray(t_orig, dtype=float).reshape(-1)
    t_new = np.asarray(t_new, dtype=float).reshape(-1)

    idx = np.searchsorted(t_orig, t_new, side="left")
    idx = np.clip(idx, 0, len(t_orig) - 1)
    idx0 = np.clip(idx - 1, 0, len(t_orig) - 1)

    d0 = np.abs(t_new - t_orig[idx0])
    d1 = np.abs(t_orig[idx] - t_new)
    use0 = d0 <= d1
    return np.where(use0, idx0, idx).astype(int)


def _as_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _parse_goal_history(
    G_hist,
    t_orig: np.ndarray,
    t_new: np.ndarray,
    R_true_orig: np.ndarray,
    interp_vec_fn,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Parse and smooth goal history.

    Returns (G_vec_smooth, G_quat_smooth, G_kind), where:
      - G_vec_smooth: (N_new,3) direction/position vectors (only valid when kind == 0)
      - G_quat_smooth: (N_new,4) quaternions (only valid when kind == 1)
      - G_kind: (N_new,) int, 0 for vector-goal, 1 for quaternion-goal

    If the history is legacy Nx3, G_kind is all zeros and G_quat_smooth is None.
    If the history is mixed Nx4, nearest-neighbor sampling is used to preserve goal-type changes.
    """
    if G_hist is None:
        return None, None, None

    G_arr = np.asarray(G_hist, dtype=float)

    # Allow a constant vector/quaternion to be repeated
    if G_arr.ndim == 1 and G_arr.size in (3, 4):
        G_arr = np.repeat(G_arr.reshape(1, -1), len(t_orig), axis=0)

    G_arr = _as_2d(G_arr)

    N = min(len(t_orig), G_arr.shape[0], R_true_orig.shape[0])
    if N <= 0:
        return None, None, None

    G_arr = G_arr[:N]
    t_orig = t_orig[:N]
    R_true_orig = R_true_orig[:N]

    # Legacy vector-only (Nx3)
    if G_arr.shape[1] == 3:
        G_vec = G_arr

        # If magnitudes look like positions (km-scale), convert to direction from sat->target.
        med_norm = float(np.nanmedian(np.linalg.norm(G_vec, axis=1)))
        if med_norm > 10.0:
            G_vec = G_vec - R_true_orig

        G_vec_smooth = interp_vec_fn(G_vec)
        G_kind = np.zeros(len(t_new), dtype=int)
        return G_vec_smooth, None, G_kind

    # Mixed (Nx4): [nan, vx,vy,vz] or [q0,q1,q2,q3]
    if G_arr.shape[1] != 4:
        return None, None, None

    is_vec = np.isnan(G_arr[:, 0])
    G_vec = np.full((N, 3), np.nan, dtype=float)
    G_quat = np.full((N, 4), np.nan, dtype=float)

    if np.any(is_vec):
        G_vec[is_vec] = G_arr[is_vec, 1:4]

        # Heuristic: if vector-goal entries look like positions, convert to direction
        vec_norms = np.linalg.norm(G_vec[is_vec], axis=1)
        med_norm = float(np.nanmedian(vec_norms)) if vec_norms.size else 0.0
        if med_norm > 10.0:
            G_vec[is_vec] = G_vec[is_vec] - R_true_orig[is_vec]

    if np.any(~is_vec):
        G_quat[~is_vec] = G_arr[~is_vec, :]

        # Normalize valid quaternions
        m = np.all(np.isfinite(G_quat), axis=1)
        qn = np.linalg.norm(G_quat[m], axis=1)
        good = qn > 1e-12
        if np.any(good):
            G_quat[m][good] = G_quat[m][good] / qn[good, None]
        # Leave invalid as NaN

    # Preserve piecewise/mixed typing by nearest-neighbor selection on the smooth timeline
    idx_sel = _nearest_indices(t_orig, t_new)
    kind_smooth = (~is_vec[idx_sel]).astype(int)  # 0 vector, 1 quat
    vec_smooth = G_vec[idx_sel]
    quat_smooth = G_quat[idx_sel]

    return vec_smooth, quat_smooth, kind_smooth


class AnimationPlot(Subplot):
    r"""
    Interactive three-dimensional orbit and attitude animation using PyVista.

    This class provides a high-level visualization of spacecraft motion and attitude
    in a dedicated PyVista render window. It is designed to integrate with the
    :class:`~ADCS.subplot.Subplot` interface while rendering outside of Matplotlib.
    The animation shows the Earth, spacecraft orbit, attitude axes, environmental
    vectors, and optional goal indicators.

    Goal visualization supports mixed goal types from the simulation history:

    - **Vector goals** are rendered as a single goal direction arrow.
    - **Quaternion goals** are rendered as a goal attitude triad (axes).

    The goal history is preferred from ``sim.target_hist`` (Nx4) but legacy
    vector-only histories (Nx3) such as ``sim.eci_target_hist`` are supported.

    :param time:
        Name of the simulation attribute containing the time vector in seconds.
    :type time:
        str

    :param title:
        Title displayed in the placeholder subplot and window title.
    :type title:
        str

    :param texture_path:
        Optional path to an Earth texture image file. If None, a default texture
        is used.
    :type texture_path:
        str or pathlib.Path or None

    :param texture_alignment_angle_deg:
        Rotation angle applied to the Earth texture around the Z-axis, in degrees.
    :type texture_alignment_angle_deg:
        float

    :param smooth_factor:
        Factor by which the original simulation timeline is densified for smoother
        animation playback.
    :type smooth_factor:
        int

    :param min_smooth_N:
        Minimum number of animation frames after smoothing.
    :type min_smooth_N:
        int

    :param show_true_orbit:
        If True, the true spacecraft orbit is displayed.
    :type show_true_orbit:
        bool

    :param show_est_orbit:
        If True, the estimated spacecraft orbit is displayed when available.
    :type show_est_orbit:
        bool

    :param show_body_axes:
        If True, the spacecraft body-fixed axes are rendered.
    :type show_body_axes:
        bool

    :param show_env_vectors:
        If True, environmental vectors such as Sun and magnetic field are rendered.
    :type show_env_vectors:
        bool

    :param goal:
        Optional goal object. If it is a coordinate-based goal, a corresponding
        marker is displayed on the Earth.
    :type goal:
        object or None

    :param window_size:
        Size of the PyVista render window in pixels as (width, height).
    :type window_size:
        tuple[int, int]

    :param axis_scale_body:
        Scaling factor for spacecraft body axes.
    :type axis_scale_body:
        float

    :param axis_scale_sun:
        Scaling factor for the Sun direction vector.
    :type axis_scale_sun:
        float

    :param axis_scale_mag:
        Scaling factor for the magnetic field vector.
    :type axis_scale_mag:
        float

    :param axis_scale_goal:
        Scaling factor for the goal direction vector and goal triad axes.
    :type axis_scale_goal:
        float

    :param axis_scale_base_mult:
        Base scaling multiplier applied relative to the Earth radius.
    :type axis_scale_base_mult:
        float

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
        goal=None,
        window_size=(1200, 900),
        axis_scale_body: float = 0.3,
        axis_scale_sun: float = 0.8,
        axis_scale_mag: float = 0.6,
        axis_scale_goal: float = 1.0,
        axis_scale_base_mult: float = 0.5,
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
        import pyvista as pv
        from pyvista import examples
        import matplotlib.pyplot as plt
        from scipy.interpolate import interp1d
        from scipy.spatial.transform import Rotation as R_scipy
        from scipy.spatial.transform import Slerp

        sims = getattr(sim, "runs", None)
        if sims is None:
            sims = [sim]
        sims = [s for s in sims if s is not None]
        if len(sims) == 0:
            raise ValueError("No simulations provided")

        try:
            from ADCS.orbits.universal_constants import EarthConstants
            R_e = float(EarthConstants.R_e)
        except Exception:
            R_e = 6378.137

        try:
            from ADCS.CONOPS.goals import Coordinate_Goal
        except Exception:
            Coordinate_Goal = None

        def _get_time_vec(s):
            t = getattr(s, self.time, None)
            if t is None:
                return None
            t = np.asarray(t, dtype=float).reshape(-1)
            return t if t.size > 0 else None

        t_list = []
        for s in sims:
            t = _get_time_vec(s)
            if t is None:
                raise ValueError(f"sim.{self.time} missing/empty")
            t_list.append(t)

        t0 = max(float(t[0]) for t in t_list)
        t1 = min(float(t[-1]) for t in t_list)
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            t0 = float(t_list[0][0])
            t1 = float(t_list[0][-1])

        orig_Ns = [int(len(t)) for t in t_list]
        target_N = max(int(max(orig_Ns) * self.smooth_factor), self.min_smooth_N)
        t_new = np.linspace(t0, t1, int(target_N))

        per = []
        for s, t_orig_full in zip(sims, t_list):
            os_hist = getattr(s, "os_hist", None)
            if os_hist is None or len(os_hist) == 0:
                raise ValueError("sim.os_hist missing/empty (required for animation)")

            state_hist = getattr(s, "state_hist", None)
            if state_hist is None or len(state_hist) == 0:
                raise ValueError("sim.state_hist missing/empty (required for animation)")

            state_hist = np.asarray(state_hist, dtype=float)
            if state_hist.ndim != 2 or state_hist.shape[1] < 7:
                raise ValueError("sim.state_hist must be (N, >=7) with quaternion in columns [3:7]")

            N = min(int(len(t_orig_full)), int(len(os_hist)), int(state_hist.shape[0]))
            if N <= 0:
                raise ValueError("Simulation has no usable samples")

            t_orig = np.asarray(t_orig_full[:N], dtype=float)

            def _clip_to_overlap(arr):
                arr = arr[:N]
                i0 = int(np.searchsorted(t_orig, t0, side="left"))
                i1 = int(np.searchsorted(t_orig, t1, side="right"))
                i0 = int(np.clip(i0, 0, N - 1))
                i1 = int(np.clip(i1, i0 + 1, N))
                return t_orig[i0:i1], arr[i0:i1]

            R_true_orig_all = np.array(
                [np.asarray(_unwrap_os(o).R, dtype=float).reshape(3) for o in os_hist[:N]],
                dtype=float,
            )
            t_clip, R_true_orig = _clip_to_overlap(R_true_orig_all)

            q_orig_all = np.asarray(state_hist[:N, 3:7], dtype=float)
            _, q_orig = _clip_to_overlap(q_orig_all)

            S_orig_all = np.array(
                [np.asarray(getattr(_unwrap_os(o), "S", [0, 0, 0]), dtype=float).reshape(3) for o in os_hist[:N]],
                dtype=float,
            )
            _, S_orig = _clip_to_overlap(S_orig_all)

            B_orig_all = np.array(
                [np.asarray(getattr(_unwrap_os(o), "B", [0, 0, 0]), dtype=float).reshape(3) for o in os_hist[:N]],
                dtype=float,
            )
            _, B_orig = _clip_to_overlap(B_orig_all)

            if t_clip.size < 2:
                raise ValueError("Not enough overlapping samples for animation")

            def interp_arr(arr: np.ndarray) -> np.ndarray:
                f = interp1d(t_clip, arr, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate")
                return f(t_new)

            R_true_smooth = interp_arr(R_true_orig)

            q_smooth = interp_arr(q_orig)
            qn = np.linalg.norm(q_smooth, axis=1, keepdims=True)
            qn[qn < 1e-12] = 1.0
            q_smooth = q_smooth / qn

            S_smooth = interp_arr(S_orig)
            B_smooth = interp_arr(B_orig)

            est_os_hist = getattr(s, "est_os_hist", None)
            R_est_static = None
            if est_os_hist is not None and len(est_os_hist) > 0:
                rows = []
                for o in est_os_hist:
                    if o is None:
                        continue
                    base = _unwrap_os(o)
                    if hasattr(base, "R"):
                        rows.append(np.asarray(base.R, dtype=float).reshape(3))
                if rows:
                    R_est_static = np.vstack(rows)

            G_hist, _ = _pick_goal_history(s)
            G_vec_smooth, G_quat_smooth, G_kind = _parse_goal_history(
                G_hist=G_hist,
                t_orig=t_clip,
                t_new=t_new,
                R_true_orig=R_true_orig,
                interp_vec_fn=interp_arr,
            )

            per.append(
                dict(
                    sim=s,
                    t_clip=t_clip,
                    R_true_orig=R_true_orig,
                    R_true_smooth=R_true_smooth,
                    q_smooth=q_smooth,
                    S_smooth=S_smooth,
                    B_smooth=B_smooth,
                    R_est_static=R_est_static,
                    G_vec_smooth=G_vec_smooth,
                    G_quat_smooth=G_quat_smooth,
                    G_kind=G_kind,
                )
            )

        os_hist0 = getattr(per[0]["sim"], "os_hist", None)
        if os_hist0 is None or len(os_hist0) == 0:
            raise ValueError("sim.os_hist missing/empty (required for Earth rotation)")

        t_orig0 = _get_time_vec(per[0]["sim"])
        if t_orig0 is None or t_orig0.size == 0:
            raise ValueError(f"sim.{self.time} missing/empty")

        N0 = min(int(len(t_orig0)), int(len(os_hist0)))
        t_orig0 = np.asarray(t_orig0[:N0], dtype=float)
        os_hist0 = os_hist0[:N0]

        basis_x = np.array([1.0, 0.0, 0.0])
        basis_y = np.array([0.0, 1.0, 0.0])
        basis_z = np.array([0.0, 0.0, 1.0])

        earth_rot_mats = []
        for o in os_hist0:
            base = _unwrap_os(o)
            col0 = np.asarray(base.ecef_to_eci(basis_x), dtype=float).reshape(3)
            col1 = np.asarray(base.ecef_to_eci(basis_y), dtype=float).reshape(3)
            col2 = np.asarray(base.ecef_to_eci(basis_z), dtype=float).reshape(3)
            earth_rot_mats.append(np.column_stack((col0, col1, col2)))

        rot_obj = R_scipy.from_matrix(np.asarray(earth_rot_mats))
        earth_slerp = Slerp(t_orig0, rot_obj)
        earth_rot_smooth = earth_slerp(t_new)

        pv.global_theme.multi_samples = 0
        pl = pv.Plotter(window_size=list(self.window_size), lighting="three lights")
        pl.set_background("black")

        earth_mesh = pv.Sphere(radius=R_e, theta_resolution=120, phi_resolution=120)

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

        n_runs = len(per)
        cmap = plt.get_cmap("tab10", max(10, n_runs))
        run_colors = [tuple(cmap(i)[:3]) for i in range(n_runs)]

        if self.show_true_orbit:
            for i, d in enumerate(per):
                pl.add_mesh(
                    pv.lines_from_points(d["R_true_orig"]),
                    color=run_colors[i],
                    line_width=2,
                    label=f"True {i}",
                )

        if self.show_est_orbit:
            for i, d in enumerate(per):
                if d["R_est_static"] is None:
                    continue
                pl.add_mesh(
                    pv.lines_from_points(d["R_est_static"]),
                    color=run_colors[i],
                    line_width=1,
                    label=f"Est {i}",
                )

        sat_actors = []
        for i in range(n_runs):
            sat_actors.append(pl.add_mesh(pv.Sphere(radius=R_e * 0.005), color=run_colors[i]))

        goal_actor = None
        goal_ecef_pos = None
        if Coordinate_Goal is not None and isinstance(self.goal, Coordinate_Goal):
            if hasattr(self.goal, "target_ecef"):
                goal_ecef_pos = np.asarray(self.goal.target_ecef, dtype=float).reshape(3)
                goal_mesh = pv.Sphere(radius=0.1 * R_e, theta_resolution=30)
                goal_actor = pl.add_mesh(goal_mesh, color="cyan", opacity=0.6)

        base_arrow = pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0), scale=1.0)

        def create_arrow_actor(color, opacity=0.5):
            return pl.add_mesh(base_arrow.copy(), color=color, opacity=opacity)

        def make_run_actors(i: int):
            op = max(0.15, 0.55 - 0.03 * i)
            return {
                "body_x": create_arrow_actor("red", opacity=op),
                "body_y": create_arrow_actor("green", opacity=op),
                "body_z": create_arrow_actor("blue", opacity=op),
                "sun": create_arrow_actor("yellow", opacity=op),
                "mag": create_arrow_actor("magenta", opacity=op),
                "goal": create_arrow_actor("cyan", opacity=op),
                "goal_x": create_arrow_actor("cyan", opacity=op * 0.6),
                "goal_y": create_arrow_actor("cyan", opacity=op * 0.6),
                "goal_z": create_arrow_actor("cyan", opacity=op * 0.6),
            }

        run_actors = [make_run_actors(i) for i in range(n_runs)]

        pl.camera_position = "iso"
        pl.show(interactive_update=True)

        scale_base = R_e * self.axis_scale_base_mult

        def update_arrow(actor, pos: np.ndarray, direction: Optional[np.ndarray], scale_mult: float):
            if direction is None:
                actor.SetVisibility(False)
                return
            d = np.asarray(direction, dtype=float).reshape(3)
            if not np.all(np.isfinite(d)) or np.linalg.norm(d) < 1e-9:
                actor.SetVisibility(False)
                return

            actor.SetVisibility(True)
            d = d / np.linalg.norm(d)
            R_align = _get_rotation_from_vectors(np.array([1.0, 0.0, 0.0]), d)

            S_mat = np.diag([scale_base * scale_mult] * 3 + [1.0])
            R_mat = np.eye(4)
            R_mat[:3, :3] = R_align
            T_mat = np.eye(4)
            T_mat[:3, 3] = pos

            actor.user_matrix = T_mat @ R_mat @ S_mat

        def hide_goal(A):
            A["goal"].SetVisibility(False)
            A["goal_x"].SetVisibility(False)
            A["goal_y"].SetVisibility(False)
            A["goal_z"].SetVisibility(False)

        idx = 0
        body_axes = np.eye(3, dtype=float)

        while not pl.render_window.GetInteractor().GetDone():
            R_mat_3x3 = earth_rot_smooth[idx].as_matrix()
            phys_transform = np.eye(4)
            phys_transform[:3, :3] = R_mat_3x3
            earth_actor.user_matrix = phys_transform

            if goal_actor is not None and goal_ecef_pos is not None:
                T_local = np.eye(4)
                T_local[:3, 3] = goal_ecef_pos
                goal_actor.user_matrix = phys_transform @ T_local

            for i, d in enumerate(per):
                pos = d["R_true_smooth"][idx]
                q_curr = d["q_smooth"][idx]
                q_scipy = np.roll(q_curr, -1)

                sat_mat = np.eye(4)
                sat_mat[:3, 3] = pos
                sat_actors[i].user_matrix = sat_mat

                A = run_actors[i]

                if self.show_body_axes:
                    R_body = R_scipy.from_quat(q_scipy).as_matrix()
                    update_arrow(A["body_x"], pos, R_body[:, 0], self.axis_scale_body)
                    update_arrow(A["body_y"], pos, R_body[:, 1], self.axis_scale_body)
                    update_arrow(A["body_z"], pos, R_body[:, 2], self.axis_scale_body)
                else:
                    A["body_x"].SetVisibility(False)
                    A["body_y"].SetVisibility(False)
                    A["body_z"].SetVisibility(False)

                if self.show_env_vectors:
                    update_arrow(A["sun"], pos, d["S_smooth"][idx], self.axis_scale_sun)
                    update_arrow(A["mag"], pos, d["B_smooth"][idx], self.axis_scale_mag)
                else:
                    A["sun"].SetVisibility(False)
                    A["mag"].SetVisibility(False)

                G_kind = d["G_kind"]
                G_vec = d["G_vec_smooth"]
                G_quat = d["G_quat_smooth"]

                if G_kind is None or (G_vec is None and G_quat is None):
                    hide_goal(A)
                else:
                    if int(G_kind[idx]) == 0:
                        update_arrow(A["goal"], pos, (None if G_vec is None else G_vec[idx]), self.axis_scale_goal)
                        A["goal_x"].SetVisibility(False)
                        A["goal_y"].SetVisibility(False)
                        A["goal_z"].SetVisibility(False)
                    else:
                        A["goal"].SetVisibility(False)
                        qg = None if G_quat is None else np.asarray(G_quat[idx], dtype=float).reshape(4)
                        if qg is None or not np.all(np.isfinite(qg)) or np.linalg.norm(qg) < 1e-12:
                            A["goal_x"].SetVisibility(False)
                            A["goal_y"].SetVisibility(False)
                            A["goal_z"].SetVisibility(False)
                        else:
                            qg = qg / np.linalg.norm(qg)
                            qg_scipy = np.roll(qg, -1)
                            R_goal = R_scipy.from_quat(qg_scipy).as_matrix()
                            goal_ax = R_goal @ body_axes
                            update_arrow(A["goal_x"], pos, goal_ax[:, 0], self.axis_scale_goal)
                            update_arrow(A["goal_y"], pos, goal_ax[:, 1], self.axis_scale_goal)
                            update_arrow(A["goal_z"], pos, goal_ax[:, 2], self.axis_scale_goal)

            pl.update()
            idx = (idx + 1) % target_N

        pl.close()
