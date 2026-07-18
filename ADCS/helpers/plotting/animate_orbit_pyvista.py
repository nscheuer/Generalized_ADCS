__all__ = ["animate_orbit_pyvista"]

import os
import warnings

os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "llvmpipe"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import numpy as np
import pyvista as pv
from pyvista import examples
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_scipy
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from typing import List, Optional

from ADCS.CONOPS.goals import Coordinate_Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants
from ADCS.state import State

TEXTURE_ALIGNMENT_ANGLE = -180
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_TEXTURE_PATH = THIS_DIR / "textures" / "2k_earth_daymap.jpg"


def get_rotation_from_vectors(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < 1e-12 or norm2 < 1e-12:
        return np.eye(3)

    a, b = (vec1 / norm1), (vec2 / norm2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s < 1e-6:
        return np.eye(3) if c > 0 else -np.eye(3)

    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def animate_orbit_pyvista(
    time_hist: np.ndarray,
    state_hist: np.ndarray,
    os_hist: List[Orbital_State],
    est_state_hist: Optional[np.ndarray] = None,
    est_os_hist: Optional[List[Orbital_State]] = None,
    boresight_goal_hist: Optional[np.ndarray] = None,
    coord_goal: Optional[Coordinate_Goal] = None,
    texture_path: Optional[str | Path] = None,
) -> None:
    # ---------------------------
    # Helpers (goal + resampling)
    # ---------------------------
    def _nearest_indices(t_orig: np.ndarray, t_new: np.ndarray) -> np.ndarray:
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

    def _normalize_quat(q: np.ndarray) -> Optional[np.ndarray]:
        q = np.asarray(q, dtype=float).reshape(4)
        if not np.all(np.isfinite(q)):
            return None
        n = np.linalg.norm(q)
        if n < 1e-12:
            return None
        return q / n

    # ---------------------------
    # Timeline smoothing setup
    # ---------------------------
    state_hist = State.stack(state_hist)
    if est_state_hist is not None:
        est_state_hist = State.stack(est_state_hist)
    original_N = len(time_hist)
    target_N = max(original_N * 4, 1000)

    t_orig = np.asarray(time_hist, dtype=float).reshape(-1)
    t_new = np.linspace(t_orig[0], t_orig[-1], target_N)

    def interp_arr(arr):
        f = interp1d(t_orig, arr, axis=0, kind="linear", fill_value="extrapolate")
        return f(t_new)

    # ---------------------------
    # Physics histories (smooth)
    # ---------------------------
    R_true_orig = np.array([os.R for os in os_hist], dtype=float)
    R_true_smooth = interp_arr(R_true_orig)

    q_orig = np.asarray(state_hist[:, 3:7], dtype=float)
    q_smooth = interp_arr(q_orig)
    norms = np.linalg.norm(q_smooth, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    q_smooth = q_smooth / norms

    S_orig = np.array([getattr(os, "S", [0, 0, 0]) for os in os_hist], dtype=float)
    B_orig = np.array([getattr(os, "B", [0, 0, 0]) for os in os_hist], dtype=float)
    S_smooth = interp_arr(S_orig)
    B_smooth = interp_arr(B_orig)

    # ---------------------------
    # Goal history (UPDATED)
    # ---------------------------
    # We produce:
    #   G_kind_smooth: (target_N,) 0=vector goal, 1=quat goal
    #   G_vec_smooth:  (target_N,3) vectors when kind==0 else NaN
    #   G_quat_smooth: (target_N,4) quats when kind==1 else NaN
    G_kind_smooth = None
    G_vec_smooth = None
    G_quat_smooth = None

    if boresight_goal_hist is not None:
        G_arr = np.asarray(boresight_goal_hist, dtype=float)

        # Allow constant [3] or [4]
        if G_arr.ndim == 1 and G_arr.size in (3, 4):
            G_arr = np.repeat(G_arr.reshape(1, -1), len(t_orig), axis=0)

        G_arr = _as_2d(G_arr)

        # Clip to common length
        N_goal = min(len(t_orig), G_arr.shape[0])
        G_arr = G_arr[:N_goal]
        t_goal = t_orig[:N_goal]

        if G_arr.shape[1] == 3:
            # Legacy: safe to interpolate the vector
            G_vec_smooth = interp_arr(G_arr)
            G_quat_smooth = None
            G_kind_smooth = np.zeros(target_N, dtype=int)

        elif G_arr.shape[1] == 4:
            # Mixed: DO NOT interpolate; preserve typing with nearest neighbor selection
            idx_sel = _nearest_indices(t_goal, t_new)
            G_sel = G_arr[idx_sel]

            is_vec = np.isnan(G_sel[:, 0])
            G_kind_smooth = (~is_vec).astype(int)  # 0 vec, 1 quat

            G_vec_smooth = np.full((target_N, 3), np.nan, dtype=float)
            G_quat_smooth = np.full((target_N, 4), np.nan, dtype=float)

            if np.any(is_vec):
                G_vec_smooth[is_vec] = G_sel[is_vec, 1:4]
            if np.any(~is_vec):
                G_quat_smooth[~is_vec] = G_sel[~is_vec, :]

            # Normalize quaternions where present
            m = np.where(G_kind_smooth == 1)[0]
            if m.size > 0:
                for j in m:
                    qn = _normalize_quat(G_quat_smooth[j])
                    if qn is None:
                        G_quat_smooth[j] = np.nan
                    else:
                        G_quat_smooth[j] = qn
        else:
            # Unsupported shape -> ignore goals
            G_kind_smooth = None
            G_vec_smooth = None
            G_quat_smooth = None

    # ---------------------------
    # Estimated orbit (static)
    # ---------------------------
    R_est_static = None
    if est_os_hist is not None:
        R_est_static = np.array([os.R for os in est_os_hist], dtype=float)

    # ---------------------------
    # Earth rotation smoothing
    # ---------------------------
    print("Extracting Earth Rotation Matrices...")
    basis_x = np.array([1.0, 0.0, 0.0])
    basis_y = np.array([0.0, 1.0, 0.0])
    basis_z = np.array([0.0, 0.0, 1.0])

    earth_rot_matrices = []
    for os_item in os_hist:
        col0 = os_item.ecef_to_eci(basis_x)
        col1 = os_item.ecef_to_eci(basis_y)
        col2 = os_item.ecef_to_eci(basis_z)
        earth_rot_matrices.append(np.column_stack((col0, col1, col2)))

    rot_obj = R_scipy.from_matrix(earth_rot_matrices)
    slerp = Slerp(t_orig, rot_obj)
    earth_rot_smooth = slerp(t_new)

    # ---------------------------
    # PyVista scene
    # ---------------------------
    pl = pv.Plotter(window_size=[1200, 900], lighting="three lights")
    pl.set_background("black")

    R_e = EarthConstants.R_e

    earth_mesh = pv.Sphere(radius=R_e, theta_resolution=120, phi_resolution=120)

    if texture_path is None:
        texture_path = DEFAULT_TEXTURE_PATH
    else:
        texture_path = Path(texture_path).expanduser().resolve()

    try:
        if Path(texture_path).exists():
            img_data = plt.imread(texture_path)
            img_data = np.flipud(img_data)
            tex = pv.numpy_to_texture(img_data)
            print(f"Loaded Earth texture: {texture_path}")
        else:
            raise FileNotFoundError(texture_path)
    except Exception as e:
        print(f"Texture load failed ({e}), using fallback Earth texture.")
        tex = examples.planets.download_earth_2k()

    earth_mesh.texture_map_to_sphere(inplace=True, prevent_seam=False)
    earth_mesh.rotate_z(TEXTURE_ALIGNMENT_ANGLE, inplace=True)

    earth_actor = pl.add_mesh(earth_mesh, texture=tex, smooth_shading=True, specular=0.2)

    # Coordinate goal (Earth-fixed marker)
    goal_actor = None
    goal_ecef_pos = None
    if coord_goal is not None:
        goal_r = 0.1 * R_e
        goal_mesh = pv.Sphere(radius=goal_r, theta_resolution=30)
        goal_ecef_pos = np.array(coord_goal.target_ecef, dtype=float).reshape(3)

        T_goal = np.eye(4)
        T_goal[:3, 3] = goal_ecef_pos

        goal_actor = pl.add_mesh(goal_mesh, color="cyan", opacity=0.6)
        goal_actor.user_matrix = T_goal

    # Orbits
    pl.add_mesh(pv.lines_from_points(R_true_orig), color="red", line_width=2, label="True")
    if R_est_static is not None:
        pl.add_mesh(pv.lines_from_points(R_est_static), color="orange", line_width=1, label="Est")

    # Satellite
    sat_actor = pl.add_mesh(pv.Sphere(radius=R_e * 0.005), color="cyan")

    # Arrows
    base_arrow = pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0), scale=1.0)

    def create_arrow_actor(color, opacity=0.5):
        return pl.add_mesh(base_arrow.copy(), color=color, opacity=opacity)

    # NOTE: Added goal triad actors goal_x/goal_y/goal_z
    actors = {
        "body_x": create_arrow_actor("red", opacity=0.5),
        "body_y": create_arrow_actor("green", opacity=0.5),
        "body_z": create_arrow_actor("blue", opacity=0.5),
        "sun": create_arrow_actor("yellow", opacity=0.5),
        "mag": create_arrow_actor("magenta", opacity=0.5),
        "goal": create_arrow_actor("cyan", opacity=0.5),      # vector-goal arrow
        "goal_x": create_arrow_actor("cyan", opacity=0.35),   # quat-goal triad
        "goal_y": create_arrow_actor("cyan", opacity=0.35),
        "goal_z": create_arrow_actor("cyan", opacity=0.35),
    }

    # Start hidden until needed
    actors["goal_x"].SetVisibility(False)
    actors["goal_y"].SetVisibility(False)
    actors["goal_z"].SetVisibility(False)

    pl.camera_position = "iso"
    print("Starting Animation... Press 'q' to close.")
    pl.show(interactive_update=True)

    idx = 0
    body_axes = np.eye(3, dtype=float)
    scale_base = R_e * 0.5

    def update_arrow(actor_key, direction, scale_mult):
        actor = actors[actor_key]
        if direction is None:
            actor.SetVisibility(False)
            return

        d = np.asarray(direction, dtype=float).reshape(3)
        if not np.all(np.isfinite(d)) or np.linalg.norm(d) < 1e-9:
            actor.SetVisibility(False)
            return

        actor.SetVisibility(True)
        d_norm = d / np.linalg.norm(d)

        R_align = get_rotation_from_vectors(np.array([1.0, 0.0, 0.0]), d_norm)

        S_mat = np.diag([scale_base * scale_mult] * 3 + [1.0])
        R_mat = np.eye(4)
        R_mat[:3, :3] = R_align
        T_mat = np.eye(4)
        T_mat[:3, 3] = pos  # uses outer-scope pos each frame

        actor.user_matrix = T_mat @ R_mat @ S_mat

    def hide_goal_all():
        actors["goal"].SetVisibility(False)
        actors["goal_x"].SetVisibility(False)
        actors["goal_y"].SetVisibility(False)
        actors["goal_z"].SetVisibility(False)

    while not pl.render_window.GetInteractor().GetDone():
        try:
            pos = R_true_smooth[idx]
            q_curr = q_smooth[idx]
            q_curr = np.roll(q_curr, -1)  # [q0,q1,q2,q3] -> scipy [x,y,z,w]

            # --- Earth transforms ---
            current_R_sci = earth_rot_smooth[idx]
            R_mat_3x3 = current_R_sci.as_matrix()
            phys_transform = np.eye(4)
            phys_transform[:3, :3] = R_mat_3x3
            earth_actor.user_matrix = phys_transform

            # --- Coordinate goal (Earth-fixed) ---
            if goal_actor is not None and goal_ecef_pos is not None:
                T_local = np.eye(4)
                T_local[:3, 3] = goal_ecef_pos
                goal_actor.user_matrix = phys_transform @ T_local

            # --- Satellite ---
            sat_mat = np.eye(4)
            sat_mat[:3, 3] = pos
            sat_actor.user_matrix = sat_mat

            # --- Body axes ---
            R_body = R_scipy.from_quat(q_curr).as_matrix()
            update_arrow("body_x", R_body[:, 0], 0.3)
            update_arrow("body_y", R_body[:, 1], 0.3)
            update_arrow("body_z", R_body[:, 2], 0.3)

            # --- Env vectors ---
            update_arrow("sun", S_smooth[idx], 0.8)
            update_arrow("mag", B_smooth[idx], 0.6)

            # --- Goal (UPDATED) ---
            if G_kind_smooth is None:
                hide_goal_all()
            else:
                k = int(G_kind_smooth[idx])

                if k == 0:
                    # Vector goal
                    g = None if G_vec_smooth is None else G_vec_smooth[idx]

                    # Heuristic: if g looks like a position (km-scale), convert to direction sat->target
                    if g is not None and np.all(np.isfinite(g)) and np.linalg.norm(g) > 10.0:
                        g = g - pos

                    update_arrow("goal", g, 1.0)
                    actors["goal_x"].SetVisibility(False)
                    actors["goal_y"].SetVisibility(False)
                    actors["goal_z"].SetVisibility(False)

                else:
                    # Quaternion goal -> draw triad
                    actors["goal"].SetVisibility(False)

                    qg = None if G_quat_smooth is None else G_quat_smooth[idx]
                    qg = None if qg is None else _normalize_quat(qg)

                    if qg is None:
                        actors["goal_x"].SetVisibility(False)
                        actors["goal_y"].SetVisibility(False)
                        actors["goal_z"].SetVisibility(False)
                    else:
                        qg_scipy = np.roll(qg, -1)  # [q0,q1,q2,q3] -> [x,y,z,w]
                        R_goal = R_scipy.from_quat(qg_scipy).as_matrix()
                        goal_ax = R_goal @ body_axes  # columns are axes in ECI

                        update_arrow("goal_x", goal_ax[:, 0], 1.0)
                        update_arrow("goal_y", goal_ax[:, 1], 1.0)
                        update_arrow("goal_z", goal_ax[:, 2], 1.0)

            pl.update()
            idx = (idx + 1) % target_N

        except Exception as e:
            print(f"Animation error: {e}")
            break

    pl.close()
