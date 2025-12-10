import os
import warnings

# --- 1. DRIVER FIXES ---
os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "llvmpipe"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import numpy as np
import pyvista as pv
from pyvista import examples
from pathlib import Path
import matplotlib.pyplot as plt 
import time
from scipy.spatial.transform import Rotation as R_scipy
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from typing import List, Optional

# --- Custom Imports ---
from ADCS.CONOPS.goals import Coordinate_Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants

# ==========================================
#   CONFIGURATION
# ==========================================
# Adjust this angle to align the texture's Greenwich meridian 
# to the Red ECEF X-axis.
# For standard NASA textures, this is often -180 or -90.
TEXTURE_ALIGNMENT_ANGLE = -180  
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_TEXTURE_PATH = THIS_DIR / "textures" / "2k_earth_daymap.jpg"

def get_rotation_from_vectors(vec1, vec2):
    """ Returns a 3x3 rotation matrix that aligns vec1 to vec2. """
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
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def animate_orbit_pyvista(
    time_hist: np.ndarray,
    state_hist: np.ndarray,
    os_hist: List[Orbital_State],
    est_state_hist: Optional[np.ndarray] = None,
    est_os_hist: Optional[List[Orbital_State]] = None,
    boresight_goal_hist: Optional[np.ndarray] = None,
    coord_goal: Optional[Coordinate_Goal] = None,
    texture_path: Optional[str | Path] = None
) -> None:
    
    print(f"Preprocessing: Interpolating data...")

    # -----------------------------
    # 2. INTERPOLATION
    # -----------------------------
    original_N = len(time_hist)
    target_N = max(original_N * 4, 1000) 
    
    t_orig = time_hist
    t_new = np.linspace(t_orig[0], t_orig[-1], target_N)
    
    def interp_arr(arr):
        f = interp1d(t_orig, arr, axis=0, kind='linear', fill_value="extrapolate")
        return f(t_new)

    # Physics
    R_true_orig = np.array([os.R for os in os_hist])
    R_true_smooth = interp_arr(R_true_orig)
    
    q_orig = state_hist[:, 3:7]
    q_smooth = interp_arr(q_orig)
    norms = np.linalg.norm(q_smooth, axis=1, keepdims=True)
    q_smooth = q_smooth / norms
    
    S_orig = np.array([getattr(os, "S", [0,0,0]) for os in os_hist])
    B_orig = np.array([getattr(os, "B", [0,0,0]) for os in os_hist])
    S_smooth = interp_arr(S_orig)
    B_smooth = interp_arr(B_orig)
    
    G_smooth = None
    if boresight_goal_hist is not None:
        G_smooth = interp_arr(boresight_goal_hist)

    R_est_static = None
    if est_os_hist is not None:
        R_est_static = np.array([os.R for os in est_os_hist])

    # Earth Rotation Matrices (ECEF -> ECI)
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

    # -----------------------------
    # 3. SETUP PLOTTER
    # -----------------------------
    pl = pv.Plotter(window_size=[1200, 900], lighting="three lights")
    pl.set_background('black')
    
    # -----------------------------
    # 4. EARTH ACTOR & TEXTURE FIX
    # -----------------------------
    R_e = EarthConstants.R_e
    
    # 1. Create Mesh
    earth_mesh = pv.Sphere(radius=R_e, theta_resolution=120, phi_resolution=120)
    
    # 2. Load Texture
    if texture_path is None:
        texture_path = DEFAULT_TEXTURE_PATH
    else:
        texture_path = Path(texture_path).expanduser().resolve()

    try:
        if texture_path.exists():
            img_data = plt.imread(texture_path)
            img_data = np.flipud(img_data)
            tex = pv.numpy_to_texture(img_data)
            print(f"Loaded Earth texture: {texture_path}")
        else:
            raise FileNotFoundError(texture_path)
    except Exception as e:
        print(f"Texture load failed ({e}), using fallback Earth texture.")
        tex = examples.planets.download_earth_2k()

    # 3. Map Texture
    earth_mesh.texture_map_to_sphere(inplace=True, prevent_seam=False)
    
    # --- CRITICAL FIX: ROTATE MESH GEOMETRY ---
    # Rotate the mesh points around Z so that Greenwich aligns with ECEF X.
    # We do this 'inplace' on the mesh data, so we don't need matrix math later.
    earth_mesh.rotate_z(TEXTURE_ALIGNMENT_ANGLE, inplace=True)
    
    earth_actor = pl.add_mesh(earth_mesh, texture=tex, smooth_shading=True, specular=0.2)

    # -----------------------------
    # DEBUG: ECEF REFERENCE AXES
    # -----------------------------
    # Red Arrow = ECEF X. This must stay fixed to Physics X.
    ref_scale = R_e * 1.5

    # -----------------------------
    # 5. OTHER ACTORS
    # -----------------------------
    goal_actor = None
    if coord_goal is not None:
        goal_r = 0.1 * R_e
        goal_mesh = pv.Sphere(radius=goal_r, theta_resolution=30)
        goal_ecef_pos = np.array(coord_goal.target_ecef)
        
        # Create a transform for the goal relative to Earth center
        T_goal = np.eye(4)
        T_goal[:3, 3] = goal_ecef_pos
        
        # We store this "Local ECEF" transform to apply physics later
        goal_actor = pl.add_mesh(goal_mesh, color='cyan', opacity=0.6)
        goal_actor.user_matrix = T_goal # Initial Position

    pl.add_mesh(pv.lines_from_points(R_true_orig), color='red', line_width=2, label="True")
    if R_est_static is not None:
        pl.add_mesh(pv.lines_from_points(R_est_static), color='orange', line_width=1, label="Est")

    sat_actor = pl.add_mesh(pv.Sphere(radius=R_e*0.005), color='cyan')
    
    base_arrow = pv.Arrow(start=(0,0,0), direction=(1,0,0), scale=1.0)
    def create_arrow_actor(color, opacity=0.5):
        return pl.add_mesh(base_arrow.copy(), color=color, opacity=opacity)

    actors = {
        'body_x': create_arrow_actor('red', opacity=0.5),
        'body_y': create_arrow_actor('green', opacity=0.5),
        'body_z': create_arrow_actor('blue', opacity=0.5),
        'sun':    create_arrow_actor('yellow', opacity=0.5),
        'mag':    create_arrow_actor('magenta', opacity=0.5),
        'goal':   create_arrow_actor('cyan', opacity=0.5)
    }

    # -----------------------------
    # 6. ANIMATION LOOP
    # -----------------------------
    pl.camera_position = 'iso'
    print("Starting Animation... Press 'q' to close.")
    pl.show(interactive_update=True)
    
    idx = 0
    
    while not pl.render_window.GetInteractor().GetDone():
        try:
            pos = R_true_smooth[idx]
            q_curr = q_smooth[idx]
            q_curr = np.roll(q_curr, -1)
            
            # --- Earth Transforms ---
            # 1. Physics Rotation (ECEF -> ECI)
            current_R_sci = earth_rot_smooth[idx]
            R_mat_3x3 = current_R_sci.as_matrix()
            phys_transform = np.eye(4)
            phys_transform[:3, :3] = R_mat_3x3
            
            # Apply Physics Transform DIRECTLY.
            # Because we rotated the MESH in setup, the texture is already aligned to ECEF X.
            earth_actor.user_matrix = phys_transform

            # --- Coordinate Goal ---
            if goal_actor is not None:
                # Goal Position in ECEF (Fixed relative to Earth)
                T_local = np.eye(4)
                T_local[:3, 3] = goal_ecef_pos
                
                # Apply Physics rotation to the local ECEF position
                goal_actor.user_matrix = phys_transform @ T_local

            # --- Satellite ---
            sat_mat = np.eye(4)
            sat_mat[:3, 3] = pos
            sat_actor.user_matrix = sat_mat

            # --- Vectors ---
            scale_base = R_e * 0.5 
            def update_arrow(actor_key, direction, scale_mult):
                actor = actors[actor_key]
                if direction is None or np.linalg.norm(direction) < 1e-9:
                    actor.SetVisibility(False)
                    return
                actor.SetVisibility(True)
                d_norm = direction / np.linalg.norm(direction)
                R_align = get_rotation_from_vectors(np.array([1,0,0]), d_norm)
                S_mat = np.diag([scale_base * scale_mult] * 3 + [1])
                R_mat = np.eye(4); R_mat[:3, :3] = R_align
                T_mat = np.eye(4); T_mat[:3, 3] = pos
                actor.user_matrix = T_mat @ R_mat @ S_mat

            R_body = R_scipy.from_quat(q_curr).as_matrix()
            update_arrow('body_x', R_body[:, 0], 0.3)
            update_arrow('body_y', R_body[:, 1], 0.3)
            update_arrow('body_z', R_body[:, 2], 0.3)
            update_arrow('sun', S_smooth[idx], 0.8)
            update_arrow('mag', B_smooth[idx], 0.6)
            if G_smooth is not None:
                update_arrow('goal', G_smooth[idx], 1.0)

            pl.update()
            idx = (idx + 1) % target_N
            
        except Exception as e:
            print(f"Animation error: {e}")
            break
            
    pl.close()