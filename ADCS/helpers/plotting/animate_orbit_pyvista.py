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
import time
from scipy.spatial.transform import Rotation as R_scipy
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from typing import List, Optional

from ADCS.CONOPS.goals import Coordinate_Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants

TEXTURE_ALIGNMENT_ANGLE = -180  
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_TEXTURE_PATH = THIS_DIR / "textures" / "2k_earth_daymap.jpg"

def get_rotation_from_vectors(vec1, vec2):
    r"""
    Compute a rotation matrix that aligns one vector with another.

    This function computes a proper orthogonal rotation matrix
    :math:`\mathbf{R} \in \mathbb{R}^{3\times3}` such that

    .. math::

        \mathbf{R}\,\hat{\mathbf{v}}_1 = \hat{\mathbf{v}}_2

    where :math:`\hat{\mathbf{v}}_1` and :math:`\hat{\mathbf{v}}_2` are the
    normalized versions of the input vectors ``vec1`` and ``vec2``,
    respectively.

    **Mathematical Formulation**

    Let

    .. math::

        \hat{\mathbf{a}} = \frac{\mathbf{v}_1}{\|\mathbf{v}_1\|}, \qquad
        \hat{\mathbf{b}} = \frac{\mathbf{v}_2}{\|\mathbf{v}_2\|}

    Define the rotation axis and angle via

    .. math::

        \mathbf{v} = \hat{\mathbf{a}} \times \hat{\mathbf{b}}, \qquad
        c = \hat{\mathbf{a}} \cdot \hat{\mathbf{b}}, \qquad
        s = \|\mathbf{v}\|

    The skew-symmetric cross-product matrix is

    .. math::

        [\mathbf{v}]_\times =
        \begin{bmatrix}
        0 & -v_z & v_y \\
        v_z & 0 & -v_x \\
        -v_y & v_x & 0
        \end{bmatrix}

    The resulting rotation matrix is given by Rodrigues' rotation formula:

    .. math::

        \mathbf{R} =
        \mathbf{I}
        + [\mathbf{v}]_\times
        + [\mathbf{v}]_\times^2 \frac{1 - c}{s^2}

    **Degenerate Cases**

    * If either input vector has near-zero magnitude, the identity matrix
      is returned.
    * If the vectors are parallel (:math:`s \approx 0, c > 0`), the identity
      matrix is returned.
    * If the vectors are anti-parallel (:math:`s \approx 0, c < 0`), a
      reflection matrix :math:`-\mathbf{I}` is returned.

    :param vec1:
        Initial vector to be rotated.
    :type vec1:
        numpy.ndarray

    :param vec2:
        Target vector after rotation.
    :type vec2:
        numpy.ndarray

    :return:
        Rotation matrix that aligns ``vec1`` with ``vec2``.
    :rtype:
        numpy.ndarray

    """
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
    r"""
    Animate spacecraft orbital motion, attitude, and environment in a 3D Earth-centered scene.

    This function generates a real-time, interactive 3D visualization using
    ``pyvista`` that depicts:

    * Earth with a textured, rotating surface
    * Spacecraft orbit trajectory
    * Spacecraft body-frame axes derived from attitude quaternions
    * Environmental vectors (Sun and magnetic field)
    * Optional coordinate-based pointing goals fixed in the Earth-fixed frame

    All physical quantities are rendered in the Earth-Centered Inertial (ECI)
    frame, while Earth-fixed quantities are transformed from the Earth-Centered
    Earth-Fixed (ECEF) frame using time-varying Earth rotation matrices.

    **Reference Frames**

    The visualization involves three principal frames:

    +----------------+---------------------------------------------+
    | Frame          | Description                                 |
    +================+=============================================+
    | ECI            | Inertial reference frame                    |
    +----------------+---------------------------------------------+
    | ECEF           | Earth-fixed rotating frame                  |
    +----------------+---------------------------------------------+
    | Body           | Spacecraft body-fixed frame                 |
    +----------------+---------------------------------------------+

    **Orbit and Attitude Interpolation**

    To ensure smooth animation, all time histories are interpolated onto a
    refined uniform time grid :math:`t_k`:

    .. math::

        t_k \in [t_0, t_f], \quad k = 1,\dots,N_\text{smooth}

    Spacecraft position:

    .. math::

        \mathbf{r}_\text{ECI}(t) \in \mathbb{R}^3

    Attitude quaternion:

    .. math::

        \mathbf{q}(t) =
        \begin{bmatrix}
        q_0 & q_1 & q_2 & q_3
        \end{bmatrix}^T,
        \qquad \|\mathbf{q}\| = 1

    Quaternions are normalized after interpolation to preserve valid rotations.

    **Earth Rotation**

    The Earth rotation is represented by a time-varying rotation matrix
    :math:`\mathbf{R}_{\text{ECEF}\rightarrow\text{ECI}}(t)` computed from
    the orbital state objects via
    :meth:`~ADCS.orbits.orbital_state.Orbital_State.ecef_to_eci`.

    Smooth Earth rotation is achieved using spherical linear interpolation
    (SLERP):

    .. math::

        \mathbf{R}(t) = \text{SLERP}\left(\mathbf{R}_i, \mathbf{R}_{i+1}, \alpha\right)

    where :math:`\alpha \in [0,1]`.

    **Attitude Visualization**

    Spacecraft body axes are extracted from the quaternion-derived rotation
    matrix:

    .. math::

        \mathbf{R}_{\mathcal{B}\rightarrow\mathcal{I}}(\mathbf{q})

    with each column representing a body axis expressed in ECI:

    .. math::

        \hat{\mathbf{x}}_b, \hat{\mathbf{y}}_b, \hat{\mathbf{z}}_b

    These axes are rendered as scaled arrows originating from the spacecraft
    position.

    **Environmental Vectors**

    Environmental vectors are visualized as inertial unit vectors:

    .. math::

        \hat{\mathbf{S}} = \frac{\mathbf{S}}{\|\mathbf{S}\|}, \qquad
        \hat{\mathbf{B}} = \frac{\mathbf{B}}{\|\mathbf{B}\|}

    representing the Sun direction and magnetic field direction,
    respectively.

    **Coordinate Goal Visualization**

    When a coordinate-based pointing goal is provided via
    :class:`~ADCS.CONOPS.goals.Coordinate_Goal`, the target location is defined
    in the ECEF frame and transformed into ECI using the Earth rotation matrix:

    .. math::

        \mathbf{r}_\text{goal}^\text{ECI}(t) =
        \mathbf{R}_{\text{ECEF}\rightarrow\text{ECI}}(t)
        \mathbf{r}_\text{goal}^\text{ECEF}

    The goal remains fixed to Earth's surface while rotating with the planet.

    **Rendering Notes**

    * Earth texture alignment is corrected by rotating the mesh geometry
      rather than modifying physics transforms.
    * All actor transforms are applied via homogeneous transformation
      matrices.

    :param time_hist:
        Time history corresponding to the provided state and orbital data.
    :type time_hist:
        numpy.ndarray

    :param state_hist:
        True spacecraft state history containing quaternion attitude in
        columns ``[3:7]``.
    :type state_hist:
        numpy.ndarray

    :param os_hist:
        List of orbital state objects providing position, environmental
        vectors, and Earth rotation information.
    :type os_hist:
        list of :class:`~ADCS.orbits.orbital_state.Orbital_State`

    :param est_state_hist:
        Optional estimated spacecraft state history.
    :type est_state_hist:
        numpy.ndarray or None

    :param est_os_hist:
        Optional estimated orbital state history.
    :type est_os_hist:
        list of :class:`~ADCS.orbits.orbital_state.Orbital_State` or None

    :param boresight_goal_hist:
        Optional inertial-frame boresight goal direction history.
    :type boresight_goal_hist:
        numpy.ndarray or None

    :param coord_goal:
        Optional Earth-fixed coordinate goal defining a surface target.
    :type coord_goal:
        :class:`~ADCS.CONOPS.goals.Coordinate_Goal` or None

    :param texture_path:
        Path to an Earth texture image. If not provided, a default or fallback
        texture is used.
    :type texture_path:
        str or pathlib.Path or None

    :return:
        None. The function launches and controls an interactive visualization.
    :rtype:
        None

    """

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

    pl = pv.Plotter(window_size=[1200, 900], lighting="three lights")
    pl.set_background('black')
    
    R_e = EarthConstants.R_e
    
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

    earth_mesh.texture_map_to_sphere(inplace=True, prevent_seam=False)

    earth_mesh.rotate_z(TEXTURE_ALIGNMENT_ANGLE, inplace=True)
    
    earth_actor = pl.add_mesh(earth_mesh, texture=tex, smooth_shading=True, specular=0.2)

    ref_scale = R_e * 1.5

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