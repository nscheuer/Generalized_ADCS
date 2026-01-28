from __future__ import annotations

__all__ = ["PlanAndTrackBase"]

import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray

from ADCS.CONOPS.goallist import GoalList
from ADCS.controller import Controller
from ADCS.controller.helpers import PlannerSettings, Trajectory, reorder_controls_cpp_to_python, reorder_gains_cpp_to_python
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.controller.helpers.build_csat import build_cpp_satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.universal_constants import TimeConstants

import trajectory_planner.build.tplaunch as tplaunch
import trajectory_planner.build.pysat as pysat


class PlanAndTrackBase(Controller):
    r"""
    Base class for Plan-and-Track trajectory controllers.

    This class factors out the shared machinery needed by all Plan-and-Track
    controller variants (e.g. time-varying LQR tracking, tracking with
    disturbance estimation, and exact/quaternion formulations). It provides:

    - Construction of a C++ planner stack from a Python satellite model.
    - Propagation of orbital/environment vectors into the exact memory layout
      expected by the C++ planner.
    - A common trajectory-optimization call path and post-processing of the
      returned controls/gains into Python actuator ordering.

    The class is a concrete :class:`~ADCS.controller.Controller` but remains
    abstract in the sense that subclasses must implement the controller’s
    control law and planning policy (e.g. ``find_u()`` and a public
    trajectory-generation entry point) on top of the helpers provided here.

    Mathematical overview
    ---------------------
    The trajectory planner receives a time grid and a collection of
    environment vectors sampled along an orbit segment.

    Let the planning horizon be :math:`T` seconds with a planner time step
    :math:`\Delta t`. The number of knot points is

    .. math::

       N = \left\lceil \frac{T}{\Delta t} \right\rceil + 1.

    The start time is expressed in J2000 centuries, :math:`t_0`, and the end
    time passed to the planner is

    .. math::

       t_1 = t_0 + T \cdot c_{\mathrm{sec}\rightarrow\mathrm{cent}},

    where :math:`c_{\mathrm{sec}\rightarrow\mathrm{cent}}` is the conversion
    factor from seconds to J2000 centuries
    (:class:`~ADCS.orbits.universal_constants.TimeConstants`).

    For each knot point :math:`k \in \{0,\dots,N-1\}`, the propagated orbit
    provides inertial position and velocity :math:`\mathbf{r}_k, \mathbf{v}_k`
    along with environment vectors such as magnetic field and sun direction
    (symbolically :math:`\mathbf{b}_k, \mathbf{s}_k`). The goal system provides
    a reference attitude/pointing direction :math:`\mathbf{e}_k` via
    :meth:`~ADCS.CONOPS.goallist.GoalList.to_ref`.

    Implementation notes
    --------------------
    The C++ side expects specific array shapes and memory layouts:

    - Time: ``(N,)`` contiguous float64.
    - Vectors: ``(3, N)`` float64 Fortran-contiguous (column-major), compatible
      with Armadillo-style matrix views.
    - Scalars: ``(N,)`` contiguous float64.

    Returned controls and gains are converted from the planner’s internal
    actuator ordering (e.g. MTQ then RW) to the Python actuator ordering using
    :func:`~ADCS.controller.helpers.reorder_controls_cpp_to_python` and
    :func:`~ADCS.controller.helpers.reorder_gains_cpp_to_python`.

    """

    est_sat: EstimatedSatellite
    planner_settings: PlannerSettings
    csat: pysat.Satellite
    planner: tplaunch.Planner
    active_trajectory: Optional[Trajectory]
    state_dim: int
    ctrl_dim: int

    def _init_planner(
        self,
        est_sat: EstimatedSatellite,
        planner_settings: PlannerSettings,
        tracking_lqr_formulation: int,
        quat_to_3vec_mode: int = 0
    ) -> None:
        r"""
        Initialize the C++ trajectory planner and bind it to the satellite model.

        This method builds the C++ satellite representation and constructs a C++
        ALTRO planner instance configured with the provided settings. It also sets
        the planner’s quaternion-to-3-vector conversion mode, which determines the
        internal attitude representation used during planning and tracking.

        Mathematical meaning of options
        -------------------------------
        The tracking formulation selects which time-varying LQR (TVLQR) variant is
        used around the optimized nominal trajectory:

        - ``tracking_lqr_formulation = 0`` uses a standard TVLQR tracking law.
        - ``tracking_lqr_formulation = 2`` uses a formulation that estimates a
        (typically constant or slowly varying) disturbance term within the
        tracking controller.

        The attitude parameterization option selects the state representation used
        by the C++ tracking logic:

        - ``quat_to_3vec_mode = 0`` uses a reduced 3-parameter attitude error
        representation (locally minimal) derived from quaternions.
        - ``quat_to_3vec_mode = 2`` uses the full 4-parameter quaternion.

        Referenced components
        ---------------------
        - Satellite conversion:
        :func:`~ADCS.controller.helpers.build_csat.build_cpp_satellite`.
        - Planner settings container:
        :class:`~ADCS.controller.helpers.PlannerSettings`.
        - Estimated satellite model:
        :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`.

        :param est_sat: Estimated satellite model that defines the state and control
            dimensions and contains actuator/sensor configuration.
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        :param planner_settings: Configuration bundle for system and solver settings
            used by the C++ planner.
        :type planner_settings: :class:`~ADCS.controller.helpers.PlannerSettings`
        :param tracking_lqr_formulation: Tracking LQR formulation selector used by
            the planner’s TVLQR cost configuration.
        :type tracking_lqr_formulation: int
        :param quat_to_3vec_mode: Quaternion-to-3-vector conversion mode used by the
            C++ planner/tracker.
        :type quat_to_3vec_mode: int
        :return: None.
        :rtype: None

        """
        self.est_sat = est_sat
        self.planner_settings = planner_settings

        self.csat = build_cpp_satellite(est_sat=est_sat, planner_settings=planner_settings)
        self.planner = tplaunch.Planner(
            self.csat,
            planner_settings.systemSettings(),
            planner_settings.mainAlilqrSettings(),
            planner_settings.secondAlilqrSettings(),
            planner_settings.initTrajSettings(),
            planner_settings.optMainCostSettings(),
            planner_settings.optSecondCostSettings(),
            planner_settings.optTVLQRCostSettings(tracking_LQR_formulation=tracking_lqr_formulation)
        )
        self.planner.setquaternionTo3VecMode(quat_to_3vec_mode)

        self.active_trajectory = None

        self.state_dim = est_sat.state_len
        self.ctrl_dim = est_sat.control_len

    def set_active_trajectory(self, traj: Trajectory) -> None:
        r"""
        Set the active trajectory used for tracking.

        The active trajectory is the nominal plan that the tracking controller
        follows. Subclasses typically reference this trajectory inside their
        control computation (e.g. selecting the nearest knot point and applying
        the corresponding TVLQR feedback gains).

        The trajectory type is the shared planning container
        :class:`~ADCS.controller.helpers.Trajectory`.

        :param traj: The trajectory to mark as active for subsequent tracking.
        :type traj: :class:`~ADCS.controller.helpers.Trajectory`
        :return: None.
        :rtype: None

        """
        self.active_trajectory = traj

    def _propagate_environment(
        self,
        os_0: Orbital_State,
        t_start: float,
        t_end: float,
        dt_seconds: float,
        N: int,
        goals: GoalList
    ) -> Tuple[NDArray[np.float64], ...]:
        r"""
        Propagate orbit and environment vectors into the C++ planner input format.

        This method samples the orbit and associated environment quantities over
        the planning window and returns a tuple of arrays matching the exact shape
        and memory-order requirements of the C++ planner.

        Time base and buffering
        -----------------------
        The planner window is :math:`[t_{\mathrm{start}}, t_{\mathrm{end}}]` in
        J2000 centuries. An additional buffer is appended beyond
        :math:`t_{\mathrm{end}}` to ensure that interpolation and internal requests
        remain valid near the terminal time:

        .. math::

        t_{\mathrm{end,buf}} = t_{\mathrm{end}} + 10 \, \Delta t \,
        c_{\mathrm{sec}\rightarrow\mathrm{cent}}.

        Orbit sampling then produces a sequence of times
        :math:`\{t_k\}_{k=0}^{N-1}` with nominal spacing :math:`\Delta t` seconds.

        Shape and layout contracts
        --------------------------
        The returned arrays satisfy:

        - ``t`` has shape ``(N,)`` and is C-contiguous.
        - ``R, V, B, S, A`` have shape ``(3, N)`` and are Fortran-contiguous.
        - ``E`` has shape ``(3, N)`` for vector goals or ``(4, N)`` for attitude
          (quaternion) goals, and is Fortran-contiguous.
        - ``p`` and ``rho`` have shape ``(N,)`` and are C-contiguous.

        Each vector matrix uses the convention that column :math:`k` corresponds to
        time :math:`t_k`:

        .. math::

        R[:,k] = \mathbf{r}(t_k), \quad
        V[:,k] = \mathbf{v}(t_k), \quad
        B[:,k] = \mathbf{b}(t_k), \quad
        S[:,k] = \mathbf{s}(t_k).

        Goal types and E array format
        -----------------------------
        The planner supports two goal formulations, detected automatically:

        1. **Vector goals** (e.g., Nadir_Goal, Sun_Goal): ``E`` is ``(3, N)``.
           The cost function penalizes misalignment between the body reference
           vector ``A[:,k]`` and the ECI goal vector ``E[:,k]``.

        2. **Attitude goals** (e.g., Fixed_Attitude_Goal): ``E`` is ``(4, N)``.
           The cost function penalizes quaternion error between the current
           attitude and the goal quaternion ``E[:,k]``.

        The C++ planner detects the mode based on ``E.n_rows`` and uses the
        appropriate cost function (``stepcost_vec`` vs ``stepcost_quat``).

        The attitude and body reference vectors are:

        - ``A[:,k]`` is the satellite body-frame reference vector (e.g. boresight)
          duplicated across the horizon. Used primarily for vector goals.
        - ``E[:,k]`` is the inertial reference (3D vector or 4D quaternion) returned
          by the goal system for time :math:`t_k`, computed via
          :meth:`~ADCS.CONOPS.goallist.GoalList.to_ref`.

        Clipping and padding
        --------------------
        If the sampled orbit yields :math:`N_{\mathrm{cur}} \neq N` points, vectors
        are clipped or padded by edge extension to enforce exactly ``N`` points,
        preserving endpoint values when padding.

        Referenced components
        ---------------------
        - Orbit propagation:
        :class:`~ADCS.orbits.orbit.Orbit`.
        - Orbital state container:
        :class:`~ADCS.orbits.orbital_state.Orbital_State`.
        - Goal reference generation:
        :class:`~ADCS.CONOPS.goallist.GoalList`.
        - Seconds-to-centuries conversion:
        :class:`~ADCS.orbits.universal_constants.TimeConstants`.

        :param os_0: Initial orbital state used to seed the orbit propagator.
        :type os_0: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :param t_start: Start time of the propagation window in J2000 centuries.
        :type t_start: float
        :param t_end: End time of the propagation window in J2000 centuries.
        :type t_end: float
        :param dt_seconds: Sampling time step in seconds.
        :type dt_seconds: float
        :param N: Number of time points required by the planner input contract.
        :type N: int
        :param goals: Goal list used to compute inertial reference vectors.
        :type goals: :class:`~ADCS.CONOPS.goallist.GoalList`
        :return: A tuple ``(t, R, V, B, S, A, E, p, rho)`` where ``t, p, rho`` are
            length-``N`` vectors and the others are ``(3, N)`` matrices with the
            memory layouts expected by the C++ planner.
        :rtype: tuple[numpy.typing.NDArray[numpy.float64], ...]

        """
        buffer_centuries = 10 * dt_seconds * TimeConstants.sec2cent
        t_end_buffered = t_end + buffer_centuries

        # Use fast orbit propagation for ~100x speedup in planner applications
        # The fast mode uses simplified J2 propagation suitable for short horizons
        sim_orbit = Orbit(os0=os_0, end_time=t_end_buffered, dt=dt_seconds, use_J2=True, fast=True)
        
        # Populate magnetic field (B) and sun vector (S) using batch computation
        # This is critical for MTQ-based control which depends on the B-field direction
        # and for sun-avoidance constraints
        sim_orbit.populate_environment(compute_B=True, compute_S=True)
        
        tp_orbit = sim_orbit.get_range(t_start, t_end, dt_seconds)

        orbit_data_lists = tp_orbit.get_vecs()
        times_arr = np.asarray(tp_orbit.times, dtype=np.float64)

        # -------------------------
        # Clip/pad times to N
        # -------------------------
        curN = times_arr.shape[0]
        if curN > N:
            times_arr = times_arr[:N]
        elif curN < N:
            times_arr = np.pad(times_arr, (0, N - curN), mode="edge")

        # Helper: force (3,N) float64 Fortran-contiguous
        def to_mat3xN(name: str, x) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            x = np.squeeze(x)

            if x.ndim != 2:
                raise ValueError(f"{name} must be 2D, got ndim={x.ndim}, shape={x.shape}")

            # Accept either (3,N) or (N,3)
            if x.shape == (3, N):
                y = x
            elif x.shape == (N, 3):
                y = x.T
            else:
                raise ValueError(f"{name} has unexpected shape {x.shape}; expected (3,{N}) or ({N},3)")

            # Armadillo-friendly memory layout
            return np.asfortranarray(y, dtype=np.float64)

        # Helper: force (N,) float64 contiguous
        def to_vecN(name: str, x) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            if x.shape[0] == N:
                return np.ascontiguousarray(x, dtype=np.float64)
            if x.shape[0] > N:
                return np.ascontiguousarray(x[:N], dtype=np.float64)
            return np.ascontiguousarray(np.pad(x, (0, N - x.shape[0]), mode="edge"), dtype=np.float64)

        # -------------------------
        # Orbit vectors: expect [R,V,B,S,Rho] from get_vecs()
        # -------------------------
        # Convert lists->arrays before shape logic
        R_raw, V_raw, B_raw, S_raw, Rho_raw = [np.asarray(d) for d in orbit_data_lists]

        # Clip/pad each (handles both (3,curN) and (curN,3))
        def clip_pad_mat(x):
            x = np.asarray(x, dtype=np.float64)
            if x.ndim != 2:
                return x
            if x.shape[0] == 3:
                # (3,curN)
                if x.shape[1] > N:  return x[:, :N]
                if x.shape[1] < N:  return np.pad(x, ((0,0),(0, N-x.shape[1])), mode="edge")
                return x
            if x.shape[1] == 3:
                # (curN,3)
                if x.shape[0] > N:  return x[:N, :]
                if x.shape[0] < N:  return np.pad(x, ((0, N-x.shape[0]), (0,0)), mode="edge")
                return x
            return x

        R = to_mat3xN("R", clip_pad_mat(R_raw))
        V = to_mat3xN("V", clip_pad_mat(V_raw))
        B = to_mat3xN("B", clip_pad_mat(B_raw))
        S = to_mat3xN("S", clip_pad_mat(S_raw))
        rho = to_vecN("Rho", Rho_raw)

        # -------------------------
        # Goals / attitude vectors
        # -------------------------
        # The C++ planner supports two goal formulations:
        #
        # 1. Vector goals (3D): The goal is a direction vector in ECI frame that
        #    the satellite's body vector should align with. E is (3, N).
        #    Used by: Nadir_Goal, Sun_Goal, Velocity_Goal, etc.
        #
        # 2. Attitude goals (4D quaternion): The goal is a full attitude quaternion
        #    specifying the desired orientation. E is (4, N).
        #    Used by: Fixed_Attitude_Goal and other full-attitude goals.
        #
        # The C++ code (Satellite::costJacobians) detects which mode based on E.n_rows:
        #   - If E has 3 rows: uses vector alignment cost (stepcost_vec)
        #   - If E has 4 rows: uses quaternion error cost (stepcost_quat)
        #
        # We detect the goal type by checking the first goal's output dimension.
        
        # Sample first goal to determine dimensionality
        t0 = float(times_arr[0])
        os_at_t0 = sim_orbit.get_os(t0)
        g_sample, _ = goals.to_ref(t0, os_at_t0)
        g_sample = np.asarray(g_sample, dtype=np.float64).flatten()
        goal_dim = g_sample.shape[0]
        
        if goal_dim not in (3, 4):
            raise ValueError(
                f"Goal output must be 3D (vector) or 4D (quaternion), got {goal_dim}D. "
                f"Check that your goal class returns the correct format from to_ref()."
            )
        
        is_quaternion_goal = (goal_dim == 4)
        
        # Allocate goal array with correct dimensions
        # E: (goal_dim, N) - either (3, N) for vector goals or (4, N) for quaternion goals
        goal_vecs_eci = np.zeros((goal_dim, N), dtype=np.float64, order="F")
        
        # A: Body-frame reference vector (always 3D)
        # For vector goals: the body axis to align with ECI goal vector
        # For quaternion goals: still used but less critical (body boresight)
        sat_body_vecs = np.zeros((3, N), dtype=np.float64, order="F")
        
        # Propulsion values (for prop disturbance modeling)
        prop_vals = np.zeros(N, dtype=np.float64)

        # Populate goal arrays for each timestep
        for i in range(N):
            t = float(times_arr[i])
            os_at_t = sim_orbit.get_os(t)
            g_vec_eci, _w_ref = goals.to_ref(t, os_at_t)
            g_vec_eci = np.asarray(g_vec_eci, dtype=np.float64).flatten()
            
            # Validate consistent goal dimensionality across timesteps
            if g_vec_eci.shape[0] != goal_dim:
                raise ValueError(
                    f"Inconsistent goal dimensions: expected {goal_dim}D at all times, "
                    f"but got {g_vec_eci.shape[0]}D at t={t}. "
                    f"Goal types may not be mixed within a single trajectory."
                )
            
            goal_vecs_eci[:, i] = g_vec_eci
            sat_body_vecs[:, i] = np.asarray(self.est_sat.boresight, dtype=np.float64).reshape(3)

        # Convert to Fortran-contiguous for Armadillo (column-major)
        A = np.asfortranarray(sat_body_vecs, dtype=np.float64)  # Body reference vec (3, N)
        E = np.asfortranarray(goal_vecs_eci, dtype=np.float64)  # ECI goal (3, N) or (4, N)
        p = np.ascontiguousarray(prop_vals.reshape(-1), dtype=np.float64)

        t_c = np.ascontiguousarray(times_arr.reshape(-1), dtype=np.float64)

        return (t_c, R, V, B, S, A, E, p, rho)

    def _calculate_trajectory_common(
        self,
        t_start: float,
        duration: float,
        x_0: np.ndarray,
        os_0: Orbital_State,
        goals: GoalList,
        verbose: bool = False
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Compute a nominal trajectory and TVLQR tracking data using the C++ planner.

        This method implements the shared planning workflow:

        1. Construct the planner time grid based on :math:`\Delta t` from
        :class:`~ADCS.controller.helpers.PlannerSettings`.
        2. Propagate orbit/environment vectors into C++-compatible arrays using
        :meth:`~PlanAndTrackBase._propagate_environment`.
        3. Call the C++ planner optimization routine to obtain a nominal state and
        control trajectory plus tracking gains.
        4. Reorder controls and gains from C++ actuator ordering to Python actuator
        ordering.

        Discretization
        --------------
        The horizon is ``duration`` seconds with planner step
        :math:`\Delta t = \texttt{planner\_settings.dt\_tvlqr}`. The knot count is:

        .. math::

        N = \left\lceil \frac{\texttt{duration}}{\Delta t} \right\rceil + 1.

        The end time passed to planning (in J2000 centuries) is:

        .. math::

        t_{\mathrm{end}} = t_{\mathrm{start}} +
        \texttt{duration} \cdot c_{\mathrm{sec}\rightarrow\mathrm{cent}}.

        Planner inputs and outputs
        --------------------------
        The C++ routine returns an optimized nominal trajectory and TVLQR data.
        Denote the nominal discrete-time trajectory:

        .. math::

        \{\mathbf{x}_k\}_{k=0}^{N-1}, \quad
        \{\mathbf{u}_k\}_{k=0}^{N-2}.

        The tracking law typically uses time-varying feedback gains:

        .. math::

        \mathbf{u}(t_k) = \mathbf{u}_k - K_k \left(\mathbf{x}(t_k) - \mathbf{x}_k\right),

        where the exact form depends on the chosen tracking formulation configured
        during :meth:`~PlanAndTrackBase._init_planner`.

        Control and gain reordering
        ---------------------------
        The planner returns controls and gains in an internal actuator ordering
        (commonly magnetorquers before reaction wheels). To match the Python
        actuator ordering defined by the estimated satellite model, the outputs are
        transformed using:

        - :func:`~ADCS.controller.helpers.reorder_controls_cpp_to_python`
        - :func:`~ADCS.controller.helpers.reorder_gains_cpp_to_python`

        Referenced components
        ---------------------
        - Planner settings:
        :class:`~ADCS.controller.helpers.PlannerSettings`.
        - Estimated satellite and actuator ordering:
        :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`.
        - Goal system:
        :class:`~ADCS.CONOPS.goallist.GoalList`.
        - Seconds-to-centuries conversion:
        :class:`~ADCS.orbits.universal_constants.TimeConstants`.

        :param t_start: Planning start time in J2000 centuries.
        :type t_start: float
        :param duration: Planning horizon length in seconds.
        :type duration: float
        :param x_0: Initial state vector used to seed the optimizer.
        :type x_0: numpy.ndarray
        :param os_0: Initial orbital state used to seed environment propagation.
        :type os_0: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :param goals: Goal list used to compute inertial reference vectors along the
            horizon.
        :type goals: :class:`~ADCS.CONOPS.goallist.GoalList`
        :param verbose: If true, enable planner verbosity and print vector shape
            diagnostics.
        :type verbose: bool
        :return: ``(lqr_times, Xset, Uset, Kset, Sset)`` where ``lqr_times`` is the
            time grid associated with the TVLQR data, ``Xset`` is the nominal state
            sequence, ``Uset`` is the reordered nominal control sequence, ``Kset``
            is the reordered TVLQR gain sequence, and ``Sset`` is the planner’s
            auxiliary TVLQR data sequence.
        :rtype: tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]

        """
        if verbose:
            print(f"Planning traj: Start={t_start:.5f}, Dur={duration}s")

        self.planner.setVerbosity(verbose)

        dt_seconds = self.planner_settings.dt_tvlqr

        # Calculate N
        N = int(np.ceil(duration / dt_seconds)) + 1

        t_end = t_start + (duration * TimeConstants.sec2cent)

        # Propagate Environment
        vecsPy = self._propagate_environment(os_0, t_start, t_end, dt_seconds, N, goals)

        # SANITIZE x_0: Force Float64, Copy, and C-Order
        x_0_clean = np.copy(x_0.astype(np.float64).flatten(), order='C')

        bdotOn = self.planner_settings.bdot_on
        if verbose:
            print("=== PYTHON VECTOR_INFO_FORM DEBUG (before trajOpt) ===")
            labels = [
                "t", "R", "V", "B", "S",
                "A (satvec)", "E (ECIvec)", "p (prop)", "rho",
            ]
            for i, (lbl, x) in enumerate(zip(labels, vecsPy)):
                if isinstance(x, np.ndarray):
                    print(
                        f"{i}: {lbl:<12} "
                        f"ndim={x.ndim} "
                        f"shape={x.shape} "
                        f"dtype={x.dtype} "
                        f"C={x.flags['C_CONTIGUOUS']} "
                        f"F={x.flags['F_CONTIGUOUS']}"
                    )
                else:
                    print(f"{i}: {lbl:<12} type={type(x)}")
            print("==============================================")

        (_, _, _, lqr_opt, _) = self.planner.trajOpt(vecsPy, N, t_start, t_end, x_0_clean, int(bdotOn))
        (Xset, Uset_cpp, Tset, Kset_cpp, Sset, lqr_times) = lqr_opt

        # Reorder controls and gains from C++ ordering (MTQ, RW) to Python actuator ordering
        Uset = reorder_controls_cpp_to_python(Uset_cpp, self.est_sat.actuators)
        Kset = reorder_gains_cpp_to_python(Kset_cpp, self.est_sat.actuators)

        return (np.array(lqr_times), Xset, Uset, Kset, Sset)
