__all__ = ["MTQ_w_1RW"]

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import itertools

from ADCS.CONOPS.goals import Goal, No_Goal
from ADCS.controller import Controller
from ADCS.controller.helpers.quaternion_math import vector_alignment_error
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.helpers.math_helpers import rot_mat, skewsym, limit


class MTQ_w_1RW(Controller):
    r"""
    Hybrid MTQ + single-RW controller with two operating modes:

    1) **Pointing mode** (any goal other than :class:`~ADCS.CONOPS.goals.No_Goal`):
       Generates a desired body torque for attitude regulation and allocates it across
       MTQs + RW using the projection-split allocator implemented in
       :meth:`allocate_max_torque_in_direction`.

    2) **No-goal mode** (:class:`~ADCS.CONOPS.goals.No_Goal`):
       Runs a "smart B-dot + momentum management" law:
       - MTQs are used to damp body rates in the plane orthogonal to the geomagnetic field,
       - the RW is biased (slowly) toward a target stored momentum :math:`h_\mathrm{tgt}`.

    Important physics constraints:

    - Magnetic torquers can only generate torque orthogonal to :math:`\mathbf{B}`:

      .. math::

         \boldsymbol{\tau}_{mtq} = \mathbf{m} \times \mathbf{B},
         \qquad
         \mathbf{B}^\top \boldsymbol{\tau}_{mtq} = 0.

    - With a **single** RW with axis :math:`\hat{\mathbf a}`,
      the wheel stores scalar angular momentum :math:`h` about that axis:

      .. math::

         \mathbf{h}_{rw} = h \, \hat{\mathbf a}.

    No-goal control law (conceptual):

    - **Rate damping (B-dot style, expressed using rate projection):**

      Let

      .. math::

         \boldsymbol{\omega}_\perp
         =
         \boldsymbol{\omega} - (\boldsymbol{\omega}^\top \hat{\mathbf b}) \hat{\mathbf b},
         \qquad \hat{\mathbf b} = \frac{\mathbf{B}}{\|\mathbf{B}\|}.

      Then

      .. math::

         \boldsymbol{\tau}_{bdot} = -k_\omega \, \boldsymbol{\omega}_\perp.

      This is equivalent (up to a scalar factor) to the classical B-dot law
      :math:`\mathbf{m} \propto -\dot{\mathbf{B}}`, under the approximation
      :math:`\dot{\mathbf{B}} \approx \boldsymbol{\omega} \times \mathbf{B}`.

    - **Wheel momentum biasing:**

      .. math::

         \boldsymbol{\tau}_{dump} = k_h (h - h_\mathrm{tgt}) \hat{\mathbf a},

      with an *optional* saturation/limiting of :math:`\|\boldsymbol{\tau}_{dump}\|`
      to enforce slow desaturation (to avoid exciting attitude dynamics).

    Because MTQs cannot create the component parallel to :math:`\mathbf{B}`,
    the implementation projects / gates the commanded torques into the plane
    orthogonal to :math:`\mathbf{B}` and scales both RW and MTQ commands
    consistently when the MTQs saturate.

    Parameters
    ----------
    est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        Satellite model containing sensors, actuators, inertia, and boresight.
    p_gain : float
        Proportional gain for pointing-mode attitude error.
    d_gain : float
        Derivative gain for pointing-mode rate error; reused as :math:`k_\omega` in No-Goal mode.
    c_gain : float
        Momentum management gain; reused as :math:`k_h` in No-Goal mode.
    h_target : float
        Target scalar stored momentum (single RW case), in :math:`\mathrm{N\,m\,s}`.

    Raises
    ------
    ValueError
        If the satellite does not contain exactly one RW, or contains fewer than
        three MTQs, or if the MTQ axes are not full-rank (rank < 3).

    Notes
    -----
    This class assumes the estimated state vector contains:

    - :math:`\boldsymbol{\omega}` as ``x_hat[0:3]`` [rad/s],
    - quaternion ``q`` as ``x_hat[3:7]`` (scalar-first),
    - the single RW momentum scalar ``h`` as ``x_hat[7]`` [Nms].

    See Also
    --------
    :class:`~ADCS.CONOPS.goals.No_Goal`
    :meth:`find_u`
    :meth:`allocate_max_torque_in_direction`
    """

    def __init__(
        self,
        est_sat: EstimatedSatellite,
        p_gain: float,
        d_gain: float,
        c_gain: float,
        h_target: float
    ) -> None:
        r"""
        Initialize controller gains and precompute sensor/actuator maps.

        In addition to standard parameter assignment, this initializer performs
        **configuration validity checks** that are critical for correctness:

        1) **Exactly one reaction wheel:**
           The no-goal momentum state is treated as a scalar :math:`h` and mapped
           along a single axis :math:`\hat{\mathbf a}`. Multiple wheels require a
           different state mapping and momentum objective.

        2) **At least three MTQs with full-rank axis matrix:**
           Define the MTQ axis matrix

           .. math::

              A_{mtq} = \begin{bmatrix}
                \hat{\mathbf u}_1 & \hat{\mathbf u}_2 & \cdots & \hat{\mathbf u}_{N}
              \end{bmatrix} \in \mathbb{R}^{3\times N}.

           We require

           .. math::

              \mathrm{rank}(A_{mtq}) = 3,

           so that the dipole vector spans 3D space. While the achievable torque is
           always constrained to :math:`\mathbf{B}^\perp`, a rank-3 dipole basis is
           the minimum practical condition for robust torque generation across orbit
           (except near degenerate field alignments).

        Parameters
        ----------
        est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
            Satellite model.
        p_gain, d_gain, c_gain : float
            Controller gains.
        h_target : float
            Target wheel momentum.

        Raises
        ------
        ValueError
            If MTQ/RW configuration checks fail.
        """
        # --- Configuration checks (required by this controller) ---
        mtqs = [a for a in est_sat.actuators if isinstance(a, MTQ)]
        rws  = [a for a in est_sat.actuators if isinstance(a, RW)]

        if len(rws) != 1:
            raise ValueError(
                f"MTQ_w_1RW requires exactly 1 reaction wheel; found {len(rws)}."
            )

        if len(mtqs) < 3:
            raise ValueError(
                f"MTQ_w_1RW requires at least 3 MTQs; found {len(mtqs)}."
            )

        A_axes = np.column_stack([np.asarray(m.axis, float).reshape(3,) for m in mtqs])
        if np.linalg.matrix_rank(A_axes) < 3:
            raise ValueError(
                "MTQ_w_1RW requires MTQ axes to be full-rank (rank=3). "
                f"Got rank={np.linalg.matrix_rank(A_axes)}."
            )

        self.p_gain = float(p_gain)
        self.d_gain = float(d_gain)
        self.c_gain = float(c_gain)
        self.h_target = float(h_target)

        # MTM reconstruction
        self.M_mtm_read, _ = self.build_sensor_matrix_pinv(
            sensors=est_sat.sensors + est_sat.rw_actuators, sensor_type=MTM
        )

        # Torque → command maps
        self.M_mtq_act, self.mtq_indices = self.build_torque_to_u_matrix_pinv(
            actuators=est_sat.actuators, actuator_type=MTQ
        )
        self.M_rw_act, self.rw_indices = self.build_torque_to_u_matrix_pinv(
            actuators=est_sat.actuators, actuator_type=RW
        )

        # Command → torque map for MTQs (full vector length)
        self.A_mtq = self.build_u_to_torque_matrix_pinv(
            actuators=est_sat.actuators, actuator_type=MTQ
        )

        # RW geometry
        self.rw_axes = np.vstack([
            np.asarray(rw.axis, float).reshape(3,)
            for rw in est_sat.actuators if isinstance(rw, RW)
        ])

        self.n_actuators = len(est_sat.actuators)
        self.mtq_umax = np.array([a.u_max for a in est_sat.actuators if isinstance(a, MTQ)], dtype=float)
        self.rw_umax  = np.array([a.u_max for a in est_sat.actuators if isinstance(a, RW)], dtype=float)

    def find_u(
        self,
        x_hat: np.ndarray,
        sens: np.ndarray,
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        goal: Goal | None = None,
    ) -> np.ndarray:
        r"""
        Compute actuator commands for MTQs and the single RW.

        Two-mode behavior:

        **A) No-goal mode** (:class:`~ADCS.CONOPS.goals.No_Goal`)
            Produces a damping torque in the plane orthogonal to :math:`\mathbf{B}`
            and a (slow) wheel momentum bias torque along the RW axis.

            Rate damping:

            .. math::

               \boldsymbol{\omega}_\perp
               =
               \boldsymbol{\omega} - (\boldsymbol{\omega}^\top \hat{\mathbf b}) \hat{\mathbf b},
               \qquad
               \boldsymbol{\tau}_{bdot} = -k_\omega \boldsymbol{\omega}_\perp.

            Wheel momentum biasing:

            .. math::

               \boldsymbol{\tau}_{dump} = k_h (h - h_\mathrm{tgt}) \hat{\mathbf a}.

            Optional limiting of :math:`\boldsymbol{\tau}_{dump}` (recommended in practice):

            .. math::

               \boldsymbol{\tau}_{dump} \leftarrow
               \mathrm{sat}_{\tau_{max}}\!\left(\boldsymbol{\tau}_{dump}\right),

            to prevent aggressive desaturation that can inject attitude energy.

            Since MTQs may saturate, a consistent scaling factor :math:`\alpha \in (0,1]`
            is applied such that both MTQ and RW requests remain matched in the intended
            torque split when MTQ commands exceed limits.

        **B) Pointing mode** (all other goals)
            Forms a pointing torque based on quaternion error and angular-rate error,
            adds gyroscopic compensation, and calls :meth:`allocate_max_torque_in_direction`
            to compute feasible RW+MTQ commands.

        Parameters
        ----------
        x_hat : np.ndarray
            Estimated state vector. Must include :math:`\boldsymbol{\omega}`, quaternion, and
            a single wheel momentum scalar :math:`h`.
        sens : np.ndarray
            Sensor vector. MTM channels are used to reconstruct :math:`\mathbf{B}` in body frame.
        est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
            Satellite model.
        os_hat : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Orbital state (used by goals, field models, etc.).
        goal : :class:`~ADCS.CONOPS.goals.Goal` or None
            Control objective. If None, defaults to :class:`~ADCS.CONOPS.goals.No_Goal`.

        Returns
        -------
        np.ndarray
            Full actuator command vector ordered by ``est_sat.actuators``.

        Notes
        -----
        - MTQ torque is always orthogonal to :math:`\mathbf{B}`.
        - The no-goal mode does **not** attempt to control pointing; it prioritizes
          detumbling and slow wheel momentum biasing.
        """

        if goal is None:
            goal = No_Goal()

        w = x_hat[0:3]
        q = x_hat[3:7]

        if isinstance(goal, No_Goal):
            # Smart B-Dot + Momentum Dumping (CONSISTENT SATURATION)

            k_w = self.d_gain
            k_h = self.c_gain

            # --- Magnetic field ---
            b_body = np.asarray(self.M_mtm_read @ sens, float).reshape(3,)
            b_norm = np.linalg.norm(b_body)
            if b_norm < 1e-9:
                return np.zeros(self.n_actuators)
            b_hat = b_body / b_norm

            # --- Wheel momentum (1 RW) ---
            h_val = x_hat[7]
            a_hat = self.rw_axes[0]

            # --- B-dot style rate damping (⊥ B only) ---
            w_perp = w - np.dot(w, b_hat) * b_hat
            tau_bdot = -k_w * w_perp
            tau_bdot_perp = tau_bdot  # already ⟂ B

            # --- Wheel momentum dumping torque (BODY torque) ---
            delta_h = h_val - self.h_target
            tau_rw_dump_des = k_h * delta_h * a_hat
            mag = np.linalg.norm(tau_rw_dump_des)
            if mag > k_h*0.001:
                tau_rw_dump_des *= k_h*0.001 / (mag + 1e-12)

            # Project dump torque into MTQ-achievable plane
            tau_rw_dump_perp = tau_rw_dump_des - np.dot(tau_rw_dump_des, b_hat) * b_hat

            # Geometry gating (avoid uncancellable dumping)
            denom = np.linalg.norm(tau_rw_dump_des) + 1e-12
            gamma = np.linalg.norm(tau_rw_dump_perp) / denom
            tau_rw_dump_cmd = gamma * tau_rw_dump_des

            # --- Desired MTQ torque ---
            tau_mtq_des = tau_bdot_perp - tau_rw_dump_perp

            # --- MTQ allocation (UNCLIPPED) ---
            B_skew = skewsym(b_body)
            M_mag_eff = -B_skew @ self.A_mtq     # (3, n_actuators)

            u_mtq_raw = np.linalg.pinv(M_mag_eff) @ tau_mtq_des

            # --- Compute MTQ saturation scaling factor ---
            mtq_cmds = u_mtq_raw[self.mtq_indices]
            alpha_mtq = 1.0
            if np.any(np.abs(mtq_cmds) > self.mtq_umax):
                alpha_mtq = np.min(self.mtq_umax / (np.abs(mtq_cmds) + 1e-12))

            # --- Apply SAME scaling to MTQs and RW torques ---
            u_mtq_scaled = alpha_mtq * u_mtq_raw
            tau_mag_actual = M_mag_eff @ u_mtq_scaled

            tau_rw_req = alpha_mtq * (tau_rw_dump_cmd + tau_bdot_perp)

            # --- RW allocation ---
            u_rw = self.M_rw_act @ tau_rw_req
            u_rw = np.clip(u_rw, -self.rw_umax, self.rw_umax)

            # --- Final actuator command ---
            u_out = u_mtq_scaled + u_rw
        else:
        
            n_rw = len([a for a in est_sat.actuators if isinstance(a, RW)])
            if len(x_hat) >= 7 + n_rw:
                h_rw_states = x_hat[7 : 7 + n_rw]
            else:
                h_rw_states = np.array([rw.h for rw in est_sat.actuators if isinstance(rw, RW)])

            goal_vec_eci, w_ref_eci = goal.to_ref(os0=os_hat)
            R_b2i = rot_mat(q)
            w_ref_body = R_b2i.T @ w_ref_eci

            q_err = goal.error(q=q, body_boresight=est_sat.boresight, os0=os_hat)
            q_err = vector_alignment_error(
                q=q,
                eci_goal=goal_vec_eci,
                body_boresight=est_sat.boresight,
            )
            w_err = w - w_ref_body
            tau_pd = -self.p_gain * q_err - self.d_gain * w_err

            h_rw_body = np.zeros(3)
            rw_counter = 0
            for actuator in est_sat.actuators:
                if isinstance(actuator, RW):
                    h_rw_body += np.asarray(actuator.axis).flatten() * h_rw_states[rw_counter]
                    rw_counter += 1
            
            J = est_sat.J_0
            tau_gyro = np.cross(w, J @ w + h_rw_body)

            tau_des = tau_pd + tau_gyro
            
            b_body = np.asarray(self.M_mtm_read @ sens, float).reshape(3,)

            u_rw_cmd, u_mtq_cmd, alpha = self.allocate_max_torque_in_direction(
                tau_des, b_body, est_sat
            )

            u_out = np.zeros(len(est_sat.actuators))
            
            rw_indices = [i for i, a in enumerate(est_sat.actuators) if isinstance(a, RW)]
            u_out[rw_indices] = u_rw_cmd
            
            mtq_indices = [i for i, a in enumerate(est_sat.actuators) if isinstance(a, MTQ)]
            u_out[mtq_indices] = u_mtq_cmd

            # self.plot_torques(tau_des, b_body, est_sat)

        return u_out

    def allocate_max_torque_in_direction(self, tau_des: np.ndarray, b_body: np.ndarray, est_sat: EstimatedSatellite) -> tuple[np.ndarray, np.ndarray, float]:
        r"""
        Compute the maximum achievable torque strictly parallel to :math:`\tau_{des}`.

        This routine solves a linear program that finds commands that **maximize the achievable
        torque magnitude** along a desired direction (unit vector :math:`\hat{\tau}`), subject
        to per-actuator bounds. It returns the RW and MTQ commands and a scalar scale factor
        :math:`\alpha` representing how much of the requested torque magnitude is feasible.

        Conceptually, it computes:

        .. math::

           \max_{u,\,T\ge 0} \;\; T
           \quad \text{s.t.}\quad
           \boldsymbol{\tau}(u) = T\,\hat{\boldsymbol{\tau}},\;
           u_{min} \le u \le u_{max},

        where :math:`\hat{\boldsymbol{\tau}} = \tau_{des}/\|\tau_{des}\|`.

        If :math:`T \ge \|\tau_{des}\|`, the returned solution is scaled down linearly to match
        exactly :math:`\tau_{des}`.

        Parameters
        ----------
        tau_des : np.ndarray
            Desired body torque vector [Nm].
        b_body : np.ndarray
            Body-frame magnetic field vector [T] (or consistent internal units).
        est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
            Satellite model.

        Returns
        -------
        (u_rw, u_mtq, alpha) : tuple
            - ``u_rw`` : np.ndarray
              RW torque command(s) [Nm] (length 1 in this class).
            - ``u_mtq`` : np.ndarray
              MTQ dipole commands [A·m²] (length = number of MTQs).
            - ``alpha`` : float
              Achieved scaling factor relative to the requested magnitude.

        Notes
        -----
        This doc intentionally omits detailed allocation matrix construction since that is
        documented elsewhere in the repository. The key contract is that this allocator
        returns a solution whose net torque is colinear with :math:`\tau_{des}`.
        """
        
        t_mag = np.linalg.norm(tau_des)
        if t_mag < 1e-9:
            # Re-calculate indices just to return correct shaped zeros
            n_rw = len([a for a in est_sat.actuators if isinstance(a, RW)])
            n_mtq = len([a for a in est_sat.actuators if isinstance(a, MTQ)])
            return np.zeros(n_rw), np.zeros(n_mtq), 1.0

        # 1. Setup Matrices
        # -----------------------------------------------------
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        mtqs = [a for a in est_sat.actuators if isinstance(a, MTQ)]
        
        # Build RW Map
        if rws:
            A_rw = np.column_stack([rw.axis for rw in rws]) 
            u_rw_lims = np.array([rw.u_max for rw in rws])
        else:
            A_rw = np.zeros((3, 0))
            u_rw_lims = np.zeros(0)

        # Build MTQ Map
        if mtqs:
            b_skew = -skewsym(b_body)
            A_mtq_axes = np.column_stack([m.axis for m in mtqs])
            A_mtq = b_skew @ A_mtq_axes 
            u_mtq_lims = np.array([m.u_max for m in mtqs])
        else:
            A_mtq = np.zeros((3, 0))
            u_mtq_lims = np.zeros(0)

        A_total = np.hstack([A_rw, A_mtq])
        n_act = A_total.shape[1]

        # 2. Setup Normalized Linear Program
        # -----------------------------------------------------
        # Instead of A @ u = alpha * tau_des, we solve:
        # A @ u = T_available * tau_hat
        # This prevents the constraint column from vanishing when tau_des is small.
        
        tau_hat = tau_des / t_mag
        
        # Decision Variables x = [u_1, ..., u_n, T_available]
        # Maximize T_available => Minimize -T_available
        c = np.zeros(n_act + 1)
        c[-1] = -1.0 

        # A_eq @ x = 0  =>  [ A_total  |  -tau_hat ] @ [u; T_avail] = 0
        A_eq = np.hstack([A_total, -tau_hat.reshape(3,1)])
        b_eq = np.zeros(3)

        # Bounds
        bounds = []
        for lim_val in u_rw_lims: bounds.append((-lim_val, lim_val))
        for lim_val in u_mtq_lims: bounds.append((-lim_val, lim_val))
        
        # T_available bounds: [0, infinity] 
        # (We find the absolute max physical limit in this direction)
        bounds.append((0, None)) 

        # 3. Solve with HiGHS
        # -----------------------------------------------------
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        n_rw = len(rws)
        n_mtq = len(mtqs)

        if res.success:
            u_sol = res.x[:n_act]
            T_max_available = res.x[-1] # Max torque Nm achievable in this direction

            # 4. Scaling Logic
            # -------------------------------------------------
            # If we are strictly physically limited to less than we asked for:
            if T_max_available <= t_mag:
                # Saturated: Output max possible (u_sol is already at max)
                alpha = T_max_available / t_mag if t_mag > 0 else 0.0
                u_rw_cmd = u_sol[:n_rw]
                u_mtq_cmd = u_sol[n_rw:]
                return u_rw_cmd, u_mtq_cmd, alpha
            else:
                # Not Saturated: We have more capacity than requested.
                # Linearly scale down the commands to exactly match tau_des.
                scale_factor = t_mag / T_max_available
                u_scaled = u_sol * scale_factor
                
                u_rw_cmd = u_scaled[:n_rw]
                u_mtq_cmd = u_scaled[n_rw:]
                return u_rw_cmd, u_mtq_cmd, 1.0

        else:
            # Solver failed (usually singular geometry where NO torque is possible)
            return np.zeros(n_rw), np.zeros(n_mtq), 0.0

    def plot_torques(self, tau_des: np.ndarray, b_body: np.ndarray, est_sat: EstimatedSatellite) -> None:
        r"""
        Visualize RW + MTQ torque allocation and capacity geometry.

        This helper produces a 3D plot showing:
        - the RW achievable torque set (as a line segment or polytope),
        - the MTQ achievable torque set for the current :math:`\mathbf{B}` (a 2D set in :math:`\mathbf{B}^\perp`),
        - the allocated RW torque, MTQ torque, and their sum,
        - the requested desired torque.

        Parameters
        ----------
        tau_des : np.ndarray
            Desired body torque [Nm].
        b_body : np.ndarray
            Body-frame magnetic field [T] (or consistent internal units).
        est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
            Satellite model.

        Notes
        -----
        This is for debugging and intuition; it is not used in flight code.
        """

        # --- Solve allocation ---
        u_rw, u_mtq, alpha = self.allocate_max_torque_in_direction(tau_des, b_body, est_sat)

        rws  = [a for a in est_sat.actuators if isinstance(a, RW)]
        mtqs = [a for a in est_sat.actuators if isinstance(a, MTQ)]

        tau_rw  = sum(np.asarray(rw.axis) * u_rw[i]  for i, rw in enumerate(rws))  if rws  else np.zeros(3)
        tau_mtq = sum(np.cross(np.asarray(m.axis) * u_mtq[i], b_body)
                    for i, m in enumerate(mtqs)) if mtqs else np.zeros(3)

        tau_tot = tau_rw + tau_mtq

        # --- Build capacity point clouds ---
        rw_pts, mtq_pts = [], []

        if rws:
            limits = [[-rw.u_max, rw.u_max] for rw in rws]
            rw_pts = np.array([
                sum(np.asarray(rws[i].axis) * c[i] for i in range(len(rws)))
                for c in itertools.product(*limits)
            ])

        if mtqs and np.linalg.norm(b_body) > 1e-9:
            limits = [[-m.u_max, m.u_max] for m in mtqs]
            mtq_pts = np.array([
                np.cross(sum(np.asarray(mtqs[i].axis) * c[i] for i in range(len(mtqs))), b_body)
                for c in itertools.product(*limits)
            ])

        # --- Figure ---
        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect([1, 1, 1])

        # --- Hulls ---
        if len(rw_pts):
            self.plot_capacity(ax, rw_pts, face_color="gray", edge_color="black", alpha=0.25)

        if len(mtq_pts):
            self.plot_capacity(ax, mtq_pts, face_color="royalblue", edge_color="navy", alpha=0.25)

        # --- Vectors ---
        def vec(v, c, lbl, ls="-"):
            ax.plot([0, v[0]], [0, v[1]], [0, v[2]],
                    color=c, linewidth=2.5, linestyle=ls, label=lbl, clip_on=False)

        vec(tau_rw,  "purple",     "RW Allocation")
        vec(tau_mtq, "royalblue",  "MTQ Allocation")
        vec(tau_tot, "limegreen",  r"$\tau_{achieved}$")

        if np.linalg.norm(tau_des) > 1e-9:
            vec(tau_des, "red", r"$\tau_{des}$", ls="--")

        # --- Axis focus: MTQ hull ---
        focus_pts = mtq_pts if len(mtq_pts) else rw_pts
        if len(focus_pts):
            m = np.max(np.linalg.norm(focus_pts, axis=1))
            lim = m * 1.2
        else:
            lim = 1.0

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)

        ax.set_xlabel("X [Nm]")
        ax.set_ylabel("Y [Nm]")
        ax.set_zlabel("Z [Nm]")
        ax.set_title(f"Torque Allocation (α = {alpha:.2f})\nScroll to zoom")
        ax.legend()

        # --- Scroll zoom ---
        def on_scroll(event):
            s = 0.85 if event.button == "up" else 1.15
            for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
                lo, hi = ax.get_xlim()
                span = (hi - lo) * s / 2
                setter(-span, span)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("scroll_event", on_scroll)
        plt.show()

    
    def plot_capacity(self, ax, points: np.ndarray,
                  face_color: str,
                  edge_color: str,
                  alpha: float = 0.25):
        r"""
        Render a convex capacity set from a point cloud in 3D.

        The capacity set is determined by the rank of the point cloud:

        - Rank 1: a line segment,
        - Rank 2: a filled polygon in 3D,
        - Rank 3: a convex hull volume.

        Parameters
        ----------
        ax : matplotlib axis
            3D matplotlib axis to draw on.
        points : np.ndarray
            Point cloud representing attainable torques.
        face_color : str
            Face color used for translucent hull surfaces.
        edge_color : str
            Edge color used for silhouettes / outlines.
        alpha : float, optional
            Transparency level.

        Notes
        -----
        This method is purely for plotting; it has no effect on control behavior.
        """

        pts = np.unique(points, axis=0)
        if len(pts) < 2:
            return

        center = pts.mean(axis=0)
        U, S, Vh = np.linalg.svd(pts - center)
        rank = np.sum(S > 1e-10)

        # --- 1D: Line ---
        if rank == 1:
            proj = (pts - center) @ Vh[0]
            p0, p1 = pts[np.argmin(proj)], pts[np.argmax(proj)]
            ax.plot(*zip(p0, p1),
                    color=edge_color,
                    linewidth=4.5,
                    alpha=alpha,
                    solid_capstyle="round",
                    clip_on=False)
            return

        # --- 2D: Filled polygon ---
        if rank == 2:
            proj = (pts - center) @ Vh[:2].T
            hull = ConvexHull(proj)
            loop = pts[hull.vertices]

            poly = Poly3DCollection([loop],
                                    facecolor=face_color,
                                    edgecolor=edge_color,
                                    linewidth=1.2,
                                    alpha=alpha)
            poly.set_clip_on(False)
            ax.add_collection3d(poly)
            return

        # --- 3D: Volume hull ---
        hull = ConvexHull(pts, qhull_options="QJ")

        faces = [pts[s] for s in hull.simplices]
        poly = Poly3DCollection(faces,
                                facecolor=face_color,
                                edgecolor="none",
                                alpha=alpha)
        poly.set_clip_on(False)
        ax.add_collection3d(poly)

        # --- silhouette edges only ---
        edge_count = {}
        for s in hull.simplices:
            for i, j in ((0,1),(1,2),(2,0)):
                e = tuple(sorted((s[i], s[j])))
                edge_count[e] = edge_count.get(e, 0) + 1

        for (i, j), c in edge_count.items():
            if c == 1:
                ax.plot(*zip(pts[i], pts[j]),
                        color=edge_color,
                        linewidth=1.0,
                        alpha=0.9,
                        clip_on=False)