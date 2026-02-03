__all__ = ["MTQ_w_RW_LP"]

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import itertools
import warnings

from ADCS.CONOPS.goals import Goal, No_Goal
from ADCS.controller import Controller
from ADCS.controller.helpers.quaternion_math import vector_alignment_error
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.helpers.math_helpers import rot_mat, skewsym, limit


class MTQ_w_RW_LP(Controller):
    r"""
    Linear–Programming–Based Torque Allocation for Mixed RW–MTQ ADCS.

    This controller allocates attitude control effort between reaction wheels (RWs) and
    magnetorquers (MTQs) using a geometry-aware linear program (LP). The LP computes the
    maximum physically achievable torque colinear with a requested torque direction while
    enforcing hard actuator saturation limits and the MTQ torque-plane constraint.

    The implementation is intended for use with an estimated satellite model
    :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
    and goal objects derived from :class:`~ADCS.CONOPS.goals.Goal`.

    Actuator Models
    ----------------

    Desired body-frame control torque:

    .. math::

        \boldsymbol{\tau}_{\mathrm{des}} \in \mathbb{R}^3, \qquad
        T_{\mathrm{des}} = \|\boldsymbol{\tau}_{\mathrm{des}}\|, \qquad
        \hat{\boldsymbol{\tau}} = \frac{\boldsymbol{\tau}_{\mathrm{des}}}{T_{\mathrm{des}}}.

    Reaction wheels (for :math:`N_{\mathrm{rw}}` wheels):

    .. math::

        \boldsymbol{\tau}_{\mathrm{rw}}
        = \sum_{i=1}^{N_{\mathrm{rw}}} \boldsymbol{a}_i u_i
        = A_{\mathrm{rw}} \boldsymbol{u}_{\mathrm{rw}},
        \qquad
        |u_i| \le u_{i,\max},

    where :math:`\boldsymbol{a}_i` are unit wheel axes.

    Magnetorquers (for :math:`N_{\mathrm{mtq}}` rods) produce a dipole moment
    :math:`\boldsymbol{m} = A_{\mathrm{mtq}} \boldsymbol{u}_{\mathrm{mtq}}` and torque:

    .. math::

        \boldsymbol{\tau}_{\mathrm{mtq}}
        = \boldsymbol{m} \times \boldsymbol{B}
        = -[\boldsymbol{B}]_\times A_{\mathrm{mtq}} \boldsymbol{u}_{\mathrm{mtq}},
        \qquad
        \boldsymbol{\tau}_{\mathrm{mtq}} \perp \boldsymbol{B},

    with the skew-symmetric matrix :math:`[\boldsymbol{B}]_\times` satisfying
    :math:`[\boldsymbol{B}]_\times \boldsymbol{x} = \boldsymbol{B} \times \boldsymbol{x}`.
    The cross-product matrix is consistent with :func:`~ADCS.helpers.math_helpers.skewsym`.

    Combined actuator mapping:

    .. math::

        \boldsymbol{\tau}
        =
        \begin{bmatrix}
            A_{\mathrm{rw}} &
            -[\boldsymbol{B}]_\times A_{\mathrm{mtq}}
        \end{bmatrix}
        \begin{bmatrix}
            \boldsymbol{u}_{\mathrm{rw}} \\
            \boldsymbol{u}_{\mathrm{mtq}}
        \end{bmatrix}
        =
        A_{\mathrm{tot}} \boldsymbol{u}.

    LP Formulation
    ---------------

    Directly enforcing :math:`A_{\mathrm{tot}} \boldsymbol{u} = \boldsymbol{\tau}_{\mathrm{des}}`
    may be infeasible due to underactuation (MTQ plane) and saturation. Instead, introduce a
    scalar availability variable :math:`T_{\mathrm{avail}} \ge 0` and solve:

    .. math::

        \max_{\boldsymbol{u},\,T_{\mathrm{avail}}}\; T_{\mathrm{avail}}
        \quad \text{s.t.}\quad
        A_{\mathrm{tot}} \boldsymbol{u} - T_{\mathrm{avail}}\hat{\boldsymbol{\tau}} = \boldsymbol{0},
        \quad
        -u_{k,\max} \le u_k \le u_{k,\max},
        \quad
        T_{\mathrm{avail}} \ge 0.

    Let :math:`T_{\max}` be the optimizer value of :math:`T_{\mathrm{avail}}`.

    - If :math:`T_{\max} \ge T_{\mathrm{des}}`, the commanded effort is scaled to reproduce
      :math:`\boldsymbol{\tau}_{\mathrm{des}}` exactly.
    - If :math:`T_{\max} < T_{\mathrm{des}}`, the controller applies the maximum feasible torque
      in the requested direction.

    Define the scalar effectiveness metric:

    .. math::

        \alpha = \frac{T_{\max}}{T_{\mathrm{des}}} \in [0,1].

    Pointing and Torque-Free Momentum Management
    ---------------------------------------------

    In pointing mode (non-:class:`~ADCS.CONOPS.goals.No_Goal`), the controller:

    1. Computes a PD+gyro torque request using goal attitude error and angular-rate error.
       The goal error is obtained from :meth:`~ADCS.CONOPS.goals.Goal.error` and the reference
       rate from :meth:`~ADCS.CONOPS.goals.Goal.to_ref`.
    2. Uses the LP allocator to achieve the largest feasible torque colinear with the request.
    3. If both RWs and MTQs are present and authority remains, attempts a torque-free secondary
       desaturation by selecting MTQ torque (projected into the MTQ plane) to reduce stored wheel
       momentum and commanding an equal-and-opposite RW torque so the net additional body torque is
       approximately zero.

    In desaturation mode (:class:`~ADCS.CONOPS.goals.No_Goal`), the controller uses MTQ-only logic
    (B-dot style) with optional RW coordination to reduce rates and dump wheel momentum while
    respecting actuator limits.
    """
    def __init__(self, est_sat: EstimatedSatellite, p_gain: float, d_gain: float, c_gain: float, h_target: np.ndarray | list = np.zeros(3)) -> None:
        r"""
        Construct the LP-based mixed RW–MTQ controller.

        The constructor inspects the satellite hardware configuration in
        :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`,
        extracts MTQs and RWs, builds sensor and actuator mapping matrices using inherited
        controller helpers, and validates configuration assumptions.

        The momentum target :math:`\boldsymbol{h}_{\mathrm{target}}` is interpreted as a
        body-frame angular momentum vector to be stored in the reaction-wheel subspace.
        If the RW axis set is rank-deficient, the target is projected into the achievable
        subspace:

        .. math::

            \boldsymbol{h}_{\mathrm{ach}} = P_{\mathrm{rw}} \boldsymbol{h}_{\mathrm{target}},
            \qquad
            P_{\mathrm{rw}} = A_{\mathrm{rw}} A_{\mathrm{rw}}^{+},

        where :math:`A_{\mathrm{rw}}^{+}` is the Moore–Penrose pseudoinverse.

        :param est_sat: Estimated satellite object providing actuators, sensors, inertia,
            and boresight information.
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        :param p_gain: Proportional gain applied to the goal attitude error signal.
        :type p_gain: float
        :param d_gain: Derivative gain applied to angular-rate error (and used for B-dot style damping).
        :type d_gain: float
        :param c_gain: Momentum management gain used to drive wheel momentum toward ``h_target``.
        :type c_gain: float
        :param h_target: Body-frame target momentum vector (3,) to be stored in wheels when possible.
        :type h_target: numpy.ndarray | list
        :return: None
        :rtype: None

        """
        self.mtqs = [a for a in est_sat.actuators if isinstance(a, MTQ)]
        self.rws  = [a for a in est_sat.actuators if isinstance(a, RW)]
        
        self.n_mtq = len(self.mtqs)
        self.n_rw = len(self.rws)
        
        self.p_gain = float(p_gain)
        self.d_gain = float(d_gain)
        self.c_gain = float(c_gain)
        self.h_target = np.asarray(h_target, dtype=float).reshape(3,)

        self.M_mtm_read, _ = self.build_sensor_matrix_pinv(
            sensors=est_sat.sensors + est_sat.rw_actuators, sensor_type=MTM
        )

        self.M_mtq_act, self.mtq_indices = self.build_torque_to_u_matrix_pinv(
            actuators=est_sat.actuators, actuator_type=MTQ
        )
        self.M_rw_act, self.rw_indices = self.build_torque_to_u_matrix_pinv(
            actuators=est_sat.actuators, actuator_type=RW
        )

        self.A_mtq = self.build_u_to_torque_matrix_pinv(
            actuators=est_sat.actuators, actuator_type=MTQ
        )

        if self.n_rw > 0:
            self.rw_axes = np.column_stack([np.asarray(rw.axis, float).reshape(3,) for rw in self.rws])
        else:
            self.rw_axes = np.zeros((3, 0))

        # 8. Limits
        self.n_actuators = len(est_sat.actuators)
        self.mtq_umax = np.array([a.u_max for a in est_sat.actuators if isinstance(a, MTQ)], dtype=float)
        self.rw_umax  = np.array([a.u_max for a in est_sat.actuators if isinstance(a, RW)], dtype=float)

        self._warnings()

    def _warnings(self):
        r"""
        Emit configuration warnings and normalize the momentum target.

        This method evaluates:
        - Presence of RWs without MTQs (no momentum desaturation capability).
        - Rank of MTQ axes combined with limited RW rank (inability to desaturate while pointing).
        - Consistency of ``h_target`` with the RW achievable momentum subspace.

        If ``h_target`` contains an unachievable component, it is projected into the RW
        subspace using:

        .. math::

            \boldsymbol{h}_{\mathrm{ach}} = A_{\mathrm{rw}} A_{\mathrm{rw}}^{+}\boldsymbol{h}_{\mathrm{target}}.

        :return: None
        :rtype: None

        """
        rank_mtq = 0
        if self.n_mtq > 0:
            mtq_axes_mat = np.column_stack([np.asarray(m.axis, float).reshape(3,) for m in self.mtqs])
            rank_mtq = np.linalg.matrix_rank(mtq_axes_mat)

        rank_rw = 0
        if self.n_rw > 0:
            rank_rw = np.linalg.matrix_rank(self.rw_axes)

        if self.n_rw > 0 and self.n_mtq == 0:
            warnings.warn(
                f"\n[Controller Config] {self.n_rw} RWs detected but 0 MTQs.\n"
                "  -> CRITICAL: Momentum desaturation is NOT possible.\n"
                "  -> Reaction wheels will eventually saturate and loss of control will occur.",
                UserWarning
            )

        elif rank_mtq >= 2 and (1 <= self.n_rw <= 2):
            warnings.warn(
                f"\n[Controller Config] {self.n_rw} RWs (Rank {rank_rw}) and Rank {rank_mtq} MTQs.\n"
                "  -> LIMITATION: Torque-free desaturation (maintaining pointing while dumping) is NOT possible.\n"
                "  -> ACTION: You must use 'No_Goal' mode to desaturate. This uses B-dot and will temporarily tumble the satellite.",
                UserWarning
            )

        elif rank_mtq == 3 and rank_rw == 3:
            print(
                f"\n[Controller Config] Verified: Rank 3 RWs + Rank 3 MTQs.\n"
                "  -> Full 3-axis control with simultaneous momentum management is enabled.\n"
                "  -> The 'MTQ_w_RW' controller logic is optimal for this configuration."
            )

        if self.n_rw > 0:
            P_rw = self.rw_axes @ np.linalg.pinv(self.rw_axes)
            h_achievable = P_rw @ self.h_target
            
            diff = self.h_target - h_achievable
            if np.linalg.norm(diff) > 1e-6:
                warnings.warn(
                    f"\n[Controller Config] Requested h_target {self.h_target} is not fully achievable "
                    f"with the current RW configuration (Rank {rank_rw}).\n"
                    f"  -> Projected target to achievable subspace: {h_achievable}\n"
                    f"  -> IGNORING unachievable component: {diff}",
                    UserWarning
                )
                self.h_target = h_achievable

        elif np.linalg.norm(self.h_target) > 1e-6:
            warnings.warn(
                f"\n[Controller Config] Non-zero h_target {self.h_target} requested with 0 RWs.\n"
                "  -> Target cannot be stored. Resetting h_target to [0, 0, 0].",
                UserWarning
            )
            self.h_target = np.zeros(3)


    def find_u(
        self,
        x_hat: np.ndarray,
        sens: np.ndarray,
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        goal: Goal | None = None,
    ) -> np.ndarray:
        r"""
        Compute actuator commands for either pointing or desaturation mode.

        If ``goal`` is :class:`~ADCS.CONOPS.goals.No_Goal` (or ``None``), this method routes
        to :meth:`~ADCS.controller.MTQ_w_RW_LP.find_u_desaturate`. Otherwise it routes to
        :meth:`~ADCS.controller.MTQ_w_RW_LP.find_u_pointing`.

        The returned vector is ordered in the same actuator ordering as
        ``est_sat.actuators`` and contains MTQ and RW command signals consistent with each
        actuator's ``u_max`` bounds.

        :param x_hat: Estimated state vector. The first 3 elements are body rates
            :math:`\boldsymbol{\omega}` and elements 3:7 are the attitude quaternion.
            If present, elements 7:7+N_rw store RW scalar momentum states.
        :type x_hat: numpy.ndarray
        :param sens: Raw sensor vector used to estimate the body magnetic field via the MTM model.
        :type sens: numpy.ndarray
        :param est_sat: Estimated satellite object providing hardware configuration and inertia.
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        :param os_hat: Estimated orbital state used by the goal to compute reference vectors and rates.
        :type os_hat: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :param goal: Pointing goal. If ``None``, the controller uses :class:`~ADCS.CONOPS.goals.No_Goal`.
        :type goal: :class:`~ADCS.CONOPS.goals.Goal` | None
        :return: Actuator command vector in ``est_sat.actuators`` ordering.
        :rtype: numpy.ndarray

        """
        if goal is None:
            goal = No_Goal()

        if isinstance(goal, No_Goal):
            return self.find_u_desaturate(
                x_hat=x_hat,
                sens=sens,
                est_sat=est_sat,
                os_hat=os_hat,
                goal=goal,
            )
        else:
            return self.find_u_pointing(
                x_hat=x_hat,
                sens=sens,
                est_sat=est_sat,
                os_hat=os_hat,
                goal=goal,
            )
        
        
    def find_u_pointing(
        self,
        x_hat: np.ndarray,
        sens: np.ndarray,
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        goal: Goal,
    ) -> np.ndarray:
        r"""
        Compute actuator commands for pointing with LP allocation and torque-free desaturation.

        This method performs two stages:

        Stage 1: LP torque allocation in the requested direction
        ---------------------------------------------------------

        A desired control torque is computed as PD feedback plus a gyroscopic compensation term:

        .. math::

            \boldsymbol{\tau}_{\mathrm{pd}} =
            -k_p \boldsymbol{q}_{\mathrm{err}} - k_d(\boldsymbol{\omega} - \boldsymbol{\omega}_{\mathrm{ref}}),

        where :math:`\boldsymbol{q}_{\mathrm{err}}` is obtained from
        :meth:`~ADCS.CONOPS.goals.Goal.error`, and :math:`\boldsymbol{\omega}_{\mathrm{ref}}`
        is the body-frame reference rate from :meth:`~ADCS.CONOPS.goals.Goal.to_ref` and
        :func:`~ADCS.helpers.math_helpers.rot_mat`.

        The gyroscopic term uses satellite inertia :math:`J` and wheel momentum:

        .. math::

            \boldsymbol{h}_{\mathrm{rw}} = A_{\mathrm{rw}}\boldsymbol{h}_{\mathrm{rw,scalars}},
            \qquad
            \boldsymbol{\tau}_{\mathrm{gyro}} =
            \boldsymbol{\omega} \times (J\boldsymbol{\omega} + \boldsymbol{h}_{\mathrm{rw}}),

        giving:

        .. math::

            \boldsymbol{\tau}_{\mathrm{des}} = \boldsymbol{\tau}_{\mathrm{pd}} + \boldsymbol{\tau}_{\mathrm{gyro}}.

        The body magnetic field :math:`\boldsymbol{B}` is reconstructed from MTM readings using the
        sensor mapping built in the constructor. The LP allocator
        :meth:`~ADCS.controller.MTQ_w_RW_LP.allocate_max_torque_in_direction` then returns a
        feasible command achieving :math:`\alpha \boldsymbol{\tau}_{\mathrm{des}}` with
        :math:`\alpha \in [0,1]`.

        Stage 2: torque-free momentum desaturation using leftover authority
        --------------------------------------------------------------------

        If both RWs and MTQs are present, the controller attempts to reduce wheel momentum error
        without introducing additional net body torque. Define the body-frame wheel momentum error:

        .. math::

            \boldsymbol{h}_{\mathrm{err}} = \boldsymbol{h}_{\mathrm{rw}} - \boldsymbol{h}_{\mathrm{target}}.

        A desired dumping torque is:

        .. math::

            \boldsymbol{\tau}_{\mathrm{dump}}^{\ast} = -k_c \boldsymbol{h}_{\mathrm{err}}.

        Because MTQ torque must satisfy :math:`\boldsymbol{\tau}_{\mathrm{mtq}} \perp \boldsymbol{B}`,
        the dump torque is projected into the achievable plane:

        .. math::

            \boldsymbol{\tau}_{\mathrm{mtq,sec}}
            =
            \boldsymbol{\tau}_{\mathrm{dump}}^{\ast}
            - (\boldsymbol{\tau}_{\mathrm{dump}}^{\ast} \cdot \hat{\boldsymbol{B}})\hat{\boldsymbol{B}},
            \qquad
            \hat{\boldsymbol{B}} = \frac{\boldsymbol{B}}{\|\boldsymbol{B}\|}.

        Torque-free cancellation is imposed by selecting the secondary wheel torque:

        .. math::

            \boldsymbol{\tau}_{\mathrm{rw,sec}} = -\boldsymbol{\tau}_{\mathrm{mtq,sec}}.

        The method maps these secondary torques into actuator commands via pseudoinverses, then
        computes the largest scalar :math:`\beta \in [0,1]` such that the combined primary+secondary
        commands remain within hard bounds:

        .. math::

            -u_{k,\max} \le u_k^{\mathrm{primary}} + \beta u_k^{\mathrm{sec}} \le u_{k,\max}.

        This coupled scaling preserves the near-zero net secondary torque by avoiding independent
        clipping of RW and MTQ channels.

        :param x_hat: Estimated state vector containing body rates, quaternion, and optionally RW momentum states.
        :type x_hat: numpy.ndarray
        :param sens: Sensor vector used to estimate the body magnetic field from MTM measurements.
        :type sens: numpy.ndarray
        :param est_sat: Estimated satellite object containing actuators, inertia, and boresight configuration.
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        :param os_hat: Estimated orbital state used by the goal to define reference vectors and rates.
        :type os_hat: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :param goal: Pointing goal providing reference vector and attitude error computation.
        :type goal: :class:`~ADCS.CONOPS.goals.Goal`
        :return: Actuator command vector in ``est_sat.actuators`` ordering, including any torque-free
            desaturation contribution that fits within remaining authority.
        :rtype: numpy.ndarray

        """
        w = x_hat[0:3]
        q = x_hat[3:7]

        rws  = [a for a in est_sat.actuators if isinstance(a, RW)]
        mtqs = [a for a in est_sat.actuators if isinstance(a, MTQ)]
        n_rw, n_mtq = len(rws), len(mtqs)

        rw_indices  = [i for i, a in enumerate(est_sat.actuators) if isinstance(a, RW)]
        mtq_indices = [i for i, a in enumerate(est_sat.actuators) if isinstance(a, MTQ)]

        # Wheel scalar momentum states (N_rw,)
        if n_rw > 0 and len(x_hat) >= 7 + n_rw:
            h_rw_states = np.asarray(x_hat[7 : 7 + n_rw], float).reshape(n_rw,)
        elif n_rw > 0:
            h_rw_states = np.array([rw.h for rw in rws], dtype=float).reshape(n_rw,)
        else:
            h_rw_states = np.zeros(0)

        # -------------------------
        # 1) Pointing torque request (PD + gyro)
        # -------------------------
        goal_vec_eci, w_ref_eci = goal.to_ref(os0=os_hat)
        R_b2i = rot_mat(q)
        w_ref_body = R_b2i.T @ w_ref_eci

        q_err = goal.error(q=q, body_boresight=est_sat.boresight, os0=os_hat)
        w_err = w - w_ref_body
        tau_pd = -self.p_gain * q_err - self.d_gain * w_err

        # Build A_rw (3 x n_rw) and body wheel momentum vector (3,)
        if n_rw > 0:
            A_rw = np.column_stack([np.asarray(rw.axis, float).reshape(3,) for rw in rws])  # (3, n_rw)
            h_rw_body = A_rw @ h_rw_states
        else:
            A_rw = np.zeros((3, 0))
            h_rw_body = np.zeros(3)

        J = est_sat.J_0
        tau_gyro = np.cross(w, J @ w + h_rw_body)

        tau_des = tau_pd + tau_gyro

        # -------------------------
        # 2) Magnetic field in body frame
        # -------------------------
        sens_clean = np.asarray(sens, float).copy()
        sens_clean[np.isnan(sens_clean)] = 0.0
        b_body = np.asarray(self.M_mtm_read @ sens_clean, float).reshape(3,)

        # -------------------------
        # 3) Primary allocation: max feasible torque in tau_des direction (your LP)
        # -------------------------
        u_rw_cmd, u_mtq_cmd, alpha = self.allocate_max_torque_in_direction(tau_des, b_body, est_sat)

        # Assemble baseline command vector in *actuator ordering*
        u_out = np.zeros(len(est_sat.actuators), dtype=float)
        if n_rw > 0:
            u_out[rw_indices] = np.asarray(u_rw_cmd, float).reshape(n_rw,)
        if n_mtq > 0:
            u_out[mtq_indices] = np.asarray(u_mtq_cmd, float).reshape(n_mtq,)

        # -------------------------
        # 4) Secondary: torque-free desaturation using leftover authority
        # -------------------------
        # Needs both RWs and MTQs to be meaningful
        if n_rw == 0 or n_mtq == 0:
            return u_out

        # If B is near zero, MTQ torque map is ill-conditioned
        b_norm = np.linalg.norm(b_body)
        if b_norm < 1e-9:
            return u_out

        b_hat = b_body / b_norm

        # ---- 4a) Compute desired momentum reduction torque (body-frame) ----
        # Current stored wheel momentum vector (3,) is h_rw_body already
        # self.h_target is BODY (3,)
        h_err_body = h_rw_body - np.asarray(self.h_target, float).reshape(3,)
        # Desired torque to reduce this momentum error (Hogan & Schaub "dump" direction)
        tau_dump_des = -self.c_gain * h_err_body

        # MTQs can only produce torque perpendicular to B: project into achievable plane
        tau_mtq_sec = tau_dump_des - np.dot(tau_dump_des, b_hat) * b_hat  # Pi_perpB(tau_dump_des)

        # If projected torque is tiny, nothing useful to do torque-free right now
        if np.linalg.norm(tau_mtq_sec) < 1e-12:
            return u_out

        # Torque-free requirement: tau_rw_sec + tau_mtq_sec = 0
        tau_rw_sec = -tau_mtq_sec

        # ---- 4b) Map secondary torques to actuator commands (pinv, Hogan&Schaub style) ----
        # RW: tau_rw = A_rw u_rw  =>  u_rw = A_rw^+ tau_rw
        u_rw_sec = np.linalg.pinv(A_rw) @ tau_rw_sec  # (n_rw,)

        # MTQ: tau_mtq = -[B]x A_mtq_axes u_mtq  => u_mtq = pinv(M_mag_eff) tau_mtq
        A_mtq_axes = np.column_stack([np.asarray(m.axis, float).reshape(3,) for m in mtqs])  # (3, n_mtq)
        M_mag_eff = -skewsym(b_body) @ A_mtq_axes                                          # (3, n_mtq)
        u_mtq_sec = np.linalg.pinv(M_mag_eff) @ tau_mtq_sec                                 # (n_mtq,)

        # ---- 4c) Use ONLY leftover authority around u_out, scale coupled (RW+MTQ together) ----
        # Build stacked vectors in the same ordering as u_out's RW/MTQ slots
        u_star_rw  = u_out[rw_indices].copy()
        u_star_mtq = u_out[mtq_indices].copy()

        # Hard limits
        rw_umax  = np.array([rw.u_max for rw in rws], dtype=float)
        mtq_umax = np.array([m.u_max  for m in mtqs], dtype=float)

        # Find the largest beta in [0,1] such that:
        #   u_star + beta*u_sec stays inside [-umax, umax] for BOTH RW and MTQ
        beta = 1.0

        # RWs
        for i in range(n_rw):
            si = u_rw_sec[i]
            if abs(si) < 1e-12:
                continue
            ui = u_star_rw[i]
            if si > 0:
                beta = min(beta, ( rw_umax[i] - ui) / si)
            else:
                beta = min(beta, (-rw_umax[i] - ui) / si)

        # MTQs
        for i in range(n_mtq):
            si = u_mtq_sec[i]
            if abs(si) < 1e-12:
                continue
            ui = u_star_mtq[i]
            if si > 0:
                beta = min(beta, ( mtq_umax[i] - ui) / si)
            else:
                beta = min(beta, (-mtq_umax[i] - ui) / si)

        beta = float(np.clip(beta, 0.0, 1.0))
        if beta <= 0.0:
            return u_out

        # Apply coupled scaled secondary command (no independent clipping!)
        u_out[rw_indices]  = u_star_rw  + beta * u_rw_sec
        u_out[mtq_indices] = u_star_mtq + beta * u_mtq_sec

        # Optional: numerical safety (should already be within limits)
        u_out[rw_indices]  = np.clip(u_out[rw_indices],  -rw_umax,  rw_umax)
        u_out[mtq_indices] = np.clip(u_out[mtq_indices], -mtq_umax, mtq_umax)

        return u_out
    

    def find_u_desaturate(
        self,
        x_hat: np.ndarray,
        sens: np.ndarray,
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        goal: Goal | None = None,    
    ):
        r"""
        Compute actuator commands for desaturation mode (No_Goal).

        This mode prioritizes:
        - Rate damping in the MTQ-achievable plane (perpendicular to :math:`\boldsymbol{B}`).
        - Momentum dumping toward :math:`\boldsymbol{h}_{\mathrm{target}}` when RWs exist.

        Magnetic field estimation
        -------------------------

        The body magnetic field :math:`\boldsymbol{B}` is reconstructed from MTM readings.
        If :math:`\|\boldsymbol{B}\|` is near zero, no MTQ torque is possible and this method
        returns zeros.

        Rate damping (B-dot style plane damping)
        ----------------------------------------

        MTQs can only generate torque perpendicular to :math:`\boldsymbol{B}`. The method dampens
        body rates only in that plane. Let:

        .. math::

            \hat{\boldsymbol{B}} = \frac{\boldsymbol{B}}{\|\boldsymbol{B}\|}, \qquad
            \boldsymbol{\omega}_{\perp} = \boldsymbol{\omega} - (\boldsymbol{\omega}\cdot\hat{\boldsymbol{B}})\hat{\boldsymbol{B}}.

        The plane-damping torque is:

        .. math::

            \boldsymbol{\tau}_{\mathrm{bdot}} = -k_{\omega}\boldsymbol{\omega}_{\perp}.

        Momentum dumping
        ----------------

        For RWs, the stored body-frame wheel momentum is:

        .. math::

            \boldsymbol{h}_{\mathrm{rw}} = A_{\mathrm{rw}}\boldsymbol{h}_{\mathrm{rw,scalars}}.

        The error relative to the target is:

        .. math::

            \boldsymbol{h}_{\mathrm{err}} = \boldsymbol{h}_{\mathrm{rw}} - \boldsymbol{h}_{\mathrm{target}}.

        A nominal dumping torque is computed (gain and sign follow the implementation):

        .. math::

            \boldsymbol{\tau}_{\mathrm{dump}}^{\ast} = k_h \boldsymbol{h}_{\mathrm{err}}.

        The commanded dump torque is then projected into the MTQ-achievable plane:

        .. math::

            \boldsymbol{\tau}_{\mathrm{dump},\perp}
            =
            \boldsymbol{\tau}_{\mathrm{dump}}^{\ast}
            - (\boldsymbol{\tau}_{\mathrm{dump}}^{\ast}\cdot\hat{\boldsymbol{B}})\hat{\boldsymbol{B}}.

        MTQ allocation and saturation scaling
        -------------------------------------

        The MTQ command is computed by a pseudoinverse of the effective MTQ torque map:

        .. math::

            A_{\mathrm{mtq,eff}} = -[\boldsymbol{B}]_\times A_{\mathrm{mtq}},
            \qquad
            \boldsymbol{u}_{\mathrm{mtq}} = A_{\mathrm{mtq,eff}}^{+}\boldsymbol{\tau}_{\mathrm{mtq,des}}.

        If any MTQ channel exceeds its limit, a scalar scale factor :math:`\alpha_{\mathrm{mtq}} \in [0,1]`
        is applied to keep commands feasible.

        RW coordination
        ---------------

        When MTQ authority is limited (small :math:`\alpha_{\mathrm{mtq}}`), the RW torque request is
        scaled accordingly to avoid injecting uncompensated body torque. The RW command is produced using
        the RW torque-to-command mapping built in the constructor and then saturated to bounds.

        :param x_hat: Estimated state vector containing body rates, quaternion, and optionally RW momentum states.
        :type x_hat: numpy.ndarray
        :param sens: Sensor vector used to estimate the body magnetic field from MTM measurements.
        :type sens: numpy.ndarray
        :param est_sat: Estimated satellite object providing actuators and inertia.
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        :param os_hat: Estimated orbital state (not required by the current desaturation logic but included for interface compatibility).
        :type os_hat: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :param goal: Optional goal (unused in this mode). Included for signature compatibility.
        :type goal: :class:`~ADCS.CONOPS.goals.Goal` | None
        :return: Actuator command vector in ``est_sat.actuators`` ordering for desaturation mode.
        :rtype: numpy.ndarray

        """
        w = x_hat[0:3]
        q = x_hat[3:7]

        k_w = self.d_gain
        k_h = self.c_gain

        # --- 1. Magnetic Field (Body Frame) ---
        sens_clean = np.asarray(sens, float).copy()
        sens_clean[np.isnan(sens_clean)] = 0.0
        b_body = np.asarray(self.M_mtm_read @ sens_clean, float).reshape(3,)
        b_norm = np.linalg.norm(b_body)
        if b_norm < 1e-9:
            return np.zeros(self.n_actuators)
        b_hat = b_body / b_norm

        # --- 2. System Momentum Vector (Generic N-RW) ---
        # Calculate total vector momentum stored in all wheels
        if self.n_rw > 0 and len(x_hat) >= 7 + self.n_rw:
            h_rw_scalars = x_hat[7 : 7 + self.n_rw]
            h_sys = self.rw_axes @ h_rw_scalars # Matrix (3, N) @ Vec (N,) -> (3,)
        else:
            h_sys = np.zeros(3)

        # --- 3. Rate Damping (Perpendicular to B) ---
        # "Magnetic B-dot" logic: dampen rates only in the plane where MTQs can act
        w_perp = w - np.dot(w, b_hat) * b_hat
        tau_bdot_perp = -k_w * w_perp

        # --- 4. Momentum Dumping Torque (3D Vector) ---
        # Desired torque on body to reduce system momentum error
        h_err = h_sys - self.h_target  # Vector subtraction
        tau_dump_des = k_h * h_err     # Gain * Vector error

        # Saturation: Limit the dumping torque magnitude to avoid transient spikes
        mag = np.linalg.norm(tau_dump_des)
        limit_val = k_h * 0.001        # Example cap (adjust as needed)
        if mag > limit_val:
            tau_dump_des *= limit_val / (mag + 1e-12)

        # Project dump torque into MTQ-achievable plane (Perp to B)
        tau_dump_perp = tau_dump_des - np.dot(tau_dump_des, b_hat) * b_hat

        # Gating: Avoid dumping if the required torque is parallel to B (uncancellable)
        denom = np.linalg.norm(tau_dump_des) + 1e-12
        gamma = np.linalg.norm(tau_dump_perp) / denom
        tau_dump_cmd = gamma * tau_dump_des

        # --- 5. MTQ Allocation & Saturation Check ---
        tau_mtq_des = tau_bdot_perp - tau_dump_perp
        
        alpha_mtq = 0.0 # Default to 0: If no MTQs, we cannot dump.
        u_mtq_scaled = np.zeros(self.n_mtq) # Default empty/zero

        if self.n_mtq > 0:
            # Standard Allocation
            B_skew = skewsym(b_body)
            M_mag_eff = -B_skew @ self.A_mtq
            u_mtq_raw = np.linalg.pinv(M_mag_eff) @ tau_mtq_des

            # Check saturation
            mtq_cmds = u_mtq_raw[self.mtq_indices]
            alpha_mtq = 1.0
            if np.any(np.abs(mtq_cmds) > self.mtq_umax):
                alpha_mtq = np.min(self.mtq_umax / (np.abs(mtq_cmds) + 1e-12))
            
            u_mtq_scaled = alpha_mtq * u_mtq_raw

        # --- 6. RW Command (Scaled) ---
        # If alpha_mtq is 0 (due to saturation or 0 MTQs), RW torque is reduced 
        # to 0 to prevent spinning up the body.
        tau_rw_req = alpha_mtq * (tau_dump_cmd + tau_bdot_perp)

        u_rw = self.M_rw_act @ tau_rw_req
        u_rw = np.clip(u_rw, -self.rw_umax, self.rw_umax)

        # --- Final Output ---
        u_out = u_mtq_scaled + u_rw  
        return u_out
        

        

    def allocate_max_torque_in_direction(self, tau_des: np.ndarray, b_body: np.ndarray, est_sat: EstimatedSatellite) -> tuple[np.ndarray, np.ndarray, float]:    
        r"""
        Solve the LP to maximize feasible torque colinear with a desired torque direction.

        This method implements the normalized LP:

        .. math::

            \max_{\boldsymbol{u},\,T_{\mathrm{avail}}}\; T_{\mathrm{avail}}
            \quad \text{s.t.}\quad
            A_{\mathrm{tot}}\boldsymbol{u} = T_{\mathrm{avail}}\hat{\boldsymbol{\tau}},
            \quad
            -u_{k,\max} \le u_k \le u_{k,\max},
            \quad
            T_{\mathrm{avail}} \ge 0,

        where :math:`\hat{\boldsymbol{\tau}} = \boldsymbol{\tau}_{\mathrm{des}}/\|\boldsymbol{\tau}_{\mathrm{des}}\|`.
        The combined map is:

        .. math::

            A_{\mathrm{tot}} =
            \begin{bmatrix}
                A_{\mathrm{rw}} & -[\boldsymbol{B}]_\times A_{\mathrm{mtq,axes}}
            \end{bmatrix}.

        The MTQ block uses the body magnetic field :math:`\boldsymbol{B}` and enforces the plane
        constraint :math:`\boldsymbol{\tau}_{\mathrm{mtq}} \perp \boldsymbol{B}` by construction.

        Output scaling
        --------------

        Let :math:`T_{\max}` be the optimizer value of :math:`T_{\mathrm{avail}}` and
        :math:`T_{\mathrm{des}} = \|\boldsymbol{\tau}_{\mathrm{des}}\|`.

        - If :math:`T_{\max} \le T_{\mathrm{des}}`, the system is saturated and the method returns
          the maximizing command and :math:`\alpha = T_{\max}/T_{\mathrm{des}}`.
        - If :math:`T_{\max} > T_{\mathrm{des}}`, the method scales the maximizing command by
          :math:`T_{\mathrm{des}}/T_{\max}` to match :math:`\boldsymbol{\tau}_{\mathrm{des}}` exactly
          and returns :math:`\alpha = 1`.

        Degenerate cases
        ----------------

        If :math:`\|\boldsymbol{\tau}_{\mathrm{des}}\|` is near zero, the method returns zeros and
        :math:`\alpha = 1`. If the LP solver fails (e.g., no torque achievable in the requested direction),
        the method returns zeros and :math:`\alpha = 0`.

        :param tau_des: Desired body-frame torque vector :math:`\boldsymbol{\tau}_{\mathrm{des}}`.
        :type tau_des: numpy.ndarray
        :param b_body: Body-frame magnetic field vector :math:`\boldsymbol{B}`.
        :type b_body: numpy.ndarray
        :param est_sat: Estimated satellite providing the RW and MTQ actuator sets and limits.
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        :return: Tuple ``(u_rw_cmd, u_mtq_cmd, alpha)`` where ``u_rw_cmd`` has shape ``(N_rw,)``,
            ``u_mtq_cmd`` has shape ``(N_mtq,)``, and ``alpha`` is the achievable torque fraction.
        :rtype: tuple[numpy.ndarray, numpy.ndarray, float]

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
        Plot achievable torque envelopes and the allocated torque vectors in 3D.

        This method:
        - Calls :meth:`~ADCS.controller.MTQ_w_RW_LP.allocate_max_torque_in_direction` to obtain RW and MTQ commands and the effectiveness metric :math:`\alpha`.
        - Constructs representative point clouds of extreme actuator combinations (hypercube corners) for RWs and MTQs and draws their convex hulls.
        - Plots vectors for RW torque, MTQ torque, total achieved torque, and desired torque.

        Torque reconstruction
        ----------------------

        RW torque is reconstructed as:

        .. math::

            \boldsymbol{\tau}_{\mathrm{rw}} = A_{\mathrm{rw}}\boldsymbol{u}_{\mathrm{rw}}.

        MTQ torque is reconstructed from dipole moment and magnetic field:

        .. math::

            \boldsymbol{\tau}_{\mathrm{mtq}} = \boldsymbol{m} \times \boldsymbol{B},
            \qquad
            \boldsymbol{m} = A_{\mathrm{mtq,axes}}\boldsymbol{u}_{\mathrm{mtq}}.

        The achieved torque is:

        .. math::

            \boldsymbol{\tau}_{\mathrm{ach}} = \boldsymbol{\tau}_{\mathrm{rw}} + \boldsymbol{\tau}_{\mathrm{mtq}}.

        Hull plotting uses :meth:`~ADCS.controller.MTQ_w_RW_LP.plot_capacity`.

        :param tau_des: Desired body-frame torque vector to visualize.
        :type tau_des: numpy.ndarray
        :param b_body: Body-frame magnetic field vector used for MTQ torque mapping.
        :type b_body: numpy.ndarray
        :param est_sat: Estimated satellite providing actuator configuration and limits.
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        :return: None
        :rtype: None

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
        Plot a convex capacity set (line, polygon, or 3D hull) from a point cloud.

        The method determines the intrinsic rank (dimension) of the point set by SVD and renders:
        - Rank 1: a line segment between extreme projections.
        - Rank 2: a filled convex polygon in 3D via a 2D convex hull in the best-fit plane.
        - Rank 3: a 3D convex hull surface, plus silhouette edges.

        Rank detection
        ---------------

        Let :math:`P \in \mathbb{R}^{N \times 3}` be the unique points and :math:`\bar{\boldsymbol{p}}`
        be their centroid. Compute the SVD of centered points:

        .. math::

            P_c = P - \mathbf{1}\bar{\boldsymbol{p}}^{\mathsf{T}} = U \Sigma V^{\mathsf{T}}.

        The number of singular values above a tolerance defines the effective rank, which selects
        the plotting primitive.

        :param ax: A Matplotlib 3D axis object used for plotting.
        :type ax: matplotlib.axes.Axes
        :param points: Point cloud in torque space with shape ``(N, 3)``.
        :type points: numpy.ndarray
        :param face_color: Face color for polygon or hull rendering.
        :type face_color: str
        :param edge_color: Edge color for line/polygon edges and hull silhouettes.
        :type edge_color: str
        :param alpha: Transparency value used for filled primitives.
        :type alpha: float
        :return: None
        :rtype: None

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