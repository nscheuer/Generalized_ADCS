__all__ = ["MTQ_w_RW_QP"]

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import lsq_linear
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import itertools

from ADCS.CONOPS.goals import Goal, No_Goal
from ADCS.controller import Controller, MTQ_w_RW_LP
from ADCS.controller.helpers.quaternion_math import vector_alignment_error
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.helpers.math_helpers import rot_mat, skewsym, limit


class MTQ_w_RW_QP(MTQ_w_RW_LP):
    r"""
    Quadratic–Programming–Based Torque Allocation for Mixed RW–MTQ ADCS.

    This controller allocates actuator commands for a mixed-actuation attitude control system
    composed of reaction wheels (RWs) and magnetorquers (MTQs). It computes bounded actuator
    inputs that best reproduce a desired body torque in a least-squares sense, while strictly
    respecting actuator saturation limits.
    Original allocator paper: `Generalized Attitude Control for Small Spacecraft
    <https://digitalcommons.usu.edu/smallsat/2026/all2026/253/>`__

    The class extends :class:`~ADCS.controller.MTQ_w_RW_LP` and replaces the LP-style directional
    prioritization with a bounded least-squares formulation solved via
    :meth:`scipy.optimize.lsq_linear`.

    Actuator Model
    ---------------

    Let the desired body torque be :math:`\boldsymbol{\tau}_{\mathrm{des}} \in \mathbb{R}^3`.
    Stack RW and MTQ commands into a single vector

    .. math::

        \boldsymbol{u}
        =
        \begin{bmatrix}
        \boldsymbol{u}_{\mathrm{rw}} \\
        \boldsymbol{u}_{\mathrm{mtq}}
        \end{bmatrix}
        \in \mathbb{R}^{n_{\mathrm{rw}} + n_{\mathrm{mtq}}}.

    Reaction wheels generate torque along their spin axes. With RW unit axes stacked as columns
    in :math:`A_{\mathrm{rw}} \in \mathbb{R}^{3 \times n_{\mathrm{rw}}}`, the RW torque is

    .. math::

        \boldsymbol{\tau}_{\mathrm{rw}} = A_{\mathrm{rw}} \, \boldsymbol{u}_{\mathrm{rw}}.

    Magnetorquers generate a magnetic dipole :math:`\boldsymbol{m}` that produces torque via
    the cross product with the geomagnetic field :math:`\boldsymbol{B}`:

    .. math::

        \boldsymbol{\tau}_{\mathrm{mtq}} = \boldsymbol{m} \times \boldsymbol{B}
        =
        -[\boldsymbol{B}]_{\times}\,\boldsymbol{m}.

    If MTQ axes are stacked in :math:`A_{\mathrm{mtq}} \in \mathbb{R}^{3 \times n_{\mathrm{mtq}}}`,
    and :math:`\boldsymbol{u}_{\mathrm{mtq}}` scales dipole moments along those axes, then

    .. math::

        \boldsymbol{\tau}_{\mathrm{mtq}}
        =
        \left(-[\boldsymbol{B}]_{\times} A_{\mathrm{mtq}}\right)\boldsymbol{u}_{\mathrm{mtq}}.

    The combined mapping becomes

    .. math::

        A_{\mathrm{tot}}
        =
        \begin{bmatrix}
        A_{\mathrm{rw}} & -[\boldsymbol{B}]_{\times} A_{\mathrm{mtq}}
        \end{bmatrix},
        \qquad
        \boldsymbol{\tau}_{\mathrm{ach}} = A_{\mathrm{tot}} \boldsymbol{u}.

    Bounded Least Squares Allocation
    ----------------------------------

    The allocation is formulated as a box-constrained least-squares problem:

    .. math::

        \min_{\boldsymbol{u}}
        \;\frac{1}{2}\,\left\|A_{\mathrm{tot}} \boldsymbol{u} - \boldsymbol{\tau}_{\mathrm{des}}\right\|_2^2

    subject to actuator limits

    .. math::

        -u_{i,\max} \le u_i \le u_{i,\max}, \qquad \forall i.

    This approach minimizes Euclidean torque error and naturally returns the closest feasible
    torque when constraints or MTQ underactuation prevent exact tracking (e.g., torque requests
    parallel to :math:`\boldsymbol{B}`).

    Post-hoc Effectiveness Metric
    ------------------------------

    After solving for :math:`\boldsymbol{u}^*`, the achieved torque is projected onto the desired
    torque direction to compute an effectiveness scalar :math:`\alpha`:

    .. math::

        \hat{\boldsymbol{\tau}} = \frac{\boldsymbol{\tau}_{\mathrm{des}}}{\|\boldsymbol{\tau}_{\mathrm{des}}\|},
        \qquad
        \alpha = \frac{\boldsymbol{\tau}_{\mathrm{ach}} \cdot \hat{\boldsymbol{\tau}}}{\|\boldsymbol{\tau}_{\mathrm{des}}\|}.

    Values near :math:`1` indicate the request is feasible; smaller values indicate saturation
    or geometric infeasibility.

    :param est_sat: Estimated satellite model providing actuator instances and configuration, via
                    :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`.
    :type est_sat: :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
    :param p_gain: Proportional gain used by the parent controller law.
    :type p_gain: float
    :param d_gain: Derivative gain used by the parent controller law.
    :type d_gain: float
    :param c_gain: Additional control gain used by the parent controller law.
    :type c_gain: float
    :param h_target: Target wheel momentum vector used by the parent mixed-actuation controller.
    :type h_target: numpy.ndarray | list

    """
    def __init__(self, est_sat: Satellite, p_gain: float, d_gain: float, c_gain: float, h_target: np.ndarray | list = np.zeros(3)) -> None:
        r"""
        Construct a QP-style allocator for mixed reaction wheel and magnetorquer actuation.

        This initializer delegates setup to :class:`~ADCS.controller.MTQ_w_RW_LP`, preserving the
        same external controller interface while changing the internal allocation strategy used
        by :meth:`~ADCS.controller.MTQ_w_RW_QP.allocate_max_torque_in_direction`.

        :param est_sat: Estimated satellite model providing actuators and current state estimates.
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
        :param p_gain: Proportional gain used by the parent control law.
        :type p_gain: float
        :param d_gain: Derivative gain used by the parent control law.
        :type d_gain: float
        :param c_gain: Additional gain used by the parent control law.
        :type c_gain: float
        :param h_target: Target wheel momentum vector used for momentum management.
        :type h_target: numpy.ndarray | list
        :return: None.
        :rtype: NoneType

        """
        super().__init__(est_sat=est_sat, p_gain=p_gain, d_gain=d_gain, c_gain=c_gain, h_target=h_target)


    def allocate_max_torque_in_direction(self, tau_des: np.ndarray, b_body: np.ndarray, est_sat: Satellite) -> tuple[np.ndarray, np.ndarray, float]:
        r"""
        Allocate bounded RW and MTQ commands that best achieve a desired body torque.

        The method constructs the combined actuator mapping

        .. math::

            A_{\mathrm{tot}}
            =
            \begin{bmatrix}
            A_{\mathrm{rw}} & -[\boldsymbol{B}]_{\times} A_{\mathrm{mtq}}
            \end{bmatrix}

        using RW axes from :class:`~ADCS.satellite_hardware.actuators.RW` actuators and MTQ axes
        from :class:`~ADCS.satellite_hardware.actuators.MTQ` actuators contained in
        :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`.

        It then solves the box-constrained least-squares problem

        .. math::

            \min_{\boldsymbol{u}}
            \;\frac{1}{2}\,\left\|A_{\mathrm{tot}} \boldsymbol{u} - \boldsymbol{\tau}_{\mathrm{des}}\right\|_2^2,
            \qquad
            -u_{i,\max} \le u_i \le u_{i,\max}

        where :math:`u_{i,\max}` are the per-actuator saturation limits (RW torque bounds and MTQ
        dipole-moment bounds, as represented in the actuator objects). The solver used is
        :meth:`scipy.optimize.lsq_linear`, which provides a robust Trust Region Reflective method
        suitable for rank-deficient cases common in magnetic control.

        For the solved command :math:`\boldsymbol{u}^*`, the achieved torque is

        .. math::

            \boldsymbol{\tau}_{\mathrm{ach}} = A_{\mathrm{tot}} \boldsymbol{u}^*.

        A post-hoc effectiveness scalar :math:`\alpha` is computed as the achieved torque component
        along the desired direction, normalized by requested magnitude:

        .. math::

            \hat{\boldsymbol{\tau}} = \frac{\boldsymbol{\tau}_{\mathrm{des}}}{\|\boldsymbol{\tau}_{\mathrm{des}}\|},
            \qquad
            \alpha = \max\!\left(0,\frac{\boldsymbol{\tau}_{\mathrm{ach}} \cdot \hat{\boldsymbol{\tau}}}{\|\boldsymbol{\tau}_{\mathrm{des}}\|}\right).

        Edge Cases
        ----------

        If :math:`\|\boldsymbol{\tau}_{\mathrm{des}}\|` is near zero, the method returns zero commands
        for all actuators and :math:`\alpha = 1` (trivially satisfied request). If there are no
        actuators, it returns empty command vectors and :math:`\alpha = 0`.

        :param tau_des: Desired body-frame torque vector :math:`\boldsymbol{\tau}_{\mathrm{des}}` in N·m.
        :type tau_des: numpy.ndarray
        :param b_body: Body-frame geomagnetic field vector :math:`\boldsymbol{B}` in tesla.
        :type b_body: numpy.ndarray
        :param est_sat: Estimated satellite model providing actuator instances and their limits.
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
        :return: Tuple of RW commands, MTQ commands, and effectiveness scalar :math:`\alpha`.
                 The RW command vector has length equal to the number of
                 :class:`~ADCS.satellite_hardware.actuators.RW` actuators. The MTQ command vector has
                 length equal to the number of :class:`~ADCS.satellite_hardware.actuators.MTQ` actuators.
        :rtype: tuple[numpy.ndarray, numpy.ndarray, float]

        """
        tau_des = np.asarray(tau_des, float).reshape(3,)
        t_mag = np.linalg.norm(tau_des)
        if t_mag < 1e-9:
            n_rw = len([a for a in est_sat.actuators if isinstance(a, RW)])
            n_mtq = len([a for a in est_sat.actuators if isinstance(a, MTQ)])
            return np.zeros(n_rw), np.zeros(n_mtq), 1.0

        # 1) Setup matrices exactly like before
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        mtqs = [a for a in est_sat.actuators if isinstance(a, MTQ)]

        # RW torque map: tau_rw = A_rw * u_rw
        if rws:
            A_rw = np.column_stack([np.asarray(rw.axis, float).reshape(3,) for rw in rws])
            u_rw_lims = np.array([rw.u_max for rw in rws], dtype=float)
        else:
            A_rw = np.zeros((3, 0))
            u_rw_lims = np.zeros(0, dtype=float)

        # MTQ torque map: tau_mtq = ( -skew(B) * A_axes ) * u_mtq
        if mtqs:
            b_skew = -skewsym(b_body)
            A_mtq_axes = np.column_stack([np.asarray(m.axis, float).reshape(3,) for m in mtqs])
            A_mtq = b_skew @ A_mtq_axes
            u_mtq_lims = np.array([m.u_max for m in mtqs], dtype=float)
        else:
            A_mtq = np.zeros((3, 0))
            u_mtq_lims = np.zeros(0, dtype=float)

        A_total = np.hstack([A_rw, A_mtq])  # (3, n_act)
        n_act = A_total.shape[1]

        n_rw = len(rws)
        n_mtq = len(mtqs)

        if n_act == 0:
            return np.zeros(n_rw), np.zeros(n_mtq), 0.0

        # 2) Bounds: [-u_max, +u_max]
        lb = np.concatenate([-u_rw_lims, -u_mtq_lims]) if n_act else np.zeros(0)
        ub = np.concatenate([ u_rw_lims,  u_mtq_lims]) if n_act else np.zeros(0)

        # 3) Solve bounded least squares: minimize ||A u - tau_des||^2
        # Note: lsq_linear can handle rank-deficient A robustly.
        res = lsq_linear(A_total, tau_des, bounds=(lb, ub), method="trf")

        if not res.success:
            # fall back: best effort (still keep shape)
            return np.zeros(n_rw), np.zeros(n_mtq), 0.0

        u_sol = res.x  # length n_act

        # 4) Compute alpha as "how much of tau_des magnitude you achieved along its direction"
        tau_ach = A_total @ u_sol
        tau_hat = tau_des / (t_mag + 1e-12)
        T_along = float(np.dot(tau_ach, tau_hat))          # signed magnitude along desired direction
        alpha = max(0.0, T_along / (t_mag + 1e-12))        # normalize by requested magnitude, clamp to [0, inf)

        # Split commands
        u_rw_cmd = u_sol[:n_rw]
        u_mtq_cmd = u_sol[n_rw:n_rw + n_mtq]

        return u_rw_cmd, u_mtq_cmd, alpha
