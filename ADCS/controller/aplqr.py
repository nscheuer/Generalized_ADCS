__all__ = ["APLQR"]

import numpy as np
from scipy.linalg import solve_continuous_are

from ADCS.CONOPS.goals import Goal, No_Goal
from ADCS.controller import Controller
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.helpers.math_helpers import rot_mat, skewsym


class APLQR(Controller):
    r"""
    Asymptotic Periodic LQR (APLQR) controller for magnetorquer-only attitude pointing.

    Implements the algorithm from:

    M. L. Psiaki,
    Magnetic torquer attitude control via asymptotic periodic linear quadratic regulation,
    Journal of Guidance, Control, and Dynamics, Vol. 24, No. 2, 2001, pp. 386–394.

    The core idea is that the magnetorquer control effectiveness matrix

    .. math::

        B(t) =
        \begin{bmatrix}
        \mathbf{0} \\
        -J^{-1}[\boldsymbol{b}(t)]_{\times}
        \end{bmatrix}

    is periodic at the orbital rate. A standard time-invariant LQR is therefore
    not directly applicable. Psiaki's approach replaces :math:`B(t)` with its
    orbital average :math:`\tilde{B}`, computed analytically from the magnetic
    dipole field model over a circular orbit:

    .. math::

        \tilde{B} =
        \sqrt{\frac{1}{T} \int_0^T B(t) R^{-1} B(t)^\top \, dt}

    The steady-state solution :math:`P_{ss}` of the associated CARE,

    .. math::

        A^\top P + P A - P \tilde{B} R^{-1} \tilde{B}^\top P + Q = 0

    is computed offline. The control law then uses :math:`P_{ss}` with the
    instantaneous :math:`B(t)`:

    .. math::

        \boldsymbol{u}
        = -\alpha R^{-1} B(t)^\top P_{ss}
          \begin{bmatrix} \boldsymbol{e}_\theta \\ \boldsymbol{e}_\omega \end{bmatrix}

    where :math:`\boldsymbol{e}_\theta` and :math:`\boldsymbol{e}_\omega` are the
    attitude and angular-velocity errors with respect to the current mission goal.

    The linearized attitude dynamics matrix :math:`A` is built for Nadir pointing
    using the Euler–Hill equations in the LVLH frame, incorporating inertia asymmetry
    ratios :math:`\sigma_i`. For Inertial pointing, :math:`A = [0\; I;\; 0\; 0]`.
    An optional gravity-gradient perturbation term can be enabled.

    Actuator saturation is enforced by uniformly scaling the dipole vector when any
    component exceeds its limit:

    .. math::

        \beta = \max_i \frac{|u_i|}{u_{i,\max}}, \qquad
        \boldsymbol{u} \leftarrow \frac{\boldsymbol{u}}{\max(1, \beta)}

    Parameters
    ----------
    est_sat : EstimatedSatellite
        Satellite model providing inertia, actuators, and sensors.
    sma : float
        Orbital semi-major axis [m].
    incl : float
        Orbital inclination [rad].
    Q : array_like, shape (6, 6)
        Positive semi-definite state cost matrix for the CARE.
    R_ctrl : array_like, shape (3, 3)
        Positive definite, diagonal input cost matrix for the CARE and control law.
    alpha : float
        Scalar gain applied to the full control output.
    target_mode : str
        ``"Nadir"`` or ``"Inertial"``. Determines the linearization point.
    has_gg_torque : bool
        If ``True``, include gravity-gradient terms in the nadir linearization.

    Notes
    -----
    - Designed for MTQ-only satellites. Works with mixed MTQ/RW systems; the RW
      command slots in the output vector are always zero.
    - :math:`x_{\hat{}}` convention: ``[ω(3), q(4), ...]``, matching the rest
      of this framework.
    - ``os_hat.B`` is not used directly; the body-frame magnetic field is
      reconstructed from the onboard magnetometer via the sensor pseudo-inverse,
      consistent with all other controllers in this package.
    - The orbital average formula assumes a dipole magnetic field model and a
      circular orbit, and a diagonal inertia matrix. If the inertia tensor is
      not diagonal the formula is approximate.

    References
    ----------
    .. [1] M. L. Psiaki,
       Magnetic torquer attitude control via asymptotic periodic linear quadratic regulation,
       Journal of Guidance, Control, and Dynamics,
       Vol. 24, No. 2, 2001, pp. 386–394.

    """

    # Earth magnetic dipole moment [T·m³]
    _MU_MAG: float = 7.9e15
    # Earth gravitational parameter [m³/s²]
    _MU_EARTH: float = 3.9860044188e14

    def __init__(
        self,
        est_sat: EstimatedSatellite,
        sma: float,
        incl: float,
        Q: np.ndarray,
        R_ctrl: np.ndarray,
        alpha: float,
        target_mode: str = "Nadir",
        has_gg_torque: bool = False,
    ) -> None:
        r"""
        Initialize the APLQR controller and precompute the steady-state Riccati solution.

        :param est_sat: Estimated satellite model.
        :type est_sat: EstimatedSatellite
        :param sma: Semi-major axis [m].
        :type sma: float
        :param incl: Orbital inclination [rad].
        :type incl: float
        :param Q: State cost matrix (6×6).
        :type Q: array_like
        :param R_ctrl: Diagonal input cost matrix (3×3).
        :type R_ctrl: array_like
        :param alpha: Scalar control gain.
        :type alpha: float
        :param target_mode: ``"Nadir"`` or ``"Inertial"``.
        :type target_mode: str
        :param has_gg_torque: Include gravity-gradient linearization term.
        :type has_gg_torque: bool

        """
        self.J = est_sat.J_0
        self.J_inv = np.linalg.inv(self.J)
        self.sma = float(sma)
        self.incl = float(incl)
        self.n0 = np.sqrt(self._MU_EARTH / sma ** 3)
        self.alpha = float(alpha)
        self.target_mode = target_mode
        self.has_gg_torque = has_gg_torque

        self.R_ctrl = np.asarray(R_ctrl, float)
        self.R_inv = np.linalg.inv(self.R_ctrl)
        self.Q_mat = np.asarray(Q, float)

        # Offline: build linearized A, orbital-average B̃, solve CARE
        self.A = self._linearized_ss_matrix()
        B_tilde = self._orbital_average_B()
        self.P_ss = solve_continuous_are(self.A, B_tilde, self.Q_mat, np.eye(3))

        # Sensor reconstruction: body-frame magnetic field from magnetometers
        self.M_read, self.mtm_indices = self.build_sensor_matrix_pinv(
            sensors=est_sat.attitude_sensors + est_sat.rw_actuators,
            sensor_type=MTM,
        )

        # Per-axis MTQ saturation limits
        self.mtq_umax = np.array(
            [a.u_max for a in est_sat.actuators if isinstance(a, MTQ)], dtype=float
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def find_u(
        self,
        x_hat: np.ndarray,
        sens: np.ndarray,
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        goal: Goal | None = None,
    ) -> np.ndarray:
        r"""
        Compute the MTQ dipole moment command vector.

        :param x_hat: State vector ``[ω(3), q(4), ...]``.
        :type x_hat: numpy.ndarray
        :param sens: Raw sensor measurement vector. MTM channels are extracted
            using the precomputed pseudo-inverse mapping.
        :type sens: numpy.ndarray
        :param est_sat: Estimated satellite model.
        :type est_sat: EstimatedSatellite
        :param os_hat: Orbital state estimate. Provides ECI position/velocity
            for Nadir target computation.
        :type os_hat: Orbital_State
        :param goal: Mission goal. If ``None``, a ``No_Goal`` is assumed.
        :type goal: Goal or None
        :return: Full actuator command vector. MTQ slots are filled with the
            commanded dipole moments; all other slots are zero.
        :rtype: numpy.ndarray

        """
        if goal is None:
            goal = No_Goal()

        w = x_hat[0:3]
        q = x_hat[3:7]

        # Body-frame angular velocity reference from goal
        _, w_ref_eci = goal.to_ref(os0=os_hat)
        R_b2i = rot_mat(q)
        w_ref_body = R_b2i.T @ w_ref_eci
        w_err = w - w_ref_body

        # 3-vector attitude error from goal (e.g. vector part of error quaternion)
        boresight = est_sat.get_boresight(goal.boresight_name)
        e_att = goal.error(q=q, body_boresight=boresight, os0=os_hat)

        x_err = np.concatenate([e_att, w_err])

        # Body-frame magnetic field from magnetometer
        sens_clean = np.asarray(sens, float).copy()
        sens_clean[np.isnan(sens_clean)] = 0.0
        b_body = self.M_read @ sens_clean

        # Instantaneous control effectiveness: B_mat = [0(3×3); -J⁻¹·[b]×]  (6×3)
        B_mat = np.vstack([np.zeros((3, 3)), -self.J_inv @ skewsym(b_body)])

        # APLQR control law (Psiaki 2001):  u = -α R⁻¹ B(t)ᵀ P_ss x_err
        u_mtq_cmd = -self.alpha * self.R_inv @ B_mat.T @ self.P_ss @ x_err

        # Uniform saturation: scale down to keep all axes within limits
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(
                np.abs(u_mtq_cmd) > 0.0,
                self.mtq_umax / np.abs(u_mtq_cmd),
                np.inf,
            )
        scale = min(1.0, float(np.min(ratios)))
        u_mtq_cmd *= scale

        # Assemble full actuator command vector
        u_out = np.zeros(len(est_sat.actuators))
        mtq_indices = [i for i, a in enumerate(est_sat.actuators) if isinstance(a, MTQ)]
        u_out[mtq_indices] = u_mtq_cmd

        return u_out

    # ------------------------------------------------------------------
    # Offline precomputation
    # ------------------------------------------------------------------

    def _linearized_ss_matrix(self) -> np.ndarray:
        r"""
        Build the 6×6 linearized attitude dynamics matrix :math:`A`.

        For Nadir pointing, uses the Euler–Hill linearization in the LVLH frame:

        .. math::

            A =
            \begin{bmatrix}
            \mathbf{0} & I \\
            n_0^2 \operatorname{diag}(-\sigma_1, 0, \sigma_3)
            + [\text{gg}] &
            n_0
            \begin{bmatrix} 0 & 0 & 1-\sigma_1 \\ 0 & 0 & 0 \\ -(1+\sigma_3) & 0 & 0 \end{bmatrix}
            \end{bmatrix}

        with asymmetry ratios :math:`\sigma_i = (J_j - J_k)/J_i`.

        For Inertial pointing, :math:`A = [\mathbf{0}\ I;\ \mathbf{0}\ \mathbf{0}]`.

        :return: Linearized state matrix.
        :rtype: numpy.ndarray

        """
        A = np.zeros((6, 6))
        A[0:3, 3:6] = np.eye(3)

        if self.target_mode == "Nadir":
            J1 = self.J[0, 0]
            J2 = self.J[1, 1]
            J3 = self.J[2, 2]
            s1 = (J2 - J3) / J1
            s2 = (J3 - J1) / J2
            s3 = (J1 - J2) / J3
            n = self.n0
            A[3:6, 0:3] = n ** 2 * np.diag([-s1, 0.0, s3])
            A[3:6, 3:6] = n * np.array([
                [0.0,       0.0, 1.0 - s1],
                [0.0,       0.0, 0.0     ],
                [-(1.0+s3), 0.0, 0.0     ],
            ])
            if self.has_gg_torque:
                A[3:6, 0:3] += 3.0 * n ** 2 * np.diag([-s1, s2, 0.0])

        # Inertial: A remains [0 I; 0 0]
        return A

    def _orbital_average_B(self) -> np.ndarray:
        r"""
        Compute the orbitally-averaged control effectiveness matrix :math:`\tilde{B}`.

        Uses the closed-form dipole-field orbital average from Psiaki (2001), which
        assumes a circular orbit, a tilted-dipole magnetic field, and a diagonal
        inertia matrix. The diagonal entries of

        .. math::

            \frac{1}{T}\int_0^T B(t) R^{-1} B(t)^\top \, dt

        are computed analytically, yielding

        .. math::

            \tilde{B} =
            \begin{bmatrix}
            \mathbf{0} \\
            J^{-1} \operatorname{diag}\!\left(\sqrt{[\tilde{B}\tilde{B}^\top]_{ii}}\right)
            \end{bmatrix}

        :return: Averaged input matrix (6×3) for use in the CARE.
        :rtype: numpy.ndarray

        """
        mu = self._MU_MAG
        a = self.sma
        inc = self.incl
        J1 = self.J[0, 0]
        J2 = self.J[1, 1]
        J3 = self.J[2, 2]
        R1 = self.R_ctrl[0, 0]
        R2 = self.R_ctrl[1, 1]
        R3 = self.R_ctrl[2, 2]
        sin2 = np.sin(inc) ** 2

        # Diagonal of orbital-average B·R⁻¹·Bᵀ (Psiaki 2001, Eq. 20)
        BBd = np.array([
            mu**2 * (R2 - R2*sin2 + 2.0*R3*sin2)       / (J1**2 * R2 * R3 * a**6),
            mu**2 * sin2 * (R1 + 4.0*R3)                / (2.0 * J2**2 * R1 * R3 * a**6),
            mu**2 * (2.0*R2 + R1*sin2 - 2.0*R2*sin2)   / (2.0 * J3**2 * R1 * R2 * a**6),
        ])

        # B̃ = [0; J⁻¹ · diag(√BBd)]  — square root because CARE uses B̃·I⁻¹·B̃ᵀ = BBd
        B_tilde = np.vstack([
            np.zeros((3, 3)),
            self.J @ np.diag(np.sqrt(BBd)),
        ])
        return B_tilde
