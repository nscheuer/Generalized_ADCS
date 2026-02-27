__all__ = ["ThreeAxisMPC"]

import numpy as np
from scipy.linalg import expm, block_diag

try:
    import tinympc as _tinympc_module
    _TINYMPC_AVAILABLE = True
except ImportError:
    _TINYMPC_AVAILABLE = False

from ADCS.CONOPS.goals import Goal, No_Goal
from ADCS.controller import Controller
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.helpers.math_helpers import rot_mat, skewsym


class ThreeAxisMPC(Controller):
    r"""
    Three-axis magnetorquer attitude pointing MPC controller.

    Implements the algorithm from:

    M. McKeen (2024), Three-Axis Pointing TinyMPC for Nadir or Inertial Pointing,
    adapted from MATLAB ThreeAxisMPCControllerv1.

    Problem Formulation
    -------------------

    The linearized error-state model around the pointing target is

    .. math::

        \boldsymbol{x}_{k+1} = A_d \boldsymbol{x}_k + B_d \boldsymbol{\tau}_k,

    where
    :math:`\boldsymbol{x} = [\boldsymbol{e}_\theta;\, \boldsymbol{e}_\omega] \in \mathbb{R}^6`
    is the stacked attitude and angular-rate error with respect to the goal
    and :math:`\boldsymbol{\tau}_k \in \mathbb{R}^3` is the body torque in the
    **target frame** at step :math:`k`.

    Because magnetorquers cannot generate torque parallel to the geomagnetic
    field, the commanded torque at each horizon step must satisfy

    .. math::

        \hat{\boldsymbol{B}}_k^\top \boldsymbol{\tau}_k = 0, \qquad k = 0, \ldots, N-2,

    where :math:`\hat{\boldsymbol{B}}_k` is the unit field vector rotated into
    the target frame at step :math:`k`.

    This constraint is incorporated either by

    * **``"closedform"``** — exact null-space projection (always available):

      .. math::

          \boldsymbol{u}^* = -\Lambda\!\left(I - \Psi^\top(\Psi\Lambda\Psi^\top)^{-1}\Psi\Lambda\right)
          \Gamma^\top \bar{Q}\, Y\, \boldsymbol{x}_0,

      where :math:`\Lambda = (\Gamma^\top\bar{Q}\Gamma + \bar{R})^{-1}` is
      precomputed offline, and :math:`\Psi` (block-diagonal of
      :math:`\hat{\boldsymbol{B}}_k^\top`) is rebuilt each control step.

    * **``"tinympc"``** — TinyMPC ADMM solver (:pypi:`tinympc`, Linux/macOS only).
      The B-field constraint is passed as an equality constraint before each
      ``solve()`` call.  The system matrices are pre-scaled for numerical
      conditioning.

    Both backends produce mathematically identical results for the unconstrained
    problem; with the B-field constraint the closed-form solution is exact while
    TinyMPC approximates it as a time-invariant equality using the step-0 field
    direction.

    Horizon Prediction
    ------------------

    At each call to :meth:`find_u` the orbital state is propagated forward
    :math:`N-1` steps via :meth:`~ADCS.orbits.orbital_state.Orbital_State.propagate_orbit_rk4`
    to obtain predicted ECI magnetic-field vectors.  For Nadir pointing the
    LVLH frame is recomputed at each predicted position; for Inertial pointing
    the target frame is fixed to ECI.

    Parameters
    ----------
    est_sat : EstimatedSatellite
        Satellite model providing inertia, actuators, and sensors.
    sma : float
        Orbital semi-major axis [m].
    incl : float
        Orbital inclination [rad].
    N : int
        Prediction horizon length (number of time steps).  There are ``N-1``
        control inputs in the QP.
    dt : float
        Control time step [s].
    Q : array_like, shape (6, 6)
        Positive semi-definite stage cost on the state error.
    Q_N : array_like, shape (6, 6)
        Positive semi-definite terminal cost on the state error.
    R : array_like, shape (3, 3)
        Positive definite cost on the torque command.
    target_mode : str
        ``"Nadir"`` or ``"Inertial"``.
    has_gg_torque : bool
        Include gravity-gradient terms in the Nadir linearization.
    solver : str
        ``"closedform"`` (default) or ``"tinympc"``.
    rho : float
        ADMM penalty parameter for TinyMPC (ignored for closed-form).
    max_iter : int
        Maximum ADMM iterations for TinyMPC.
    abs_tol : float
        Absolute primal/dual tolerance for TinyMPC.

    Notes
    -----
    - The ``"tinympc"`` solver is only available on Linux/macOS.  On Windows
      use WSL, or select ``"closedform"``.
    - Q and R should ideally be diagonal for TinyMPC to perform optimally.
    - The output dipole saturation uses uniform scaling (preserves direction).

    References
    ----------
    .. [1] M. L. Psiaki, "Magnetic Torquer Attitude Control via Asymptotic
       Periodic LQR," JGCD, 2001.
    .. [2] A. Agrawal, K. Kim, et al., "TinyMPC: Model-Predictive Control on
       Resource-Constrained Microcontrollers," ICRA 2024.

    """

    _MU_EARTH: float = 3.9860044188e14

    def __init__(
        self,
        est_sat: EstimatedSatellite,
        sma: float,
        incl: float,
        N: int,
        dt: float,
        Q: np.ndarray,
        Q_N: np.ndarray,
        R: np.ndarray,
        target_mode: str = "Nadir",
        has_gg_torque: bool = False,
        solver: str = "closedform",
        rho: float = 1.0,
        max_iter: int = 1000,
        abs_tol: float = 1e-5,
    ) -> None:
        r"""
        Build the MPC controller and precompute all offline matrices.

        :param est_sat: Estimated satellite model.
        :param sma: Semi-major axis [m].
        :param incl: Orbital inclination [rad].
        :param N: Prediction horizon length (>= 2).
        :param dt: Control time step [s].
        :param Q: Stage state cost (6×6).
        :param Q_N: Terminal state cost (6×6).
        :param R: Torque cost (3×3).
        :param target_mode: ``"Nadir"`` or ``"Inertial"``.
        :param has_gg_torque: Include gravity-gradient linearization term.
        :param solver: ``"closedform"`` or ``"tinympc"``.
        :param rho: TinyMPC ADMM penalty.
        :param max_iter: TinyMPC maximum iterations.
        :param abs_tol: TinyMPC convergence tolerance.
        """
        self.J = est_sat.J_0
        self.J_inv = np.linalg.inv(self.J)
        self.sma = float(sma)
        self.incl = float(incl)
        self.n0 = np.sqrt(self._MU_EARTH / sma ** 3)
        self.N = int(N)
        self.dt = float(dt)
        self.target_mode = target_mode
        self.has_gg_torque = has_gg_torque
        self.solver = solver

        self.Q = np.asarray(Q, float)
        self.Q_N = np.asarray(Q_N, float)
        self.R = np.asarray(R, float)

        # ------------------------------------------------------------------
        # Discretize linearized dynamics via matrix exponential
        # ------------------------------------------------------------------
        Ac = self._linearized_ss_matrix()
        Bc = np.vstack([np.zeros((3, 3)), self.J_inv])  # [0; J⁻¹] torque → ω̇

        # Zero-order hold: [Ad Bd; 0 I] = expm([Ac Bc; 0 0] * dt)
        n, m = 6, 3
        AB = np.block([[Ac, Bc], [np.zeros((m, n + m))]])
        eABd = expm(AB * self.dt)
        self.Ad = eABd[:n, :n]
        self.Bd = eABd[:n, n:n + m]

        # ------------------------------------------------------------------
        # Variable scaling for TinyMPC numerical conditioning
        # (matches MATLAB: xS, uS based on representative angle/rate/torque)
        # ------------------------------------------------------------------
        thetaref = 30.0 * np.pi / 180.0          # 30 deg representative angle
        omegaref = self.n0 / 2.0                  # half orbital rate
        self.mtq_umax = np.array(
            [a.u_max for a in est_sat.actuators if isinstance(a, MTQ)], dtype=float
        )
        taumax = float(np.min(self.mtq_umax)) * 30e-6   # dipole_max × ~30 µT

        self.xS = np.diag([thetaref] * 3 + [omegaref] * 3)
        self.uS = np.diag([taumax] * 3)
        self.xS_inv = np.diag([1.0 / thetaref] * 3 + [1.0 / omegaref] * 3)
        self.uS_inv = np.diag([1.0 / taumax] * 3)

        # Scaled system (for TinyMPC)
        self.Ad_hat = self.xS_inv @ self.Ad @ self.xS
        self.Bd_hat = self.xS_inv @ self.Bd @ self.uS
        self.Q_hat  = self.xS.T  @ self.Q   @ self.xS
        self.QN_hat = self.xS.T  @ self.Q_N @ self.xS
        self.R_hat  = self.uS.T  @ self.R   @ self.uS

        # ------------------------------------------------------------------
        # Precompute closed-form matrices (unscaled: Ad, Bd, Q, Q_N, R)
        # ------------------------------------------------------------------
        self._precompute_closedform()

        # ------------------------------------------------------------------
        # Sensor and actuator mappings
        # ------------------------------------------------------------------
        self.M_read, self.mtm_indices = self.build_sensor_matrix_pinv(
            sensors=est_sat.attitude_sensors + est_sat.rw_actuators,
            sensor_type=MTM,
        )

        # ------------------------------------------------------------------
        # TinyMPC solver setup (if requested)
        # ------------------------------------------------------------------
        self._tinympc = None
        if solver == "tinympc":
            self._setup_tinympc(rho=rho, max_iter=max_iter, abs_tol=abs_tol)

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
        :param sens: Raw sensor measurement vector.
        :param est_sat: Estimated satellite model.
        :param os_hat: Current orbital state (used to propagate the horizon).
        :param goal: Mission goal.  ``None`` defaults to :class:`No_Goal`.
        :return: Full actuator command vector with MTQ dipole commands filled.
        :rtype: numpy.ndarray
        """
        if goal is None:
            goal = No_Goal()

        w = x_hat[0:3]
        q = x_hat[3:7]

        # Angular rate error
        _, w_ref_eci = goal.to_ref(os0=os_hat)
        R_b2i = rot_mat(q)
        w_ref_body = R_b2i.T @ w_ref_eci
        w_err = w - w_ref_body

        # Attitude error (3-vector)
        boresight = est_sat.get_boresight(goal.boresight_name)
        e_att = goal.error(q=q, body_boresight=boresight, os0=os_hat)

        # Early exit when already on target (deadzone matching MATLAB code)
        if np.linalg.norm(w_err) < 1e-5 and np.linalg.norm(e_att) < 1e-2:
            return np.zeros(len(est_sat.actuators))

        x_err = np.concatenate([e_att, w_err])

        # Propagate orbit horizon → B vectors in target frame
        B_tgt_list, R_tgt2eci_0 = self._predict_horizon(os_hat)

        # Solve MPC
        if self.solver == "tinympc" and self._tinympc is not None:
            tau_tgt = self._solve_tinympc(x_err, B_tgt_list)
        else:
            tau_tgt = self._solve_closedform(x_err, B_tgt_list)

        # Convert: torque in target frame → dipole in target frame → ECI → body
        B_tgt0 = B_tgt_list[0]
        B_norm2 = float(np.dot(B_tgt0, B_tgt0))
        if B_norm2 < 1e-30:
            return np.zeros(len(est_sat.actuators))

        mu_tgt   = np.cross(B_tgt0, tau_tgt) / B_norm2   # dipole in target frame
        mu_eci   = R_tgt2eci_0 @ mu_tgt                   # rotate to ECI
        mu_body  = R_b2i.T @ mu_eci                        # rotate to body

        # Uniform saturation (preserve dipole direction)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(
                np.abs(mu_body) > 0.0,
                self.mtq_umax / np.abs(mu_body),
                np.inf,
            )
        mu_body *= min(1.0, float(np.min(ratios)))

        u_out = np.zeros(len(est_sat.actuators))
        mtq_indices = [i for i, a in enumerate(est_sat.actuators) if isinstance(a, MTQ)]
        u_out[mtq_indices] = mu_body
        return u_out

    # ------------------------------------------------------------------
    # Horizon propagation
    # ------------------------------------------------------------------

    def _predict_horizon(
        self, os_hat: Orbital_State
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Propagate the orbital state ``N-1`` steps forward and return
        the geomagnetic field expressed in the target frame at each step.

        :return: ``(B_tgt_list, R_tgt2eci_0)``
            - ``B_tgt_list``: list of N-1 vectors, each shape (3,) [T].
            - ``R_tgt2eci_0``: (3,3) rotation from target to ECI at step 0.
        """
        N1 = self.N - 1
        B_tgt_list: list[np.ndarray] = []
        R_tgt2eci_0: np.ndarray | None = None

        os_curr = os_hat
        for i in range(N1):
            R_tgt2eci_i = self._target_rotation(os_curr)
            if i == 0:
                R_tgt2eci_0 = R_tgt2eci_i

            B_eci_i = np.asarray(os_curr.B, float).reshape(3,)
            B_tgt_list.append(R_tgt2eci_i.T @ B_eci_i)

            # Propagate (J2 off for speed; re-enable if accuracy needed)
            os_curr = os_curr.propagate_orbit_rk4(self.dt, J2_perturbation_on=False)

        return B_tgt_list, R_tgt2eci_0  # type: ignore[return-value]

    def _target_rotation(self, os: Orbital_State) -> np.ndarray:
        """
        Return the rotation matrix :math:`R_{\\text{tgt}\\to\\text{ECI}}` at
        the given orbital state.

        For Nadir pointing this is the LVLH frame built from the position and
        velocity vectors.  For Inertial pointing it is the identity.

        :return: (3,3) rotation matrix, columns = target axes in ECI.
        """
        if self.target_mode == "Inertial":
            return np.eye(3)

        # Nadir: build LVLH frame (matches MATLAB GetNadirAttitude)
        r = np.asarray(os.R, float).reshape(3,)
        v = np.asarray(os.V, float).reshape(3,)
        z = -r / np.linalg.norm(r)          # nadir (down)
        y = np.cross(z, v)
        y_n = np.linalg.norm(y)
        if y_n < 1e-12:
            return np.eye(3)
        y = y / y_n                          # negative orbit normal
        x = np.cross(y, z)
        x = x / np.linalg.norm(x)           # ~ velocity
        return np.column_stack([x, y, z])    # LVLH → ECI

    # ------------------------------------------------------------------
    # MPC solvers
    # ------------------------------------------------------------------

    def _solve_closedform(
        self, x_err: np.ndarray, B_tgt_list: list[np.ndarray]
    ) -> np.ndarray:
        r"""
        Exact null-space projection closed-form MPC solution.

        Builds :math:`\Psi` (block-diagonal of :math:`\hat{B}_k^\top`)
        then solves

        .. math::

            \boldsymbol{u}^* =
            -\Lambda\!\left(I - \Psi^\top(\Psi\Lambda\Psi^\top)^{-1}\Psi\Lambda\right)
            \Gamma^\top \bar{Q}\, Y\, \boldsymbol{x}_0

        :param x_err: Current error state (6,).
        :param B_tgt_list: List of N-1 B-field vectors in target frame.
        :return: First step optimal torque (3,) in target frame [N·m].
        """
        N1 = self.N - 1
        nu = 3

        # Build Psi: (N1) × (nu*N1), block-diagonal of B̂_k^T
        Psi = np.zeros((N1, nu * N1))
        valid = np.zeros(N1, dtype=bool)
        for i, B_tgt in enumerate(B_tgt_list):
            B_n = np.linalg.norm(B_tgt)
            if B_n > 1e-30:
                Psi[i, nu * i : nu * (i + 1)] = B_tgt / B_n
                valid[i] = True

        # Reduce Psi to only valid rows (avoids rank-deficiency from zero-B steps)
        Psi_r = Psi[valid, :]

        # Base unconstrained gradient term: g = Γ' Q̄ Y x_err
        g = self._cf_Gamma.T @ (self._cf_bigQ @ (self._cf_Y @ x_err))

        # Unconstrained optimal: u_unc = -Λ g
        u_unc = -self._cf_Lambda @ g

        if Psi_r.shape[0] == 0:
            # No valid B field → skip constraint
            return u_unc[:nu]

        # Projected onto null space of Psi_r: u* = u_unc - Λ Ψ'(ΨΛΨ')^{-1} Ψ u_unc
        PsiLam = Psi_r @ self._cf_Lambda                    # (nr, nu*N1)
        PsiLamPsiT = PsiLam @ Psi_r.T                       # (nr, nr)
        try:
            correction = Psi_r.T @ np.linalg.solve(PsiLamPsiT, Psi_r @ u_unc)
            u_opt = u_unc - self._cf_Lambda @ correction
        except np.linalg.LinAlgError:
            u_opt = u_unc

        return u_opt[:nu]

    def _solve_tinympc(
        self, x_err: np.ndarray, B_tgt_list: list[np.ndarray]
    ) -> np.ndarray:
        r"""
        TinyMPC ADMM solver backend.

        The state is scaled before solving and the output is unscaled.
        The B-field equality constraint is set using the step-0 field
        direction as a time-invariant approximation for the full horizon.

        :param x_err: Current error state (6,).
        :param B_tgt_list: List of N-1 B-field vectors in target frame.
        :return: First step optimal torque (3,) in target frame [N·m].
        """
        # Scale initial state
        zx0 = self.xS_inv @ x_err
        self._tinympc.set_x0(zx0)
        self._tinympc.set_x_ref(np.zeros(6))
        self._tinympc.set_u_ref(np.zeros(3))

        # Set B-field equality constraint from step-0 field (time-invariant approx)
        B_tgt0 = B_tgt_list[0]
        B_norm = np.linalg.norm(B_tgt0)
        if B_norm > 1e-30:
            # In scaled space: (B̂ · uS) · z_u = 0
            B_hat_scaled = (B_tgt0 / B_norm) @ self.uS   # shape (3,)
            Aeq_u = B_hat_scaled.reshape(1, 3)
            beq_u = np.zeros(1)
            try:
                self._tinympc.set_equality_constraints(
                    Aeq_x=np.zeros((1, 6)),
                    beq_x=np.zeros(1),
                    Aeq_u=Aeq_u,
                    beq_u=beq_u,
                )
            except Exception:
                pass  # solver may not support mid-run constraint updates; proceed anyway

        solution = self._tinympc.solve()

        # Extract first control step and unscale
        controls = solution.get("controls", None)
        if controls is None or controls.shape[1] == 0:
            # Fallback to closed-form if TinyMPC returns empty solution
            return self._solve_closedform(x_err, B_tgt_list)

        zu0 = controls[:, 0]            # scaled torque (3,)
        return self.uS @ zu0            # unscale → [N·m]

    # ------------------------------------------------------------------
    # Offline precomputation (closed-form)
    # ------------------------------------------------------------------

    def _precompute_closedform(self) -> None:
        r"""
        Precompute the constant matrices for the closed-form solution.

        Computes (unscaled):

        * :math:`\Gamma \in \mathbb{R}^{6(N-1)\times 3(N-1)}` — control
          response matrix (:math:`\Gamma_{ij} = A_d^{i-j} B_d` for :math:`j\le i`).
        * :math:`Y \in \mathbb{R}^{6(N-1)\times 6}` — free-response matrix
          (:math:`Y_i = A_d^{i+1}`).
        * :math:`\bar{Q}` — block-diagonal stage + terminal cost.
        * :math:`\Lambda = (\Gamma^\top \bar{Q}\Gamma + \bar{R})^{-1}`.
        """
        N1 = self.N - 1
        nx, nu = 6, 3

        # Gamma: (nx*N1) × (nu*N1)
        Gamma = np.zeros((nx * N1, nu * N1))
        Ad_pows = [None] * N1
        Apow = np.eye(nx)
        Ad_pows[0] = Apow
        for k in range(1, N1):
            Apow = Apow @ self.Ad
            Ad_pows[k] = Apow

        for i in range(N1):
            for j in range(i + 1):
                Gamma[nx * i : nx * (i + 1), nu * j : nu * (j + 1)] = (
                    Ad_pows[i - j] @ self.Bd
                )

        # Y: (nx*N1) × nx
        Y = np.zeros((nx * N1, nx))
        Apow = np.eye(nx)
        for i in range(N1):
            Apow = Apow @ self.Ad
            Y[nx * i : nx * (i + 1), :] = Apow

        # bigQ: block-diagonal [Q, ..., Q, Q_N]
        Q_blocks = [self.Q] * (N1 - 1) + [self.Q_N]
        bigQ = block_diag(*Q_blocks)

        # bigR: block-diagonal [R, ..., R]
        bigR = block_diag(*([self.R] * N1))

        # Lambda
        GtQ = Gamma.T @ bigQ
        Lambda = np.linalg.inv(GtQ @ Gamma + bigR)

        self._cf_Gamma  = Gamma
        self._cf_Y      = Y
        self._cf_bigQ   = bigQ
        self._cf_Lambda = Lambda

    def _setup_tinympc(self, rho: float, max_iter: int, abs_tol: float) -> None:
        """
        Initialise the TinyMPC solver with the scaled system matrices.

        :raises ImportError: If the ``tinympc`` package is not installed.
        :raises RuntimeError: If TinyMPC setup fails.
        """
        if not _TINYMPC_AVAILABLE:
            raise ImportError(
                "tinympc is not installed.  Install with:\n"
                "    pip install tinympc\n"
                "(Note: currently only available on Linux/macOS — use WSL on Windows.)"
            )

        prob = _tinympc_module.TinyMPC()
        prob.setup(
            self.Ad_hat,
            self.Bd_hat,
            self.Q_hat,
            self.R_hat,
            self.N,
            rho=rho,
            verbose=False,
        )
        prob.update_settings(
            max_iter=max_iter,
            abs_pri_tol=abs_tol,
            abs_dua_tol=abs_tol,
        )
        self._tinympc = prob

    # ------------------------------------------------------------------
    # Linearized dynamics (shared with APLQR)
    # ------------------------------------------------------------------

    def _linearized_ss_matrix(self) -> np.ndarray:
        r"""
        Build the 6×6 continuous-time linearized attitude dynamics matrix.

        Identical to the APLQR linearization — Euler–Hill equations for Nadir,
        :math:`A = [\mathbf{0}\ I;\ \mathbf{0}\ \mathbf{0}]` for Inertial.

        :return: Linearized state matrix (6×6).
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
                [0.0,        0.0, 1.0 - s1],
                [0.0,        0.0, 0.0     ],
                [-(1.0 + s3), 0.0, 0.0    ],
            ])
            if self.has_gg_torque:
                A[3:6, 0:3] += 3.0 * n ** 2 * np.diag([-s1, s2, 0.0])

        return A
