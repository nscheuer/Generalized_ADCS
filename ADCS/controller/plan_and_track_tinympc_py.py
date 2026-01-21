"""
Plan and Track TinyMPC Controller (Pure Python Implementation).

Implements ADMM-based MPC for trajectory tracking using only numpy/scipy.
Useful for debugging, prototyping, and platforms without C++ bindings.

Mathematical Formulation
------------------------
TinyMPC solves the following tracking MPC problem at each control cycle:

    minimize    J = sum_{k=0}^{N-1} [(x_k - x_ref_k)' Q (x_k - x_ref_k)
                                   + (u_k - u_ref_k)' R (u_k - u_ref_k)]
                  + (x_N - x_ref_N)' Qf (x_N - x_ref_N)

    subject to  x_{k+1} = A x_k + B u_k + c     (linearized dynamics)
                x_0 = x_current                  (initial condition)
                u_min <= u_k <= u_max            (actuator bounds)

where:
    - N: MPC prediction horizon (5-10 steps)
    - Q, R, Qf: Cost matrices for state error, control deviation, terminal cost
    - A, B, c: Linearized dynamics about reference trajectory
    - x_ref, u_ref: Reference trajectory from ALTRO planner

The ADMM algorithm solves this by splitting u = z where z satisfies bounds:
    1. x-update: Solve unconstrained LQR with ADMM penalty term
    2. z-update: Project controls onto bounds: z = clip(u + y, u_min, u_max)
    3. y-update: Dual variable update: y = y + u - z
"""
from __future__ import annotations

__all__ = ["Plan_and_Track_TinyMPC_Py", "TinyMPCSolverPy"]

import time
import numpy as np
from typing import Optional, Tuple, NamedTuple
from numpy.typing import NDArray
from scipy.linalg import solve

from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_base import PlanAndTrackBase
from ADCS.controller.helpers import PlannerSettings, Trajectory
from ADCS.controller.helpers.tinympc_settings import TinyMPCSettings
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.helpers.math_helpers import quat_mult, quat_inv, quat_to_vec3, quat_diff, normalize


class TinyMPCResult(NamedTuple):
    """Result from TinyMPC solve."""
    u_opt: NDArray[np.float64]          # Optimal control to apply
    X_pred: NDArray[np.float64]         # Predicted state trajectory (n, N+1)
    U_pred: NDArray[np.float64]         # Predicted control trajectory (m, N)
    iterations: int                      # Number of ADMM iterations
    solve_time_ms: float                # Solve time in milliseconds
    converged: bool                      # Whether solver converged
    tracking_error: float               # State tracking error norm


class TinyMPCSolverPy:
    """
    Pure Python ADMM-based MPC solver for trajectory tracking.

    This solver implements the TinyMPC algorithm using numpy, providing
    a portable implementation for debugging and platforms without C++ bindings.

    The ADMM (Alternating Direction Method of Multipliers) algorithm efficiently
    handles the box constraints on controls by splitting the problem:
        - Solve unconstrained LQR with an augmented cost term
        - Project controls onto feasible set (simple clipping)
        - Update dual variables

    Attributes:
        n: State dimension
        m: Control dimension
        settings: TinyMPC solver configuration
        Q, R, Qf: Cost matrices for tracking
        u_min, u_max: Control bounds from actuators
    """

    def __init__(
        self,
        n: int,
        m: int,
        settings: TinyMPCSettings,
        u_min: NDArray[np.float64],
        u_max: NDArray[np.float64]
    ):
        """
        Initialize TinyMPC solver.

        Args:
            n: State dimension (7 + n_rw for spacecraft)
            m: Control dimension (n_mtq + n_rw)
            settings: Solver configuration
            u_min: Lower control bounds (m,)
            u_max: Upper control bounds (m,)
        """
        self.n = n
        self.m = m
        # Error state dimension: 3 (ang vel) + 3 (attitude) + n_rw = n - 1
        # This matches TVLQR's reduced quaternion representation
        self.n_err = n - 1  # Reduced dimension for quaternion linearization
        self.settings = settings
        self.u_min = np.asarray(u_min, dtype=np.float64)
        self.u_max = np.asarray(u_max, dtype=np.float64)

        N = settings.track_horizon

        # Cost matrices for ERROR state (set via set_cost_matrices)
        # These are (n_err x n_err) to match the reduced error state
        self.Q = np.eye(self.n_err)
        self.R = np.eye(m)
        self.Qf = np.eye(self.n_err)

        # ADMM variables
        self.Z = np.zeros((m, N))       # Projected controls
        self.Y = np.zeros((m, N))       # Dual variables
        self._Z_prev = np.zeros((m, N))  # For dual residual

        # Warm start storage
        self.X_warm: Optional[NDArray] = None
        self.U_warm: Optional[NDArray] = None
        self.has_warm_start = False

        # Reference trajectory storage
        self.X_ref: Optional[NDArray] = None   # (n, N_ref+1)
        self.U_ref: Optional[NDArray] = None   # (m, N_ref)
        self.K_ref: Optional[NDArray] = None   # Optional LQR gains
        self.times_ref: Optional[NDArray] = None
        self.dt_ref = 1.0
        self.has_reference = False

        # Linearized ERROR dynamics (updated each solve)
        # A_err: (n_err x n_err), B_err: (n_err x m)
        self.A_err = np.eye(self.n_err)
        self.B_err = np.zeros((self.n_err, m))
        self.c_err = np.zeros(self.n_err)

        # Full state dynamics for propagation
        self.A = np.eye(n)
        self.B = np.zeros((n, m))
        self.c = np.zeros(n)

        # Current rho (may be adapted)
        self._rho = settings.rho

    def set_cost_matrices(
        self,
        Q: NDArray[np.float64],
        R: NDArray[np.float64],
        Qf: NDArray[np.float64]
    ) -> None:
        """
        Set the cost matrices for tracking.

        Args:
            Q: State tracking error cost (n, n)
            R: Control deviation cost (m, m)
            Qf: Terminal tracking error cost (n, n)
        """
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.Qf = np.asarray(Qf, dtype=np.float64)

    def load_reference(
        self,
        X_ref: NDArray[np.float64],
        U_ref: NDArray[np.float64],
        K_ref: Optional[NDArray[np.float64]],
        times: NDArray[np.float64],
        dt: float
    ) -> None:
        """
        Load reference trajectory from ALTRO output.

        Args:
            X_ref: Reference states, shape (n, N+1) or (N+1, n)
            U_ref: Reference controls, shape (m, N) or (N, m)
            K_ref: Optional LQR gains from ALTRO
            times: Time stamps for reference points
            dt: Reference trajectory timestep
        """
        # Normalize to (n, N+1) layout
        X_ref = np.asarray(X_ref, dtype=np.float64)
        if X_ref.shape[0] != self.n:
            X_ref = X_ref.T
        self.X_ref = X_ref.copy()

        # Normalize to (m, N) layout
        U_ref = np.asarray(U_ref, dtype=np.float64)
        if U_ref.shape[0] != self.m:
            U_ref = U_ref.T
        self.U_ref = U_ref.copy()

        self.K_ref = K_ref.copy() if K_ref is not None else None
        self.times_ref = np.asarray(times, dtype=np.float64).flatten()
        self.dt_ref = float(dt)
        self.has_reference = True

        # Reset ADMM state for new trajectory
        self.reset()

    def interpolate_reference(self, t: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Interpolate reference state and control at time t.

        Uses linear interpolation for most states and SLERP-like
        interpolation for quaternions.

        Args:
            t: Time to interpolate at

        Returns:
            (x_ref, u_ref): Interpolated reference state and control
        """
        if not self.has_reference:
            return np.zeros(self.n), np.zeros(self.m)

        t_start = self.times_ref[0]
        t_end = self.times_ref[-1]
        t = np.clip(t, t_start, t_end)

        # Find interpolation index
        t_rel = t - t_start
        idx_float = t_rel / self.dt_ref
        idx = int(np.floor(idx_float))
        alpha = idx_float - idx

        N_ref = self.X_ref.shape[1] - 1
        idx = np.clip(idx, 0, N_ref - 1)

        # Interpolate state
        x0 = self.X_ref[:, idx]
        x1 = self.X_ref[:, min(idx + 1, N_ref)]

        x_interp = (1 - alpha) * x0 + alpha * x1

        # SLERP-like interpolation for quaternion (indices 3:7)
        q0 = x0[3:7]
        q1 = x1[3:7]
        # Ensure same hemisphere
        if np.dot(q0, q1) < 0:
            q1 = -q1
        q_interp = (1 - alpha) * q0 + alpha * q1
        q_norm = np.linalg.norm(q_interp)
        if q_norm > 1e-9:
            q_interp /= q_norm
        x_interp[3:7] = q_interp

        # Interpolate control
        N_u = self.U_ref.shape[1]
        idx_u = min(idx, N_u - 1)
        if alpha < 1e-10 or idx_u >= N_u - 1:
            u_interp = self.U_ref[:, idx_u]
        else:
            u_interp = (1 - alpha) * self.U_ref[:, idx_u] + alpha * self.U_ref[:, min(idx_u + 1, N_u - 1)]

        return x_interp, u_interp

    def build_local_reference(
        self,
        t_start: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Build local reference trajectory for MPC horizon.

        Args:
            t_start: Start time for local horizon

        Returns:
            (X_ref_local, U_ref_local): Local reference trajectory
                X_ref_local shape (n, N+1)
                U_ref_local shape (m, N)
        """
        N = self.settings.track_horizon
        dt = self.settings.track_dt

        X_ref_local = np.zeros((self.n, N + 1))
        U_ref_local = np.zeros((self.m, N))

        for k in range(N + 1):
            t = t_start + k * dt
            x_ref, u_ref = self.interpolate_reference(t)
            X_ref_local[:, k] = x_ref
            if k < N:
                U_ref_local[:, k] = u_ref

        return X_ref_local, U_ref_local

    def linearize_dynamics(
        self,
        x_op: NDArray[np.float64],
        u_op: NDArray[np.float64],
        est_sat: EstimatedSatellite,
        os: Orbital_State
    ) -> None:
        """
        Linearize dynamics about operating point using finite differences.

        Computes both:
        1. Full state discretized dynamics: x_{k+1} = A*x_k + B*u_k + c
        2. Error state dynamics: dx_{k+1} = A_err*dx_k + B_err*du_k

        The error dynamics are used for Riccati/ADMM in the reduced (n_err) space.

        Args:
            x_op: Operating point state
            u_op: Operating point control
            est_sat: Satellite model for dynamics evaluation
            os: Orbital state for environment
        """
        dt = self.settings.track_dt
        eps = 1e-7

        # Get nominal dynamics
        xdot_nom = est_sat.dynamics_core(x_op, u_op, os)

        # Compute A = df/dx via finite differences (full state)
        A_cont = np.zeros((self.n, self.n))
        for i in range(self.n):
            x_pert = x_op.copy()
            x_pert[i] += eps
            # Renormalize quaternion if perturbed
            if 3 <= i < 7:
                x_pert[3:7] = normalize(x_pert[3:7])
            xdot_pert = est_sat.dynamics_core(x_pert, u_op, os)
            A_cont[:, i] = (xdot_pert - xdot_nom) / eps

        # Compute B = df/du via finite differences (full state)
        B_cont = np.zeros((self.n, self.m))
        for i in range(self.m):
            u_pert = u_op.copy()
            u_pert[i] += eps
            xdot_pert = est_sat.dynamics_core(x_op, u_pert, os)
            B_cont[:, i] = (xdot_pert - xdot_nom) / eps

        # Discretize full state dynamics using forward Euler
        self.A = np.eye(self.n) + dt * A_cont
        self.B = dt * B_cont
        self.c = dt * (xdot_nom - A_cont @ x_op - B_cont @ u_op)

        # Compute error state dynamics by mapping full state Jacobians
        # Error state: [w_err(3), theta_err(3), h_err(n_rw)]
        # Full state: [w(3), q(4), h(n_rw)]
        #
        # The mapping from full state perturbation to error state:
        # dw -> dw (direct)
        # dq -> d_theta = 2*vec(q_err) ≈ 2*dq[1:4] for small angles
        # dh -> dh (direct)
        #
        # So we extract the relevant rows/cols from A_cont and B_cont
        n_rw = self.n - 7

        # Build projection matrix from full state (n) to error state (n_err)
        # This extracts the relevant Jacobian blocks
        # A_err = G * A_cont * G^T where G projects full to error
        #
        # For simplicity, directly extract blocks:
        # A_err[0:3, 0:3] = A_cont[0:3, 0:3]  (w wrt w)
        # A_err[0:3, 3:6] ≈ A_cont[0:3, 4:7] * 0.5  (w wrt theta, via dq[1:4])
        # A_err[3:6, 0:3] = A_cont[4:7, 0:3] * 2  (d_theta wrt w)
        # A_err[3:6, 3:6] ≈ A_cont[4:7, 4:7]  (d_theta wrt d_theta)
        # etc.

        # Simpler approach: Use the small-angle approximation that the error
        # dynamics have the same structure, just with 3D attitude instead of 4D quat
        A_err_cont = np.zeros((self.n_err, self.n_err))
        B_err_cont = np.zeros((self.n_err, self.m))

        # Angular velocity block (0:3, 0:3)
        A_err_cont[0:3, 0:3] = A_cont[0:3, 0:3]

        # Angular velocity wrt attitude (0:3, 3:6) - from quaternion vector part
        # The quaternion derivative dq/dt has vector part related to angular velocity
        # For small angles: d(theta)/dt ≈ w, so d(w)/d(theta) comes from attitude-dependent torques
        A_err_cont[0:3, 3:6] = A_cont[0:3, 4:7] * 0.5  # Scale by 0.5 for quat->angle

        # Attitude rate wrt angular velocity (3:6, 0:3)
        # d(theta)/dt ≈ w for small angles, so d(d_theta)/d(w) ≈ I
        A_err_cont[3:6, 0:3] = A_cont[4:7, 0:3] * 2.0  # Scale by 2 for quat->angle

        # Attitude wrt attitude (3:6, 3:6)
        A_err_cont[3:6, 3:6] = A_cont[4:7, 4:7]

        # RW momentum blocks
        if n_rw > 0:
            # h wrt w
            A_err_cont[6:6+n_rw, 0:3] = A_cont[7:7+n_rw, 0:3]
            # h wrt attitude
            A_err_cont[6:6+n_rw, 3:6] = A_cont[7:7+n_rw, 4:7] * 0.5
            # w wrt h
            A_err_cont[0:3, 6:6+n_rw] = A_cont[0:3, 7:7+n_rw]
            # attitude wrt h
            A_err_cont[3:6, 6:6+n_rw] = A_cont[4:7, 7:7+n_rw] * 2.0
            # h wrt h
            A_err_cont[6:6+n_rw, 6:6+n_rw] = A_cont[7:7+n_rw, 7:7+n_rw]

        # Control Jacobian for error state
        B_err_cont[0:3, :] = B_cont[0:3, :]  # w wrt u
        B_err_cont[3:6, :] = B_cont[4:7, :] * 2.0  # d_theta wrt u
        if n_rw > 0:
            B_err_cont[6:6+n_rw, :] = B_cont[7:7+n_rw, :]  # h wrt u

        # Discretize error dynamics
        self.A_err = np.eye(self.n_err) + dt * A_err_cont
        self.B_err = dt * B_err_cont
        self.c_err = np.zeros(self.n_err)  # Error dynamics are centered at reference

    def solve_riccati(self) -> Tuple[list, list]:
        """
        Solve discrete Riccati equation for LQR gains in ERROR state space.

        Uses the reduced error state dynamics (A_err, B_err) and cost matrices
        (Q, Qf) which are all (n_err x n_err) dimension.

        Returns:
            (P_riccati, K_riccati): Lists of cost-to-go matrices and gains
                P_riccati[k]: (n_err x n_err) cost-to-go matrix
                K_riccati[k]: (m x n_err) feedback gain matrix
        """
        N = self.settings.track_horizon

        P_riccati = [None] * (N + 1)
        K_riccati = [None] * N

        # Terminal cost (n_err x n_err)
        P_riccati[N] = self.Qf.copy()

        # Backward recursion using ERROR dynamics
        for k in range(N - 1, -1, -1):
            P_next = P_riccati[k + 1]

            # Use error state dynamics (A_err, B_err)
            BtP = self.B_err.T @ P_next
            BtPB = BtP @ self.B_err
            BtPA = BtP @ self.A_err

            # Add regularization for numerical stability
            R_reg = self.R + self._rho * np.eye(self.m) + 1e-8 * np.eye(self.m)

            # K = (R + rho*I + B'PB)^{-1} B'PA
            # K is (m x n_err)
            K_riccati[k] = solve(R_reg + BtPB, BtPA, assume_a='pos')

            # P = Q + A'PA - A'PB*K
            P_riccati[k] = self.Q + self.A_err.T @ P_next @ self.A_err - BtPA.T @ K_riccati[k]

            # Ensure symmetry
            P_riccati[k] = 0.5 * (P_riccati[k] + P_riccati[k].T)

        return P_riccati, K_riccati

    def compute_state_error(
        self,
        x: NDArray[np.float64],
        x_ref: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute state error with proper quaternion handling.

        Returns the REDUCED error state (n_err = 6 + n_rw) matching TVLQR's formulation:
        - Angular velocity error: 3D (indices 0:3)
        - Attitude error: 3D linearized (indices 3:6), NOT 4D quaternion
        - RW momentum error: n_rw (indices 6:6+n_rw)

        This reduced representation linearizes the quaternion to a 3D attitude error
        using the small-angle approximation: d_theta ≈ 2 * vec(q_err)

        Args:
            x: Current state [w(3), q(4), h(n_rw)]
            x_ref: Reference state [w(3), q(4), h(n_rw)]

        Returns:
            State error vector (n_err,) where n_err = n - 1
        """
        n_rw = self.n - 7
        error = np.zeros(self.n_err)  # Reduced dimension!

        # Angular velocity error (indices 0:3)
        error[0:3] = x[0:3] - x_ref[0:3]

        # Quaternion error using quat_diff (same as TVLQR)
        q = normalize(x[3:7])
        q_ref = normalize(x_ref[3:7])
        q_err = quat_diff(q, q_ref)

        # Linearized attitude error: 2 * vec(q_err)
        # This gives the small-angle approximation of the rotation error
        error[3:6] = 2.0 * quat_to_vec3(q_err)

        # RW momentum error (indices 6:6+n_rw)
        if n_rw > 0:
            error[6:6+n_rw] = x[7:7+n_rw] - x_ref[7:7+n_rw]

        return error

    def admm_x_update(
        self,
        X: NDArray[np.float64],
        U: NDArray[np.float64],
        x0: NDArray[np.float64],
        X_ref: NDArray[np.float64],
        U_ref: NDArray[np.float64],
        K_riccati: list
    ) -> None:
        """
        ADMM x-update: solve unconstrained LQR with ADMM penalty.

        This computes the optimal trajectory assuming u = z (no constraints),
        but adds an ADMM penalty term (rho/2)||u - z + y||^2 to push u toward
        the constrained solution.

        Args:
            X: State trajectory to fill (n, N+1)
            U: Control trajectory to fill (m, N)
            x0: Initial state
            X_ref: Reference states (n, N+1)
            U_ref: Reference controls (m, N)
            K_riccati: LQR gains from Riccati solution
        """
        N = self.settings.track_horizon

        X[:, 0] = x0

        for k in range(N):
            # State error for LQR
            x_err = self.compute_state_error(X[:, k], X_ref[:, k])

            # Riccati control (without ADMM term)
            u_riccati = U_ref[:, k] - K_riccati[k] @ x_err

            # ADMM term pushes u toward z - y
            # Modified control: u_admm = (R + rho*I)^{-1} (R*u_riccati + rho*(z - y))
            R_diag = np.diag(self.R)
            u_admm = (R_diag * u_riccati + self._rho * (self.Z[:, k] - self.Y[:, k])) / (R_diag + self._rho)

            U[:, k] = u_admm

            # Propagate dynamics
            X[:, k + 1] = self.A @ X[:, k] + self.B @ U[:, k] + self.c

            # Normalize quaternion
            X[3:7, k + 1] = normalize(X[3:7, k + 1])

    def admm_z_update(self, U: NDArray[np.float64]) -> None:
        """
        ADMM z-update: project controls onto bounds.

        z = clip(u + y, u_min, u_max)

        Args:
            U: Current control trajectory (m, N)
        """
        for k in range(self.settings.track_horizon):
            self.Z[:, k] = np.clip(U[:, k] + self.Y[:, k], self.u_min, self.u_max)

    def admm_y_update(self, U: NDArray[np.float64]) -> None:
        """
        ADMM y-update: dual variable update.

        y = y + u - z

        Args:
            U: Current control trajectory (m, N)
        """
        self.Y = self.Y + U - self.Z

    def compute_residuals(self, U: NDArray[np.float64]) -> Tuple[float, float]:
        """
        Compute primal and dual residuals for convergence check.

        Args:
            U: Current control trajectory

        Returns:
            (primal_residual, dual_residual)
        """
        primal_res = np.linalg.norm(U - self.Z, 'fro')
        dual_res = self._rho * np.linalg.norm(self.Z - self._Z_prev, 'fro')
        self._Z_prev = self.Z.copy()
        return primal_res, dual_res

    def check_convergence(self, primal_res: float, dual_res: float) -> bool:
        """
        Check ADMM convergence using primal/dual residual tolerances.

        Args:
            primal_res: Primal residual ||u - z||
            dual_res: Dual residual rho * ||z^{k+1} - z^k||

        Returns:
            True if converged
        """
        N = self.settings.track_horizon
        m = self.m

        eps_pri = self.settings.abs_tol * np.sqrt(N * m) + \
                  self.settings.rel_tol * max(np.linalg.norm(self.Z, 'fro'), 1.0)
        eps_dual = self.settings.abs_tol * np.sqrt(N * m) + \
                   self.settings.rel_tol * self._rho * np.linalg.norm(self.Y, 'fro')

        return primal_res < eps_pri and dual_res < eps_dual

    def update_rho(self, primal_res: float, dual_res: float) -> None:
        """
        Adaptive rho update based on primal/dual residual ratio.

        If primal >> dual: increase rho to penalize constraint violation more
        If dual >> primal: decrease rho to allow more freedom in u

        Args:
            primal_res: Primal residual
            dual_res: Dual residual
        """
        if not self.settings.adaptive_rho:
            return

        ratio = primal_res / (dual_res + 1e-10)
        if ratio > 10.0:
            self._rho = min(self._rho * 2.0, self.settings.rho_max)
        elif ratio < 0.1:
            self._rho = max(self._rho / 2.0, self.settings.rho_min)

    def solve(
        self,
        x_current: NDArray[np.float64],
        t_current: float,
        est_sat: EstimatedSatellite,
        os: Orbital_State,
        use_altro_gains: bool = True
    ) -> TinyMPCResult:
        """
        Solve tracking MPC at current state and time.

        Args:
            x_current: Current state vector
            t_current: Current time (J2000 centuries)
            est_sat: Satellite model for dynamics
            os: Orbital state for environment
            use_altro_gains: If True and K_ref available, use ALTRO's pre-computed
                           gains with saturation (faster, more robust). If False,
                           use full ADMM optimization.

        Returns:
            TinyMPCResult with optimal control and solve statistics
        """
        start_time = time.perf_counter()

        if not self.has_reference:
            return TinyMPCResult(
                u_opt=np.zeros(self.m),
                X_pred=np.zeros((self.n, 1)),
                U_pred=np.zeros((self.m, 1)),
                iterations=0,
                solve_time_ms=0.0,
                converged=False,
                tracking_error=0.0
            )

        # Get reference state and control at current time
        x_ref_0, u_ref_0 = self.interpolate_reference(t_current)

        # Compute state error (reduced dimension)
        x_err = self.compute_state_error(x_current, x_ref_0)
        tracking_error = np.linalg.norm(x_err)

        # Option 1: Use ALTRO's pre-computed K gains (constrained TVLQR)
        # This is faster and more robust since ALTRO's gains are computed with
        # the exact dynamics and cost function.
        if use_altro_gains and self.K_ref is not None:
            # Interpolate K gain at current time
            K_interp = self._interpolate_K_gain(t_current)

            if K_interp is not None:
                # Apply tracking control: u = u_ref - K @ dx
                u_opt = u_ref_0 - K_interp @ x_err

                # Saturate to control bounds
                u_opt = np.clip(u_opt, self.u_min, self.u_max)

                solve_time = (time.perf_counter() - start_time) * 1000

                return TinyMPCResult(
                    u_opt=u_opt,
                    X_pred=np.zeros((self.n, 1)),  # Not computed in this mode
                    U_pred=np.zeros((self.m, 1)),
                    iterations=0,
                    solve_time_ms=solve_time,
                    converged=True,  # Direct computation always "converges"
                    tracking_error=tracking_error
                )

        # Option 2: Full ADMM optimization (fallback)
        N = self.settings.track_horizon

        # Build local reference trajectory for MPC horizon
        X_ref_local, U_ref_local = self.build_local_reference(t_current)

        # Linearize dynamics about reference
        self.linearize_dynamics(x_ref_0, u_ref_0, est_sat, os)

        # Solve Riccati for LQR gains
        P_riccati, K_riccati = self.solve_riccati()

        # Initialize trajectories
        X = np.zeros((self.n, N + 1))
        U = np.zeros((self.m, N))

        if self.has_warm_start and self.X_warm is not None and self.U_warm is not None:
            # Shift warm start by one timestep
            X[:, :-1] = self.X_warm[:, 1:]
            X[:, -1] = self.X_warm[:, -1]
            U[:, :-1] = self.U_warm[:, 1:]
            U[:, -1] = self.U_warm[:, -1]
        else:
            # Initialize from reference
            X = X_ref_local.copy()
            U = U_ref_local.copy()
            self.Z = U.copy()
            self.Y = np.zeros_like(U)

        self._Z_prev = self.Z.copy()

        # ADMM iterations
        converged = False
        iterations = 0

        for iter in range(self.settings.max_iter):
            self.admm_x_update(X, U, x_current, X_ref_local, U_ref_local, K_riccati)
            self.admm_z_update(U)
            self.admm_y_update(U)

            iterations = iter + 1

            if (iter + 1) % self.settings.check_interval == 0:
                primal_res, dual_res = self.compute_residuals(U)

                if self.check_convergence(primal_res, dual_res):
                    converged = True
                    break

                self.update_rho(primal_res, dual_res)

        # Extract optimal control (first element of Z, which satisfies bounds)
        u_opt = self.Z[:, 0]

        # Store for warm start
        self.X_warm = X.copy()
        self.U_warm = U.copy()
        self.has_warm_start = True

        solve_time = (time.perf_counter() - start_time) * 1000

        return TinyMPCResult(
            u_opt=u_opt,
            X_pred=X,
            U_pred=U,
            iterations=iterations,
            solve_time_ms=solve_time,
            converged=converged,
            tracking_error=tracking_error
        )

    def _interpolate_K_gain(self, t: float) -> Optional[NDArray[np.float64]]:
        """
        Interpolate ALTRO's K gain at time t.

        K_ref is stored as (m*n_err, N) where each column is a flattened gain matrix.

        Args:
            t: Time to interpolate at

        Returns:
            K matrix (m, n_err) or None if not available
        """
        if self.K_ref is None or self.times_ref is None:
            return None

        t_start = self.times_ref[0]
        t_end = self.times_ref[-1]

        # Clamp to valid range
        t = np.clip(t, t_start, t_end - 1e-10)

        # Find interpolation index
        t_rel = t - t_start
        idx_float = t_rel / self.dt_ref
        idx = int(np.floor(idx_float))

        N = len(self.times_ref) - 1
        idx = np.clip(idx, 0, N - 1)

        # K_ref shape: (m * n_err, N) - each column is flattened K
        try:
            K_flat = self.K_ref[:, idx]
            K = K_flat.reshape(self.m, self.n_err)
            return K
        except (IndexError, ValueError):
            return None

    def reset(self) -> None:
        """Reset solver state (clear warm start and ADMM variables)."""
        N = self.settings.track_horizon
        self.Z = np.zeros((self.m, N))
        self.Y = np.zeros((self.m, N))
        self._Z_prev = np.zeros((self.m, N))
        self.has_warm_start = False
        self.X_warm = None
        self.U_warm = None
        self._rho = self.settings.rho


class Plan_and_Track_TinyMPC_Py(PlanAndTrackBase):
    """
    Trajectory-following controller using ALTRO planning and pure Python TinyMPC.

    This controller uses:
    - ALTRO (via C++) for trajectory planning (Plan phase)
    - TinyMPC (pure Python) for real-time tracking with constraints (Track phase)

    The pure Python implementation is useful for:
    - Debugging and algorithm development
    - Platforms without C++ toolchain
    - Educational purposes and experimentation

    Compared to TVLQR tracking:
    - TinyMPC respects actuator bounds explicitly
    - Better disturbance rejection via receding horizon optimization
    - More computation (~1-50ms per solve vs ~0.1ms for TVLQR)

    Attributes:
        tinympc_settings: Configuration for TinyMPC solver
        active_trajectory: Current ALTRO trajectory being tracked
    """

    tinympc_settings: TinyMPCSettings
    _mpc: TinyMPCSolverPy
    _last_replan_time: float

    def __init__(
        self,
        est_sat: EstimatedSatellite,
        planner_settings: PlannerSettings,
        tinympc_settings: Optional[TinyMPCSettings] = None
    ) -> None:
        """
        Initialize Plan and Track TinyMPC controller.

        Args:
            est_sat: Estimated satellite model with actuators and sensors
            planner_settings: Configuration for the ALTRO trajectory planner
            tinympc_settings: Configuration for TinyMPC solver (uses defaults if None)
        """
        # Initialize ALTRO planner (standard TVLQR formulation=0)
        self._init_planner(est_sat, planner_settings, tracking_lqr_formulation=0)

        # Store TinyMPC settings
        self.tinympc_settings = tinympc_settings if tinympc_settings is not None else TinyMPCSettings()

        # Get control limits from actuators
        u_max = np.array([act.u_max for act in est_sat.actuators])
        u_min = -u_max

        # Create Python TinyMPC solver
        self._mpc = TinyMPCSolverPy(
            n=self.state_dim,
            m=self.ctrl_dim,
            settings=self.tinympc_settings,
            u_min=u_min,
            u_max=u_max
        )

        # Set cost matrices from planner settings
        Q, R, Qf = self._build_cost_matrices()
        self._mpc.set_cost_matrices(Q, R, Qf)

        # Re-planning state
        self._last_replan_time = -np.inf

    def _build_cost_matrices(self) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Build Q, R, Qf matrices from planner cost settings.

        Maps the CostWeights from planner_settings to REDUCED ERROR STATE cost
        matrices suitable for MPC tracking.

        Error state dimension: n_err = 6 + n_rw = n - 1
        - Angular velocity error: indices 0:3
        - Attitude error (3D linearized): indices 3:6
        - RW momentum error: indices 6:6+n_rw

        Returns:
            (Q, R, Qf): Cost matrices for error state
        """
        cost = self.planner_settings.cost_tvlqr
        n = self.state_dim
        n_err = n - 1  # Reduced dimension for quaternion linearization
        n_rw = n - 7
        m = self.ctrl_dim

        # State tracking cost Q (n_err x n_err)
        Q = np.zeros((n_err, n_err))
        # Angular velocity weights (indices 0:3)
        for i in range(3):
            Q[i, i] = cost.ang_vel
        # Attitude error weights (indices 3:6) - 3D linearized, not 4D quaternion
        for i in range(3, 6):
            Q[i, i] = cost.angle
        # RW momentum weights (indices 6:6+n_rw) - typically smaller
        for i in range(6, n_err):
            Q[i, i] = cost.ang_vel * 0.1

        # Control cost R (m x m)
        R = cost.control_mult * np.eye(m)

        # Terminal cost Qf (n_err x n_err)
        Qf = np.zeros((n_err, n_err))
        for i in range(3):
            Qf[i, i] = cost.ang_vel_N
        for i in range(3, 6):
            Qf[i, i] = cost.angle_N
        for i in range(6, n_err):
            Qf[i, i] = cost.ang_vel_N * 0.1

        return Q, R, Qf

    def _check_replan_needed(
        self,
        x_current: NDArray[np.float64],
        current_time: float
    ) -> bool:
        """
        Check if re-planning should be triggered based on tracking error.

        Args:
            x_current: Current state
            current_time: Current time (J2000 centuries)

        Returns:
            True if re-planning is recommended
        """
        if not self.tinympc_settings.replan_enabled:
            return False

        # Check minimum interval since last replan
        # Convert J2000 centuries to seconds for comparison
        time_since_replan = (current_time - self._last_replan_time) * 3155760000  # centuries to seconds
        if time_since_replan < self.tinympc_settings.replan_min_interval:
            return False

        if self.active_trajectory is None:
            return True

        if not self.active_trajectory.is_valid_time(current_time):
            return True

        # Get reference state
        x_ref = self.active_trajectory.get_state_at(current_time)

        # Compute attitude error (geodesic distance)
        q_curr = normalize(x_current[3:7])
        q_ref = normalize(x_ref[3:7])
        q_err = quat_diff(q_curr, q_ref)
        # Geodesic angle = 2 * arcsin(|vec(q_err)|)
        vec_err = quat_to_vec3(q_err)
        vec_norm = np.linalg.norm(vec_err)
        attitude_error = 2.0 * np.arcsin(min(vec_norm, 1.0))

        # Compute angular velocity error
        w_err = np.linalg.norm(x_current[0:3] - x_ref[0:3])

        # Check thresholds
        if attitude_error > self.tinympc_settings.replan_attitude_threshold:
            return True
        if w_err > self.tinympc_settings.replan_angvel_threshold:
            return True

        return False

    def find_u(
        self,
        x_hat: NDArray[np.float64],
        sens: NDArray[np.float64],
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        goal_vector_eci: Optional[NDArray[np.float64]] = None,
        w_ref: Optional[NDArray[np.float64]] = None,
        clip: bool = True
    ) -> NDArray[np.float64]:
        """
        Compute control using TinyMPC tracking.

        Args:
            x_hat: Estimated state vector [w(3), q(4), h(n_rw)]
            sens: Sensor measurements (unused)
            est_sat: Estimated satellite model for dynamics
            os_hat: Estimated orbital state
            goal_vector_eci: Goal vector in ECI (unused, from trajectory)
            w_ref: Reference angular velocity (unused, from trajectory)
            clip: Clip control to hardware limits (default True)

        Returns:
            Optimal control vector (m,)
        """
        current_time = os_hat.J2000

        # Check trajectory validity
        if self.active_trajectory is None:
            raise RuntimeError(f"TinyMPC_Py: No active trajectory at t={current_time}")

        if not self.active_trajectory.is_valid_time(current_time):
            raise RuntimeError(
                f"TinyMPC_Py: Trajectory expired at t={current_time}. "
                f"Valid range: [{self.active_trajectory.start_time}, {self.active_trajectory.end_time}]"
            )

        # Solve TinyMPC
        result = self._mpc.solve(x_hat, current_time, est_sat, os_hat)

        if self.tinympc_settings.verbose >= 1:
            status = "converged" if result.converged else "max_iter"
            print(f"TinyMPC_Py: {result.iterations} iters, {result.solve_time_ms:.2f}ms, "
                  f"{status}, err={result.tracking_error:.4f}")

        return self.clip_control(result.u_opt, clip=clip)

    def calculate_trajectory(
        self,
        t_start: float,
        duration: float,
        x_0: np.ndarray,
        os_0: Orbital_State,
        goals: GoalList,
        verbose: bool = False,
        vecsPy_precomputed: tuple = None,
        N_precomputed: int = None,
        t_end_precomputed: float = None
    ) -> Trajectory:
        """
        Calculate an optimal trajectory using ALTRO and load into TinyMPC.

        This calls the ALTRO planner to generate an optimal reference trajectory,
        then loads it into the TinyMPC solver for tracking.

        Args:
            t_start: Start time in J2000 centuries
            duration: Duration in seconds
            x_0: Initial state vector
            os_0: Initial orbital state
            goals: Goal list for attitude reference
            verbose: Whether to print debug information
            vecsPy_precomputed: Optional pre-computed environment vectors
            N_precomputed: Number of timesteps (with precomputed vecs)
            t_end_precomputed: End time (with precomputed vecs)

        Returns:
            Trajectory object (also loaded into internal TinyMPC solver)
        """
        # Call base class trajectory computation (ALTRO)
        lqr_times, Xset, Uset, Kset, Sset = self._calculate_trajectory_common(
            t_start, duration, x_0, os_0, goals, verbose,
            vecsPy_precomputed, N_precomputed, t_end_precomputed
        )

        traj = Trajectory(lqr_times, Xset, Uset, Kset, Sset)

        # Load reference into TinyMPC solver
        dt = self.planner_settings.dt_tvlqr
        self._mpc.load_reference(Xset, Uset, Kset, lqr_times, dt)

        # Update replan tracking
        self._last_replan_time = t_start

        return traj

    def needs_replan(self, x_current: NDArray[np.float64], os: Orbital_State) -> bool:
        """
        Check if re-planning is recommended.

        This is a convenience method that can be called by the user to decide
        whether to trigger a new trajectory calculation.

        Args:
            x_current: Current state
            os: Current orbital state

        Returns:
            True if re-planning is recommended
        """
        return self._check_replan_needed(x_current, os.J2000)
