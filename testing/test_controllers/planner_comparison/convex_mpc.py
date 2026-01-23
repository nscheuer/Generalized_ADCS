"""
Convex MPC (Model Predictive Control) Trajectory Planner.

This planner uses convex optimization (QP/SOCP) for trajectory generation.
The nonlinear attitude dynamics are linearized around a reference trajectory,
and the resulting convex problem is solved using efficient solvers like OSQP.

This approach is commonly used for real-time spacecraft control where
computational efficiency is critical.

References:
    Guiggiani et al. (2015) "Fixed-Point Constrained Model Predictive Control"
    Kalabic et al. (2017) "MPC for Spacecraft Attitude Control"
    Eren et al. (2017) "Model Predictive Control in Aerospace Systems"
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple
from dataclasses import dataclass
import cvxpy as cp

from .base_planner import BasePlanner, PlannerResult, PlannerConfig, DynamicsModel


@dataclass
class ConvexMPCConfig(PlannerConfig):
    """Configuration for convex MPC planner."""
    
    # MPC horizon (may be shorter than full horizon for real-time)
    mpc_horizon: int = 30            # Number of MPC steps
    
    # Solver settings
    solver: str = "OSQP"             # OSQP, ECOS, or SCS
    warm_start: bool = True
    
    # Linearization
    n_linearization_iters: int = 3   # Number of SCP-like iterations
    
    # Trust region for linearization
    trust_region: float = 1.0


class ConvexMPCPlanner(BasePlanner):
    """
    Convex MPC trajectory planner.
    
    The approach:
    1. Linearize dynamics around current/reference trajectory
    2. Formulate convex QP with linearized dynamics constraints
    3. Solve using efficient convex solver (OSQP)
    4. Optionally iterate (like SCP) for better accuracy
    
    The QP has the form:
        min  Σ (x_k - x_ref)' Q (x_k - x_ref) + u_k' R u_k + terminal cost
        s.t. x_{k+1} = A_k x_k + B_k u_k + c_k  (linearized dynamics)
             u_min ≤ u_k ≤ u_max               (control bounds)
             x_0 = x_init                       (initial condition)
    """
    
    def __init__(self, config: Optional[ConvexMPCConfig] = None):
        """Initialize convex MPC planner."""
        if config is None:
            config = ConvexMPCConfig()
        super().__init__(config)
        self.config: ConvexMPCConfig = config
        self._name = "ConvexMPC"
    
    def solve(
        self,
        x0: NDArray[np.float64],
        x_goal: NDArray[np.float64],
        J_inertia: NDArray[np.float64],
        u_max: NDArray[np.float64],
        B_field: Optional[NDArray[np.float64]] = None,
        **kwargs
    ) -> PlannerResult:
        """
        Solve trajectory optimization using convex MPC.
        
        Args:
            x0: Initial state
            x_goal: Goal state
            J_inertia: Inertia matrix
            u_max: Control limits
            B_field: Magnetic field (optional)
            
        Returns:
            PlannerResult with optimized trajectory
        """
        import time
        start_time = time.perf_counter()
        
        # Setup
        n_rw = len(x0) - 7
        n_x = len(x0)
        n_u = len(u_max)
        
        N = min(self.config.mpc_horizon, int(self.config.horizon / self.config.dt) + 1)
        dt = self.config.dt
        
        # Extend to full horizon if needed
        N_full = int(self.config.horizon / self.config.dt) + 1
        
        times = np.linspace(0, self.config.horizon, N_full)
        
        # Create dynamics model
        dynamics = DynamicsModel(
            J_inertia=J_inertia,
            rw_axes=np.eye(3)[:, :n_rw] if n_rw > 0 else None,
            n_mtq=n_u - n_rw,
            has_rw=n_rw > 0
        )
        
        # Initialize reference trajectory
        x_ref, u_ref = self._initialize_trajectory(x0, x_goal, N, n_x, n_u)
        
        # Iterative linearization (SCP-like)
        converged = False
        for iter_idx in range(self.config.n_linearization_iters):
            # Solve convex subproblem
            x_new, u_new, cost, status = self._solve_qp(
                x0, x_goal, x_ref, u_ref, u_max, N, n_x, n_u, dt, dynamics, B_field
            )
            
            if status != "optimal":
                break
            
            # Check convergence
            dx_norm = np.linalg.norm(x_new - x_ref)
            if dx_norm < self.config.convergence_tol:
                converged = True
                x_ref = x_new
                u_ref = u_new
                break
            
            x_ref = x_new
            u_ref = u_new
        
        solve_time = time.perf_counter() - start_time
        
        # If MPC horizon < full horizon, extend with final values
        if N < N_full:
            states, controls = self._extend_trajectory(x_ref, u_ref, N, N_full, n_x, n_u)
        else:
            states = x_ref
            controls = u_ref[:N_full-1] if len(u_ref) >= N_full-1 else u_ref
        
        # Normalize quaternions
        for k in range(len(states)):
            states[k, 3:7] = states[k, 3:7] / np.linalg.norm(states[k, 3:7])
        
        # Compute errors
        q_final = states[-1, 3:7]
        q_goal = x_goal[3:7]
        angle_error = self._quaternion_angle(q_final, q_goal)
        omega_error = np.linalg.norm(states[-1, :3] - x_goal[:3])
        
        return PlannerResult(
            times=times,
            states=states,
            controls=controls,
            solve_time=solve_time,
            converged=converged,
            iterations=iter_idx + 1,
            final_cost=cost if 'cost' in dir() else 0.0,
            max_constraint_violation=0.0,
            solver_info={
                "final_angle_error_deg": np.degrees(angle_error),
                "final_omega_error": omega_error,
                "mpc_horizon": N,
                "n_linearization_iters": iter_idx + 1,
                "qp_status": status if 'status' in dir() else "unknown",
            }
        )
    
    def _initialize_trajectory(
        self,
        x0: NDArray[np.float64],
        x_goal: NDArray[np.float64],
        N: int,
        n_x: int,
        n_u: int
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Initialize reference trajectory with linear interpolation."""
        x_ref = np.zeros((N, n_x))
        u_ref = np.zeros((N - 1, n_u))
        
        for k in range(N):
            alpha = k / (N - 1)
            x_ref[k] = (1 - alpha) * x0 + alpha * x_goal
            x_ref[k, 3:7] = x_ref[k, 3:7] / np.linalg.norm(x_ref[k, 3:7])
        
        return x_ref, u_ref
    
    def _solve_qp(
        self,
        x0: NDArray[np.float64],
        x_goal: NDArray[np.float64],
        x_ref: NDArray[np.float64],
        u_ref: NDArray[np.float64],
        u_max: NDArray[np.float64],
        N: int,
        n_x: int,
        n_u: int,
        dt: float,
        dynamics: DynamicsModel,
        B_field: Optional[NDArray[np.float64]]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float, str]:
        """
        Solve the convex QP subproblem.
        
        Returns:
            (x_new, u_new, cost, status)
        """
        # Decision variables
        x = cp.Variable((N, n_x))
        u = cp.Variable((N - 1, n_u))
        
        # Cost function
        cost = 0
        
        # Running cost
        Q_omega = self.config.Q_angular_vel * np.eye(3)
        R = self.config.R_control * np.eye(n_u)
        
        for k in range(N - 1):
            # Angular velocity cost
            cost += cp.quad_form(x[k, :3], Q_omega)
            # Control cost
            cost += cp.quad_form(u[k], R)
        
        # Terminal cost
        Q_att = self.config.Q_attitude_terminal * np.eye(3)
        Q_omega_term = self.config.Q_angular_vel_terminal * np.eye(3)
        
        # Quaternion error (linearized around reference)
        q_goal = x_goal[3:7]
        q_ref_N = x_ref[-1, 3:7]
        
        # Linearized quaternion error: e ≈ 2 * (q - q_ref) for small errors
        # We use the vector part of quaternion as approximate error
        cost += cp.quad_form(x[N-1, 3:6] - q_goal[:3], Q_att)
        
        # Terminal angular velocity
        cost += cp.quad_form(x[N-1, :3] - x_goal[:3], Q_omega_term)
        
        # Constraints
        constraints = []
        
        # Initial condition
        constraints.append(x[0] == x0)
        
        # Linearized dynamics constraints
        for k in range(N - 1):
            # Get linearization point
            xk_ref = x_ref[k]
            uk_ref = u_ref[k] if k < len(u_ref) else np.zeros(n_u)
            
            # Get B-field
            B_k = self._get_B_at_k(B_field, k, N)
            
            # Linearize
            A_k, B_k_mat = dynamics.discretize_jacobians(xk_ref, uk_ref, dt, B_k)
            x_nom = dynamics.discretize(xk_ref, uk_ref, dt, B_k)
            
            # Affine dynamics: x_{k+1} = x_nom + A_k(x_k - x_ref_k) + B_k(u_k - u_ref_k)
            constraints.append(
                x[k+1] == x_nom + A_k @ (x[k] - xk_ref) + B_k_mat @ (u[k] - uk_ref)
            )
        
        # Control bounds
        for k in range(N - 1):
            constraints.append(u[k] >= -u_max)
            constraints.append(u[k] <= u_max)
        
        # Trust region (optional, helps convergence)
        if self.config.trust_region > 0:
            for k in range(N):
                constraints.append(cp.norm(x[k] - x_ref[k], 2) <= self.config.trust_region)
        
        # Solve
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            if self.config.solver == "OSQP":
                problem.solve(solver=cp.OSQP, warm_start=self.config.warm_start, verbose=False)
            elif self.config.solver == "ECOS":
                problem.solve(solver=cp.ECOS, verbose=False)
            elif self.config.solver == "SCS":
                problem.solve(solver=cp.SCS, verbose=False)
            else:
                problem.solve(verbose=False)
        except Exception as e:
            return x_ref, u_ref, float('inf'), f"failed: {e}"
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            return x_ref, u_ref, float('inf'), problem.status
        
        # Extract solution
        x_new = x.value
        u_new = u.value
        
        # Normalize quaternions
        for k in range(N):
            if np.linalg.norm(x_new[k, 3:7]) > 1e-6:
                x_new[k, 3:7] = x_new[k, 3:7] / np.linalg.norm(x_new[k, 3:7])
        
        return x_new, u_new, problem.value, "optimal"
    
    def _extend_trajectory(
        self,
        x_mpc: NDArray[np.float64],
        u_mpc: NDArray[np.float64],
        N_mpc: int,
        N_full: int,
        n_x: int,
        n_u: int
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Extend MPC trajectory to full horizon."""
        states = np.zeros((N_full, n_x))
        controls = np.zeros((N_full - 1, n_u))
        
        # Copy MPC solution
        states[:N_mpc] = x_mpc
        controls[:min(N_mpc-1, len(u_mpc))] = u_mpc[:min(N_mpc-1, len(u_mpc))]
        
        # Extend with final values
        for k in range(N_mpc, N_full):
            states[k] = states[N_mpc - 1]
        
        for k in range(min(N_mpc-1, len(u_mpc)), N_full - 1):
            controls[k] = np.zeros(n_u)  # Zero control after MPC horizon
        
        return states, controls
    
    def _get_B_at_k(
        self,
        B_field: Optional[NDArray[np.float64]],
        k: int,
        N: int
    ) -> Optional[NDArray[np.float64]]:
        """Get magnetic field at timestep k."""
        if B_field is None:
            return None
        if B_field.ndim == 1:
            return B_field
        return B_field[min(k, len(B_field) - 1)]
    
    @staticmethod
    def _quaternion_angle(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> float:
        """Compute angle between two quaternions."""
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        dot = np.abs(np.dot(q1, q2))
        return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))
