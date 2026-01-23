"""
Direct Collocation (DIRCOL) Trajectory Planner.

This planner uses direct transcription to convert the continuous optimal control
problem into a nonlinear program (NLP). It uses Hermite-Simpson collocation
for dynamics constraints.

Direct collocation is one of the most widely used methods in aerospace trajectory
optimization, used in many NASA missions and commercial applications.

References:
    Hargraves & Paris (1987) "Direct Trajectory Optimization Using Nonlinear Programming"
    Betts (1998) "Survey of Numerical Methods for Trajectory Optimization"
    Kelly (2017) "An Introduction to Trajectory Optimization"
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize, NonlinearConstraint
from scipy.interpolate import CubicSpline

from .base_planner import BasePlanner, PlannerResult, PlannerConfig, DynamicsModel


@dataclass
class DirectCollocationConfig(PlannerConfig):
    """Configuration for direct collocation planner."""
    
    # NLP solver settings
    method: str = "SLSQP"            # scipy optimizer: SLSQP, trust-constr
    ftol: float = 1e-6               # Function tolerance
    gtol: float = 1e-6               # Gradient tolerance
    
    # Collocation settings
    collocation_method: str = "hermite-simpson"  # hermite-simpson or trapezoidal
    
    # Constraint tolerances
    dynamics_tol: float = 1e-4       # Dynamics constraint tolerance


class DirectCollocationPlanner(BasePlanner):
    """
    Direct Collocation trajectory planner using Hermite-Simpson transcription.
    
    The continuous optimal control problem:
        min ∫ L(x,u) dt + Φ(x_f)
        s.t. ẋ = f(x,u)
             g(x,u) ≤ 0
             x(0) = x_0
             
    Is transcribed into an NLP:
        min Σ L_k + Φ(x_N)
        s.t. defect constraints (dynamics)
             path constraints
             boundary constraints
             
    Hermite-Simpson uses cubic polynomial interpolation between knot points,
    with the midpoint derivative matching the dynamics.
    """
    
    def __init__(self, config: Optional[DirectCollocationConfig] = None):
        """Initialize direct collocation planner."""
        if config is None:
            config = DirectCollocationConfig()
        super().__init__(config)
        self.config: DirectCollocationConfig = config
        self._name = "DIRCOL"
    
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
        Solve trajectory optimization using direct collocation.
        
        Args:
            x0: Initial state [omega (3), quaternion (4), rw_momentum (n_rw)]
            x_goal: Goal state
            J_inertia: Spacecraft inertia matrix
            u_max: Control limits
            B_field: Magnetic field (optional)
            
        Returns:
            PlannerResult with optimized trajectory
        """
        import time
        start_time = time.perf_counter()
        
        # Setup dimensions
        n_rw = len(x0) - 7
        n_x = len(x0)
        n_u = len(u_max)
        
        N = int(np.ceil(self.config.horizon / self.config.dt)) + 1
        dt = self.config.dt
        times = np.linspace(0, self.config.horizon, N)
        
        # Create dynamics model
        dynamics = DynamicsModel(
            J_inertia=J_inertia,
            rw_axes=np.eye(3)[:, :n_rw] if n_rw > 0 else None,
            n_mtq=n_u - n_rw,
            has_rw=n_rw > 0
        )
        
        # Decision variables: [x_0, u_0, x_1, u_1, ..., x_{N-1}, u_{N-1}, x_N]
        # Total: N+1 states, N controls
        n_vars = (N + 1) * n_x + N * n_u
        
        # Initialize with linear interpolation
        z0 = self._initialize_decision_vars(x0, x_goal, N, n_x, n_u)
        
        # Cost function
        def cost(z):
            return self._compute_cost(z, x_goal, N, n_x, n_u, dt)
        
        # Cost gradient (for efficiency)
        def cost_grad(z):
            return self._compute_cost_gradient(z, x_goal, N, n_x, n_u, dt)
        
        # Dynamics constraints (defects)
        def dynamics_constraints(z):
            return self._compute_defects(z, N, n_x, n_u, dt, dynamics, B_field)
        
        # Control bounds
        bounds = self._create_bounds(x0, N, n_x, n_u, u_max)
        
        # Nonlinear constraints for dynamics
        defect_constraint = NonlinearConstraint(
            dynamics_constraints,
            -self.config.dynamics_tol,
            self.config.dynamics_tol
        )
        
        # Solve NLP
        result = minimize(
            cost,
            z0,
            method=self.config.method,
            jac=cost_grad,
            bounds=bounds,
            constraints=[defect_constraint],
            options={
                'ftol': self.config.ftol,
                'maxiter': self.config.max_iterations,
                'disp': self.config.verbose
            }
        )
        
        solve_time = time.perf_counter() - start_time
        
        # Extract solution
        states, controls = self._extract_solution(result.x, N, n_x, n_u)
        
        # Normalize quaternions
        for k in range(N + 1):
            states[k, 3:7] = states[k, 3:7] / np.linalg.norm(states[k, 3:7])
        
        # Compute final errors
        q_final = states[-1, 3:7]
        q_goal = x_goal[3:7]
        angle_error = self._quaternion_angle(q_final, q_goal)
        omega_error = np.linalg.norm(states[-1, :3] - x_goal[:3])
        
        # Compute constraint violation
        defects = dynamics_constraints(result.x)
        max_violation = np.max(np.abs(defects))
        
        return PlannerResult(
            times=times,
            states=states,
            controls=controls,
            solve_time=solve_time,
            converged=result.success,
            iterations=result.nit if hasattr(result, 'nit') else -1,
            final_cost=result.fun,
            max_constraint_violation=max_violation,
            solver_info={
                "final_angle_error_deg": np.degrees(angle_error),
                "final_omega_error": omega_error,
                "nlp_status": result.message,
                "n_function_evals": result.nfev if hasattr(result, 'nfev') else -1,
            }
        )
    
    def _initialize_decision_vars(
        self,
        x0: NDArray[np.float64],
        x_goal: NDArray[np.float64],
        N: int,
        n_x: int,
        n_u: int
    ) -> NDArray[np.float64]:
        """Initialize decision variables with linear interpolation."""
        n_vars = (N + 1) * n_x + N * n_u
        z = np.zeros(n_vars)
        
        for k in range(N + 1):
            alpha = k / N
            x_k = (1 - alpha) * x0 + alpha * x_goal
            # Normalize quaternion
            x_k[3:7] = x_k[3:7] / np.linalg.norm(x_k[3:7])
            
            idx_x = k * (n_x + n_u) if k < N else N * (n_x + n_u)
            z[idx_x:idx_x + n_x] = x_k
        
        return z
    
    def _compute_cost(
        self,
        z: NDArray[np.float64],
        x_goal: NDArray[np.float64],
        N: int,
        n_x: int,
        n_u: int,
        dt: float
    ) -> float:
        """Compute total cost (running + terminal)."""
        cost = 0.0
        
        # Running cost
        for k in range(N):
            idx_x = k * (n_x + n_u)
            idx_u = idx_x + n_x
            
            x_k = z[idx_x:idx_x + n_x]
            u_k = z[idx_u:idx_u + n_u]
            
            # Control cost
            cost += self.config.R_control * np.sum(u_k**2) * dt
            
            # State cost (angular velocity)
            cost += self.config.Q_angular_vel * np.sum(x_k[:3]**2) * dt
        
        # Terminal cost
        idx_xN = N * (n_x + n_u)
        x_N = z[idx_xN:idx_xN + n_x]
        
        # Attitude error
        q_err = self._quaternion_error_vec(x_N[3:7], x_goal[3:7])
        cost += self.config.Q_attitude_terminal * np.sum(q_err**2)
        
        # Angular velocity error
        omega_err = x_N[:3] - x_goal[:3]
        cost += self.config.Q_angular_vel_terminal * np.sum(omega_err**2)
        
        return cost
    
    def _compute_cost_gradient(
        self,
        z: NDArray[np.float64],
        x_goal: NDArray[np.float64],
        N: int,
        n_x: int,
        n_u: int,
        dt: float
    ) -> NDArray[np.float64]:
        """Compute cost gradient."""
        grad = np.zeros_like(z)
        
        # Running cost gradient
        for k in range(N):
            idx_x = k * (n_x + n_u)
            idx_u = idx_x + n_x
            
            x_k = z[idx_x:idx_x + n_x]
            u_k = z[idx_u:idx_u + n_u]
            
            # Control gradient
            grad[idx_u:idx_u + n_u] = 2 * self.config.R_control * u_k * dt
            
            # Angular velocity gradient
            grad[idx_x:idx_x + 3] = 2 * self.config.Q_angular_vel * x_k[:3] * dt
        
        # Terminal cost gradient
        idx_xN = N * (n_x + n_u)
        x_N = z[idx_xN:idx_xN + n_x]
        
        # Quaternion error gradient (simplified)
        q_err = self._quaternion_error_vec(x_N[3:7], x_goal[3:7])
        grad[idx_xN + 3:idx_xN + 6] = 2 * self.config.Q_attitude_terminal * q_err
        
        # Angular velocity gradient
        omega_err = x_N[:3] - x_goal[:3]
        grad[idx_xN:idx_xN + 3] = 2 * self.config.Q_angular_vel_terminal * omega_err
        
        return grad
    
    def _compute_defects(
        self,
        z: NDArray[np.float64],
        N: int,
        n_x: int,
        n_u: int,
        dt: float,
        dynamics: DynamicsModel,
        B_field: Optional[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        """
        Compute Hermite-Simpson defects.
        
        For each segment k:
        x_{k+1} - x_k - (dt/6) * (f_k + 4*f_c + f_{k+1}) = 0
        
        where f_c is the dynamics at the collocation midpoint.
        """
        defects = []
        
        for k in range(N):
            idx_xk = k * (n_x + n_u)
            idx_uk = idx_xk + n_x
            idx_xkp1 = (k + 1) * (n_x + n_u) if k < N - 1 else N * (n_x + n_u)
            idx_ukp1 = idx_xkp1 + n_x if k < N - 1 else idx_uk  # Use same control for last
            
            x_k = z[idx_xk:idx_xk + n_x]
            u_k = z[idx_uk:idx_uk + n_u]
            x_kp1 = z[idx_xkp1:idx_xkp1 + n_x]
            u_kp1 = z[idx_ukp1:idx_ukp1 + n_u] if k < N - 1 else u_k
            
            # Get B-field at this timestep
            B_k = self._get_B_at_k(B_field, k, N)
            
            # Dynamics at knot points
            f_k = dynamics.dynamics(x_k, u_k, B_k)
            f_kp1 = dynamics.dynamics(x_kp1, u_kp1, B_k)
            
            # Midpoint state and control (cubic interpolation)
            x_c = 0.5 * (x_k + x_kp1) + (dt / 8) * (f_k - f_kp1)
            u_c = 0.5 * (u_k + u_kp1)
            
            # Normalize midpoint quaternion
            x_c[3:7] = x_c[3:7] / np.linalg.norm(x_c[3:7])
            
            # Dynamics at midpoint
            f_c = dynamics.dynamics(x_c, u_c, B_k)
            
            # Hermite-Simpson defect
            defect = x_kp1 - x_k - (dt / 6) * (f_k + 4 * f_c + f_kp1)
            
            defects.extend(defect.tolist())
        
        return np.array(defects)
    
    def _create_bounds(
        self,
        x0: NDArray[np.float64],
        N: int,
        n_x: int,
        n_u: int,
        u_max: NDArray[np.float64]
    ) -> list:
        """Create variable bounds."""
        bounds = []
        
        for k in range(N):
            # State bounds (mostly unbounded, but fix initial state)
            for i in range(n_x):
                if k == 0:
                    # Fix initial state
                    bounds.append((x0[i], x0[i]))
                else:
                    bounds.append((None, None))
            
            # Control bounds
            for i in range(n_u):
                bounds.append((-u_max[i], u_max[i]))
        
        # Final state (unbounded)
        for i in range(n_x):
            bounds.append((None, None))
        
        return bounds
    
    def _extract_solution(
        self,
        z: NDArray[np.float64],
        N: int,
        n_x: int,
        n_u: int
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Extract states and controls from decision variable vector."""
        states = np.zeros((N + 1, n_x))
        controls = np.zeros((N, n_u))
        
        for k in range(N):
            idx_x = k * (n_x + n_u)
            idx_u = idx_x + n_x
            
            states[k] = z[idx_x:idx_x + n_x]
            controls[k] = z[idx_u:idx_u + n_u]
        
        # Final state
        idx_xN = N * (n_x + n_u)
        states[N] = z[idx_xN:idx_xN + n_x]
        
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
    
    def _quaternion_error_vec(
        self,
        q: NDArray[np.float64],
        q_goal: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute 3-vector quaternion error. Uses scalar-first: q = [w, x, y, z]."""
        # Normalize
        q = q / np.linalg.norm(q)
        q_goal = q_goal / np.linalg.norm(q_goal)
        
        # q_err = q_goal^(-1) * q
        # For scalar-first: conjugate is [w, -x, -y, -z]
        w0, x0, y0, z0 = q_goal
        w1, x1, y1, z1 = q
        
        q_err = np.array([
            w0*w1 + x0*x1 + y0*y1 + z0*z1,   # w
            w0*x1 - x0*w1 - y0*z1 + z0*y1,   # x  
            w0*y1 + x0*z1 - y0*w1 - z0*x1,   # y
            w0*z1 - x0*y1 + y0*x1 - z0*w1    # z
        ])
        
        # Return vector part scaled by 2 (small angle approximation)
        # Vector part is [1:4] for scalar-first
        return 2.0 * q_err[1:4] * np.sign(q_err[0])
    
    @staticmethod
    def _quaternion_angle(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> float:
        """Compute angle between two quaternions in radians."""
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        dot = np.abs(np.dot(q1, q2))
        dot = np.clip(dot, -1.0, 1.0)
        return 2.0 * np.arccos(dot)
