"""
Pseudospectral (Gauss-Lobatto) Trajectory Planner.

This planner uses Legendre-Gauss-Lobatto (LGL) collocation points and spectral
differentiation matrices for trajectory optimization. Pseudospectral methods
achieve spectral (exponential) convergence for smooth problems.

These methods have been used in several NASA missions and are the basis for
tools like GPOPS-II and PSOPT.

References:
    Ross & Fahroo (2004) "Pseudospectral Knotting Methods for Solving Optimal Control Problems"
    Garg et al. (2010) "A Unified Framework for the Numerical Solution of Optimal Control Problems"
    Fahroo & Ross (2002) "Direct Trajectory Optimization by a Chebyshev Pseudospectral Method"
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.special import legendre

from .base_planner import BasePlanner, PlannerResult, PlannerConfig, DynamicsModel


@dataclass
class PseudospectralConfig(PlannerConfig):
    """Configuration for pseudospectral planner."""
    
    # Number of collocation points (more = higher accuracy but slower)
    n_nodes: int = 15
    
    # NLP solver settings
    method: str = "SLSQP"
    ftol: float = 1e-6
    
    # Dynamics constraint tolerance
    dynamics_tol: float = 1e-4


class PseudospectralPlanner(BasePlanner):
    """
    Pseudospectral trajectory planner using Legendre-Gauss-Lobatto collocation.
    
    The problem is transcribed by:
    1. Transforming time to τ ∈ [-1, 1]
    2. Placing collocation points at LGL nodes
    3. Approximating states as Lagrange polynomials
    4. Using differentiation matrix for derivative approximation
    5. Enforcing dynamics at collocation points
    
    Key property: For smooth problems, the error decreases exponentially
    with the number of nodes (spectral convergence).
    """
    
    def __init__(self, config: Optional[PseudospectralConfig] = None):
        """Initialize pseudospectral planner."""
        if config is None:
            config = PseudospectralConfig()
        super().__init__(config)
        self.config: PseudospectralConfig = config
        self._name = "Pseudospectral"
        
        # Pre-compute LGL nodes and differentiation matrix
        self._nodes, self._weights = self._compute_lgl_nodes(config.n_nodes)
        self._D = self._compute_differentiation_matrix(self._nodes)
    
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
        Solve trajectory optimization using pseudospectral method.
        
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
        N = self.config.n_nodes
        T = self.config.horizon
        
        # Create dynamics model
        dynamics = DynamicsModel(
            J_inertia=J_inertia,
            rw_axes=np.eye(3)[:, :n_rw] if n_rw > 0 else None,
            n_mtq=n_u - n_rw,
            has_rw=n_rw > 0
        )
        
        # Transform LGL nodes from [-1,1] to [0,T]
        times = 0.5 * T * (self._nodes + 1)
        
        # Decision variables: [X (N x n_x), U (N x n_u)]
        n_vars = N * (n_x + n_u)
        
        # Initialize
        z0 = self._initialize(x0, x_goal, N, n_x, n_u)
        
        # Cost function
        def cost(z):
            return self._compute_cost(z, x_goal, N, n_x, n_u, T)
        
        # Dynamics constraint
        def dynamics_constraint(z):
            return self._compute_dynamics_residual(z, N, n_x, n_u, T, dynamics, B_field)
        
        # Initial state constraint
        def initial_constraint(z):
            X = z[:N * n_x].reshape(N, n_x)
            return X[0] - x0
        
        # Bounds
        bounds = self._create_bounds(N, n_x, n_u, u_max)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': initial_constraint},
            {'type': 'eq', 'fun': dynamics_constraint},
        ]
        
        # Solve
        result = minimize(
            cost,
            z0,
            method=self.config.method,
            bounds=bounds,
            constraints=constraints,
            options={
                'ftol': self.config.ftol,
                'maxiter': self.config.max_iterations,
                'disp': self.config.verbose
            }
        )
        
        solve_time = time.perf_counter() - start_time
        
        # Extract solution
        X = result.x[:N * n_x].reshape(N, n_x)
        U = result.x[N * n_x:].reshape(N, n_u)
        
        # Normalize quaternions
        for k in range(N):
            X[k, 3:7] = X[k, 3:7] / np.linalg.norm(X[k, 3:7])
        
        # Interpolate to uniform time grid for output
        n_output = int(np.ceil(T / self.config.dt)) + 1
        times_uniform = np.linspace(0, T, n_output)
        states_uniform, controls_uniform = self._interpolate_to_uniform(
            times, X, U, times_uniform
        )
        
        # Compute errors
        q_final = states_uniform[-1, 3:7]
        q_goal = x_goal[3:7]
        angle_error = self._quaternion_angle(q_final, q_goal)
        omega_error = np.linalg.norm(states_uniform[-1, :3] - x_goal[:3])
        
        return PlannerResult(
            times=times_uniform,
            states=states_uniform,
            controls=controls_uniform,
            solve_time=solve_time,
            converged=result.success,
            iterations=result.nit if hasattr(result, 'nit') else -1,
            final_cost=result.fun,
            max_constraint_violation=np.max(np.abs(dynamics_constraint(result.x))),
            solver_info={
                "final_angle_error_deg": np.degrees(angle_error),
                "final_omega_error": omega_error,
                "n_nodes": N,
                "nlp_status": result.message,
            }
        )
    
    def _compute_lgl_nodes(self, N: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute Legendre-Gauss-Lobatto nodes and weights.
        
        LGL nodes are the roots of (1-τ²)P'_{N-1}(τ) plus the endpoints ±1.
        """
        if N < 2:
            raise ValueError("Need at least 2 nodes")
        
        # Use eigenvalue method to find interior nodes
        # LGL nodes are eigenvalues of a symmetric tridiagonal matrix
        n = N - 2  # Number of interior nodes
        
        if n == 0:
            nodes = np.array([-1.0, 1.0])
        else:
            # Build companion matrix for Jacobi polynomial
            i = np.arange(1, n + 1, dtype=np.float64)
            beta = 0.5 / np.sqrt(1.0 - (2.0 * i)**(-2))
            
            # Symmetric tridiagonal matrix
            T = np.diag(beta[:-1], 1) + np.diag(beta[:-1], -1)
            
            # Eigenvalues are the interior nodes
            interior_nodes = np.linalg.eigvalsh(T)
            
            # Add endpoints
            nodes = np.concatenate([[-1.0], np.sort(interior_nodes), [1.0]])
        
        # Compute weights using the formula:
        # w_k = 2 / (N(N-1)[P_{N-1}(τ_k)]²)
        P_Nm1 = legendre(N - 1)
        P_vals = np.array([float(P_Nm1(x)) for x in nodes])  # Evaluate as float
        weights = 2.0 / (N * (N - 1) * P_vals**2)
        
        return nodes, weights
    
    def _compute_differentiation_matrix(self, nodes: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute spectral differentiation matrix.
        
        D[i,j] gives the derivative of the j-th Lagrange polynomial at node i.
        """
        N = len(nodes)
        D = np.zeros((N, N))
        
        # Barycentric weights
        c = np.ones(N)
        c[0] = 2.0
        c[-1] = 2.0
        c *= (-1.0) ** np.arange(N)
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    D[i, j] = c[j] / (c[i] * (nodes[i] - nodes[j]))
                    D[i, i] -= D[i, j]
        
        return D
    
    def _initialize(
        self,
        x0: NDArray[np.float64],
        x_goal: NDArray[np.float64],
        N: int,
        n_x: int,
        n_u: int
    ) -> NDArray[np.float64]:
        """Initialize decision variables."""
        z = np.zeros(N * (n_x + n_u))
        
        # Linear interpolation for states
        for k in range(N):
            alpha = (self._nodes[k] + 1) / 2  # Map [-1,1] to [0,1]
            x_k = (1 - alpha) * x0 + alpha * x_goal
            x_k[3:7] = x_k[3:7] / np.linalg.norm(x_k[3:7])
            z[k * n_x:(k + 1) * n_x] = x_k
        
        # Zero control
        # (already zeros)
        
        return z
    
    def _compute_cost(
        self,
        z: NDArray[np.float64],
        x_goal: NDArray[np.float64],
        N: int,
        n_x: int,
        n_u: int,
        T: float
    ) -> float:
        """Compute cost using Gauss-Lobatto quadrature."""
        X = z[:N * n_x].reshape(N, n_x)
        U = z[N * n_x:].reshape(N, n_u)
        
        cost = 0.0
        
        # Running cost (quadrature)
        for k in range(N):
            # Control cost
            cost += self._weights[k] * self.config.R_control * np.sum(U[k]**2)
            
            # Angular velocity cost
            cost += self._weights[k] * self.config.Q_angular_vel * np.sum(X[k, :3]**2)
        
        # Scale by time transformation
        cost *= 0.5 * T
        
        # Terminal cost
        x_N = X[-1]
        q_err = self._quaternion_error_vec(x_N[3:7], x_goal[3:7])
        cost += self.config.Q_attitude_terminal * np.sum(q_err**2)
        
        omega_err = x_N[:3] - x_goal[:3]
        cost += self.config.Q_angular_vel_terminal * np.sum(omega_err**2)
        
        return cost
    
    def _compute_dynamics_residual(
        self,
        z: NDArray[np.float64],
        N: int,
        n_x: int,
        n_u: int,
        T: float,
        dynamics: DynamicsModel,
        B_field: Optional[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        """
        Compute dynamics residual using spectral differentiation.
        
        Residual = D @ X - (T/2) * f(X, U)
        """
        X = z[:N * n_x].reshape(N, n_x)
        U = z[N * n_x:].reshape(N, n_u)
        
        # Compute derivative approximation: D @ X
        dX_dtau = self._D @ X  # Shape: (N, n_x)
        
        # Compute dynamics at each node
        F = np.zeros((N, n_x))
        for k in range(N):
            B_k = self._get_B_at_k(B_field, k, N)
            F[k] = dynamics.dynamics(X[k], U[k], B_k)
        
        # Residual: dX/dτ = (T/2) * f(X, U)
        # => D @ X - (T/2) * f = 0
        residual = dX_dtau - 0.5 * T * F
        
        return residual.flatten()
    
    def _create_bounds(
        self,
        N: int,
        n_x: int,
        n_u: int,
        u_max: NDArray[np.float64]
    ) -> list:
        """Create variable bounds."""
        bounds = []
        
        # State bounds (unbounded)
        for _ in range(N * n_x):
            bounds.append((None, None))
        
        # Control bounds
        for k in range(N):
            for i in range(n_u):
                bounds.append((-u_max[i], u_max[i]))
        
        return bounds
    
    def _interpolate_to_uniform(
        self,
        t_nodes: NDArray[np.float64],
        X_nodes: NDArray[np.float64],
        U_nodes: NDArray[np.float64],
        t_uniform: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Interpolate solution to uniform time grid."""
        from scipy.interpolate import interp1d
        
        n_x = X_nodes.shape[1]
        n_u = U_nodes.shape[1]
        n_out = len(t_uniform)
        
        # Interpolate states
        X_uniform = np.zeros((n_out, n_x))
        for i in range(n_x):
            interp = interp1d(t_nodes, X_nodes[:, i], kind='cubic', fill_value='extrapolate')
            X_uniform[:, i] = interp(t_uniform)
        
        # Normalize quaternions
        for k in range(n_out):
            X_uniform[k, 3:7] = X_uniform[k, 3:7] / np.linalg.norm(X_uniform[k, 3:7])
        
        # Interpolate controls
        U_uniform = np.zeros((n_out - 1, n_u))
        for i in range(n_u):
            interp = interp1d(t_nodes, U_nodes[:, i], kind='linear', fill_value='extrapolate')
            U_uniform[:, i] = interp(t_uniform[:-1])
        
        return X_uniform, U_uniform
    
    def _get_B_at_k(
        self,
        B_field: Optional[NDArray[np.float64]],
        k: int,
        N: int
    ) -> Optional[NDArray[np.float64]]:
        """Get magnetic field at node k."""
        if B_field is None:
            return None
        if B_field.ndim == 1:
            return B_field
        # Interpolate based on node position
        idx = int((self._nodes[k] + 1) / 2 * (len(B_field) - 1))
        return B_field[min(idx, len(B_field) - 1)]
    
    def _quaternion_error_vec(
        self,
        q: NDArray[np.float64],
        q_goal: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute 3-vector quaternion error. Uses scalar-first: q = [w, x, y, z]."""
        q = q / np.linalg.norm(q)
        q_goal = q_goal / np.linalg.norm(q_goal)
        
        # q_err = q_goal^(-1) * q
        w0, x0, y0, z0 = q_goal
        w1, x1, y1, z1 = q
        
        q_err = np.array([
            w0*w1 + x0*x1 + y0*y1 + z0*z1,   # w
            w0*x1 - x0*w1 - y0*z1 + z0*y1,   # x  
            w0*y1 + x0*z1 - y0*w1 - z0*x1,   # y
            w0*z1 - x0*y1 + y0*x1 - z0*w1    # z
        ])
        
        return 2.0 * q_err[1:4] * np.sign(q_err[0])
    
    @staticmethod
    def _quaternion_angle(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> float:
        """Compute angle between two quaternions."""
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        dot = np.abs(np.dot(q1, q2))
        return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))
