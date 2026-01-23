"""
Base class for trajectory planners.

All planners must implement this interface for consistent comparison.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np
from numpy.typing import NDArray
import time


@dataclass
class PlannerResult:
    """Container for trajectory planner output."""
    
    # Core trajectory data
    times: NDArray[np.float64]       # (N,) time array
    states: NDArray[np.float64]      # (N, n_states) state trajectory
    controls: NDArray[np.float64]    # (N-1, n_controls) or (N, n_controls) control trajectory
    
    # Performance metrics
    solve_time: float                # Wall-clock solve time in seconds
    converged: bool                  # Whether solver converged
    iterations: int                  # Number of iterations used
    final_cost: float                # Final objective value
    
    # Constraint info
    max_constraint_violation: float  # Maximum constraint violation
    constraint_violations: Optional[NDArray[np.float64]] = None
    
    # Solver-specific info
    solver_info: Dict[str, Any] = field(default_factory=dict)
    
    # Optional feedback gains (if computed)
    gains: Optional[NDArray[np.float64]] = None  # (N-1, n_controls, n_states) gain matrices K
    
    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return len(self.times)
    
    @property
    def final_state(self) -> NDArray[np.float64]:
        """Final state of trajectory."""
        return self.states[-1]
    
    @property
    def dt(self) -> float:
        """Average time step."""
        return (self.times[-1] - self.times[0]) / (len(self.times) - 1)


@dataclass
class PlannerConfig:
    """Configuration for trajectory planners."""
    
    # Time discretization
    dt: float = 1.0                  # Time step in seconds
    horizon: float = 60.0            # Planning horizon in seconds
    
    # Convergence settings
    max_iterations: int = 100
    cost_tolerance: float = 1e-4
    constraint_tolerance: float = 1e-4
    
    # Cost weights (generic - planners may interpret differently)
    Q_attitude: float = 1e4          # Attitude error weight
    Q_angular_vel: float = 1e2       # Angular velocity weight  
    Q_attitude_terminal: float = 1e6 # Terminal attitude weight
    Q_angular_vel_terminal: float = 1e4  # Terminal angular velocity weight
    R_control: float = 1e0           # Control effort weight
    
    # Constraint settings
    enforce_control_limits: bool = True
    control_limit_margin: float = 0.9  # Use 90% of max control
    
    # Verbosity
    verbose: bool = False


class BasePlanner(ABC):
    """
    Abstract base class for trajectory planners.
    
    All planner implementations must inherit from this class and implement
    the solve() method.
    """
    
    def __init__(self, config: Optional[PlannerConfig] = None):
        """
        Initialize planner with configuration.
        
        Args:
            config: Planner configuration. If None, uses defaults.
        """
        self.config = config or PlannerConfig()
        self._name = self.__class__.__name__
    
    @property
    def name(self) -> str:
        """Return planner name for identification."""
        return self._name
    
    @abstractmethod
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
        Solve trajectory optimization problem.
        
        Args:
            x0: Initial state [omega (3), quaternion (4), rw_momentum (n_rw)]
            x_goal: Goal state (same format as x0)
            J_inertia: Spacecraft inertia matrix (3x3)
            u_max: Maximum control for each actuator
            B_field: Magnetic field in body frame (for MTQ), (N, 3) or (3,)
            **kwargs: Additional solver-specific parameters
            
        Returns:
            PlannerResult containing the solution trajectory and metrics
        """
        pass
    
    def warm_start(self, x_init: NDArray[np.float64], u_init: NDArray[np.float64]) -> None:
        """
        Provide warm start for solver (optional).
        
        Args:
            x_init: Initial state trajectory guess
            u_init: Initial control trajectory guess
        """
        pass
    
    def _time_solve(self, solve_func, *args, **kwargs) -> tuple:
        """
        Time a solve function and return result with timing.
        
        Returns:
            (result, solve_time)
        """
        start = time.perf_counter()
        result = solve_func(*args, **kwargs)
        solve_time = time.perf_counter() - start
        return result, solve_time


class DynamicsModel:
    """
    Spacecraft attitude dynamics model.
    
    State: x = [omega (3), q (4), h_rw (n_rw)]
    Control: u = [tau_mtq (3) or m_dipole (3), tau_rw (n_rw)]
    
    Dynamics:
        J * omega_dot = -omega x (J*omega + A_rw*h_rw) + tau_ext + A_rw*tau_rw + tau_mtq
        q_dot = 0.5 * Omega(omega) * q
        h_rw_dot = tau_rw
        
    where tau_mtq = m_dipole x B for magnetorquers
    """
    
    def __init__(
        self,
        J_inertia: NDArray[np.float64],
        rw_axes: Optional[NDArray[np.float64]] = None,
        n_mtq: int = 3,
        has_rw: bool = True
    ):
        """
        Initialize dynamics model.
        
        Args:
            J_inertia: 3x3 inertia matrix
            rw_axes: (n_rw, 3) array of reaction wheel axes (unit vectors)
            n_mtq: Number of magnetorquers (typically 3)
            has_rw: Whether satellite has reaction wheels
        """
        self.J = np.array(J_inertia, dtype=np.float64)
        self.J_inv = np.linalg.inv(self.J)
        self.n_mtq = n_mtq
        self.has_rw = has_rw
        
        if has_rw and rw_axes is not None:
            self.A_rw = np.array(rw_axes, dtype=np.float64).T  # (3, n_rw)
            self.n_rw = self.A_rw.shape[1]
        elif has_rw:
            # Default: 3 orthogonal reaction wheels
            self.A_rw = np.eye(3)
            self.n_rw = 3
        else:
            self.A_rw = np.zeros((3, 0))
            self.n_rw = 0
        
        # State and control dimensions
        self.n_omega = 3
        self.n_quat = 4
        self.n_states = self.n_omega + self.n_quat + self.n_rw
        self.n_controls = self.n_mtq + self.n_rw
    
    def dynamics(
        self,
        x: NDArray[np.float64],
        u: NDArray[np.float64],
        B: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """
        Compute state derivative.
        
        Args:
            x: State vector
            u: Control vector [mtq_dipole (3), rw_torque (n_rw)]
            B: Magnetic field in body frame (required for MTQ)
            
        Returns:
            State derivative dx/dt
        """
        omega = x[:3]
        q = x[3:7]
        h_rw = x[7:7+self.n_rw] if self.has_rw else np.zeros(0)
        
        # Extract controls
        m_dipole = u[:self.n_mtq]
        tau_rw = u[self.n_mtq:self.n_mtq+self.n_rw] if self.has_rw else np.zeros(0)
        
        # MTQ torque (if magnetic field provided)
        if B is not None and self.n_mtq > 0:
            tau_mtq = np.cross(m_dipole, B)
        else:
            tau_mtq = np.zeros(3)
        
        # Total angular momentum
        H_total = self.J @ omega
        if self.has_rw:
            H_total += self.A_rw @ h_rw
        
        # Angular velocity dynamics
        tau_total = tau_mtq
        if self.has_rw:
            tau_total = tau_total + self.A_rw @ tau_rw
        omega_dot = self.J_inv @ (-np.cross(omega, H_total) + tau_total)
        
        # Quaternion kinematics
        q_dot = 0.5 * self._omega_matrix(omega) @ q
        
        # Reaction wheel dynamics
        h_rw_dot = tau_rw if self.has_rw else np.zeros(0)
        
        return np.concatenate([omega_dot, q_dot, h_rw_dot])
    
    def linearize(
        self,
        x: NDArray[np.float64],
        u: NDArray[np.float64],
        B: Optional[NDArray[np.float64]] = None,
        eps: float = 1e-6
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Linearize dynamics around (x, u) using finite differences.
        
        Returns:
            (A, B) matrices for linear system dx = A*x + B*u
        """
        n_x = len(x)
        n_u = len(u)
        
        f0 = self.dynamics(x, u, B)
        
        # State Jacobian A
        A = np.zeros((n_x, n_x))
        for i in range(n_x):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            A[:, i] = (self.dynamics(x_plus, u, B) - self.dynamics(x_minus, u, B)) / (2 * eps)
        
        # Control Jacobian B
        B_mat = np.zeros((n_x, n_u))
        for i in range(n_u):
            u_plus = u.copy()
            u_plus[i] += eps
            u_minus = u.copy()
            u_minus[i] -= eps
            B_mat[:, i] = (self.dynamics(x, u_plus, B) - self.dynamics(x, u_minus, B)) / (2 * eps)
        
        return A, B_mat
    
    def discretize(
        self,
        x: NDArray[np.float64],
        u: NDArray[np.float64],
        dt: float,
        B: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """
        Integrate dynamics for one timestep using RK4.
        
        Returns:
            Next state x_{k+1}
        """
        k1 = self.dynamics(x, u, B)
        k2 = self.dynamics(x + 0.5*dt*k1, u, B)
        k3 = self.dynamics(x + 0.5*dt*k2, u, B)
        k4 = self.dynamics(x + dt*k3, u, B)
        
        x_next = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Normalize quaternion
        x_next[3:7] = x_next[3:7] / np.linalg.norm(x_next[3:7])
        
        return x_next
    
    def discretize_jacobians(
        self,
        x: NDArray[np.float64],
        u: NDArray[np.float64],
        dt: float,
        B: Optional[NDArray[np.float64]] = None,
        eps: float = 1e-6
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute Jacobians of discrete dynamics using finite differences.
        
        Returns:
            (A_d, B_d) discrete Jacobians
        """
        n_x = len(x)
        n_u = len(u)
        
        # State Jacobian
        A_d = np.zeros((n_x, n_x))
        for i in range(n_x):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            A_d[:, i] = (self.discretize(x_plus, u, dt, B) - self.discretize(x_minus, u, dt, B)) / (2 * eps)
        
        # Control Jacobian
        B_d = np.zeros((n_x, n_u))
        for i in range(n_u):
            u_plus = u.copy()
            u_plus[i] += eps
            u_minus = u.copy()
            u_minus[i] -= eps
            B_d[:, i] = (self.discretize(x, u_plus, dt, B) - self.discretize(x, u_minus, dt, B)) / (2 * eps)
        
        return A_d, B_d
    
    @staticmethod
    def _omega_matrix(omega: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Quaternion rate matrix.
        
        q_dot = 0.5 * Omega(omega) * q
        """
        wx, wy, wz = omega
        return np.array([
            [0, wz, -wy, wx],
            [-wz, 0, wx, wy],
            [wy, -wx, 0, wz],
            [-wx, -wy, -wz, 0]
        ])
    
    @staticmethod
    def quaternion_error(q: NDArray[np.float64], q_goal: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute quaternion error q_err = q_goal^(-1) * q.
        
        Uses scalar-first convention: q = [w, x, y, z]
        
        Returns 3-vector (reduced attitude error) for optimization.
        """
        # Extract components (scalar-first: w=q[0], v=q[1:4])
        w0, x0, y0, z0 = q_goal
        w1, x1, y1, z1 = q
        
        # Quaternion conjugate of q_goal: [w, -x, -y, -z]
        # q_err = q_goal^(-1) * q (quaternion multiplication)
        q_err = np.array([
            w0*w1 + x0*x1 + y0*y1 + z0*z1,   # w
            w0*x1 - x0*w1 - y0*z1 + z0*y1,   # x  
            w0*y1 + x0*z1 - y0*w1 - z0*x1,   # y
            w0*z1 - x0*y1 + y0*x1 - z0*w1    # z
        ])
        
        # Return vector part (small angle approximation: 2*q_vec ≈ rotation vector)
        # Vector part is q_err[1:4] for scalar-first
        return 2.0 * q_err[1:4]
    
    @staticmethod
    def quaternion_angle(q: NDArray[np.float64], q_goal: NDArray[np.float64]) -> float:
        """Compute angle between two quaternions in radians."""
        dot = np.abs(np.dot(q, q_goal))
        dot = np.clip(dot, -1.0, 1.0)
        return 2.0 * np.arccos(dot)
