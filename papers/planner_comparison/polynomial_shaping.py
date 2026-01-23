"""
Polynomial Trajectory Shaping Planner.

This planner generates smooth attitude trajectories using polynomial interpolation.
It's commonly used for offline trajectory generation where smoothness and
predictability are more important than optimality.

Methods supported:
- 5th order polynomial: satisfies position, velocity, acceleration at endpoints
- 7th order polynomial: adds jerk continuity at endpoints
- Bezier curves: alternative smooth interpolation

The trajectory is generated in "attitude angle space" (eigenaxis rotation)
and then converted to quaternion representation.

References:
    Junkins, J.L., Turner, J.D. "Optimal Spacecraft Rotational Maneuvers"
    Craig, J.J. "Introduction to Robotics", Chapter 7 (Trajectory Generation)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Literal
from dataclasses import dataclass
from scipy.interpolate import BPoly

from .base_planner import BasePlanner, PlannerResult, PlannerConfig


@dataclass
class PolynomialConfig(PlannerConfig):
    """Configuration for polynomial trajectory planner."""
    
    # Polynomial order
    poly_order: Literal[5, 7] = 5    # 5th or 7th order polynomial
    
    # Boundary conditions
    zero_initial_velocity: bool = True
    zero_final_velocity: bool = True
    zero_initial_acceleration: bool = True
    zero_final_acceleration: bool = True
    
    # For 7th order, also specify jerk
    zero_initial_jerk: bool = True
    zero_final_jerk: bool = True
    
    # Maneuver time (if None, computed from constraints)
    maneuver_time: Optional[float] = None
    
    # Constraints for time computation (if maneuver_time is None)
    omega_max: float = 0.05          # Max angular velocity (rad/s)
    alpha_max: float = 0.01          # Max angular acceleration (rad/s^2)
    
    # Time margin factor
    time_margin: float = 1.2         # 20% margin for polynomial trajectories


class PolynomialShapingPlanner(BasePlanner):
    """
    Polynomial trajectory shaping for attitude maneuvers.
    
    This planner:
    1. Computes the eigenaxis rotation from initial to goal attitude
    2. Fits a polynomial to the rotation angle profile with boundary conditions
    3. Generates smooth angular velocity and acceleration profiles
    4. Converts back to quaternion representation
    5. Computes feedforward torques
    
    5th order polynomial satisfies:
    - θ(0), θ(T) - initial and final angle
    - θ'(0), θ'(T) - initial and final angular velocity
    - θ''(0), θ''(T) - initial and final angular acceleration
    
    7th order polynomial additionally satisfies:
    - θ'''(0), θ'''(T) - initial and final jerk
    """
    
    def __init__(self, config: Optional[PolynomialConfig] = None):
        """Initialize planner with configuration."""
        if config is None:
            config = PolynomialConfig()
        super().__init__(config)
        self.config: PolynomialConfig = config
        self._name = f"Polynomial-{config.poly_order}"
    
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
        Generate polynomial-shaped trajectory.
        
        Args:
            x0: Initial state [omega (3), quaternion (4), rw_momentum (n_rw)]
            x_goal: Goal state
            J_inertia: Spacecraft inertia matrix (3x3)
            u_max: Maximum control for each actuator
            B_field: Magnetic field (not used)
            
        Returns:
            PlannerResult with trajectory and metrics
        """
        import time
        start_time = time.perf_counter()
        
        # Extract quaternions and normalize
        q0 = x0[3:7].copy()
        q_goal = x_goal[3:7].copy()
        omega0 = x0[:3].copy()
        omega_goal = x_goal[:3].copy()
        
        q0 = q0 / np.linalg.norm(q0)
        q_goal = q_goal / np.linalg.norm(q_goal)
        
        n_rw = len(x0) - 7
        has_rw = n_rw > 0
        
        # Compute eigenaxis and angle
        eigenaxis, total_angle = self._compute_eigenaxis(q0, q_goal)
        
        # Compute maneuver time
        if self.config.maneuver_time is not None:
            T = self.config.maneuver_time
        else:
            T = self._compute_maneuver_time(total_angle)
        
        # Use horizon if specified and longer
        if self.config.horizon > T:
            T = self.config.horizon
        
        # Generate time array
        N = int(np.ceil(T / self.config.dt)) + 1
        times = np.linspace(0, T, N)
        dt = times[1] - times[0] if N > 1 else self.config.dt
        
        # Compute polynomial coefficients
        # Project initial/final omega onto eigenaxis
        omega0_proj = np.dot(omega0, eigenaxis) if not self.config.zero_initial_velocity else 0.0
        omega_goal_proj = np.dot(omega_goal, eigenaxis) if not self.config.zero_final_velocity else 0.0
        
        if self.config.poly_order == 5:
            coeffs = self._compute_5th_order_coeffs(
                theta_0=0.0,
                theta_f=total_angle,
                omega_0=omega0_proj,
                omega_f=omega_goal_proj,
                alpha_0=0.0 if self.config.zero_initial_acceleration else None,
                alpha_f=0.0 if self.config.zero_final_acceleration else None,
                T=T
            )
        else:  # 7th order
            coeffs = self._compute_7th_order_coeffs(
                theta_0=0.0,
                theta_f=total_angle,
                omega_0=omega0_proj,
                omega_f=omega_goal_proj,
                alpha_0=0.0 if self.config.zero_initial_acceleration else None,
                alpha_f=0.0 if self.config.zero_final_acceleration else None,
                jerk_0=0.0 if self.config.zero_initial_jerk else None,
                jerk_f=0.0 if self.config.zero_final_jerk else None,
                T=T
            )
        
        # Evaluate polynomial at each time step
        theta_profile = np.zeros(N)
        omega_profile_mag = np.zeros(N)
        alpha_profile_mag = np.zeros(N)
        
        for i, t in enumerate(times):
            tau = t / T if T > 0 else 0.0  # Normalized time [0, 1]
            tau = np.clip(tau, 0.0, 1.0)
            
            theta_profile[i] = self._eval_polynomial(coeffs, tau, total_angle)
            omega_profile_mag[i] = self._eval_polynomial_deriv(coeffs, tau, total_angle, T, order=1)
            alpha_profile_mag[i] = self._eval_polynomial_deriv(coeffs, tau, total_angle, T, order=2)
        
        # Convert to 3D angular velocity
        omega_profile = np.outer(omega_profile_mag, eigenaxis)
        
        # Integrate quaternions
        states = np.zeros((N, len(x0)))
        states[0] = x0.copy()
        states[0, 3:7] = q0
        
        for k in range(N - 1):
            # Use the polynomial angle directly to compute quaternion
            theta_k = theta_profile[k + 1]
            q_k = self._angle_axis_to_quaternion(theta_k, eigenaxis, q0)
            
            states[k+1, :3] = omega_profile[k+1]
            states[k+1, 3:7] = q_k
            
            if has_rw:
                states[k+1, 7:] = states[k, 7:]
        
        # Compute feedforward controls
        controls = self._compute_inverse_dynamics(
            states, omega_profile, alpha_profile_mag, eigenaxis,
            J_inertia, u_max, dt, n_rw
        )
        
        solve_time = time.perf_counter() - start_time
        
        # Compute final errors
        q_final = states[-1, 3:7]
        omega_final = states[-1, :3]
        angle_error = self._quaternion_angle(q_final, q_goal)
        omega_error = np.linalg.norm(omega_final - omega_goal)
        
        # Check constraint violations
        max_omega = np.max(np.abs(omega_profile_mag))
        max_alpha = np.max(np.abs(alpha_profile_mag))
        constraint_violation = max(
            max_omega / self.config.omega_max - 1.0,
            max_alpha / self.config.alpha_max - 1.0,
            0.0
        )
        
        return PlannerResult(
            times=times,
            states=states,
            controls=controls,
            solve_time=solve_time,
            converged=True,
            iterations=1,
            final_cost=angle_error + omega_error,
            max_constraint_violation=constraint_violation,
            solver_info={
                "eigenaxis": eigenaxis.tolist(),
                "rotation_angle_deg": np.degrees(total_angle),
                "maneuver_time": T,
                "poly_order": self.config.poly_order,
                "max_omega": max_omega,
                "max_alpha": max_alpha,
                "final_angle_error_deg": np.degrees(angle_error),
                "final_omega_error": omega_error,
                "coefficients": coeffs.tolist(),
            }
        )
    
    def _compute_maneuver_time(self, angle: float) -> float:
        """
        Compute minimum maneuver time satisfying velocity and acceleration constraints.
        
        For a 5th order polynomial rest-to-rest maneuver:
        - Max velocity occurs at t = T/2
        - Max acceleration occurs at t ≈ 0.21T and t ≈ 0.79T
        
        Approximate constraints:
        - omega_max ≈ 1.875 * angle / T
        - alpha_max ≈ 5.77 * angle / T^2
        """
        # Time from velocity constraint
        T_omega = 1.875 * angle / self.config.omega_max if self.config.omega_max > 0 else 0
        
        # Time from acceleration constraint
        T_alpha = np.sqrt(5.77 * angle / self.config.alpha_max) if self.config.alpha_max > 0 else 0
        
        T = max(T_omega, T_alpha, 1.0)  # At least 1 second
        
        return T * self.config.time_margin
    
    def _compute_5th_order_coeffs(
        self,
        theta_0: float,
        theta_f: float,
        omega_0: float,
        omega_f: float,
        alpha_0: Optional[float],
        alpha_f: Optional[float],
        T: float
    ) -> NDArray[np.float64]:
        """
        Compute 5th order polynomial coefficients.
        
        θ(τ) = a0 + a1*τ + a2*τ^2 + a3*τ^3 + a4*τ^4 + a5*τ^5
        
        where τ = t/T is normalized time.
        
        Boundary conditions:
        - θ(0) = theta_0
        - θ(1) = theta_f
        - θ'(0) = omega_0 * T
        - θ'(1) = omega_f * T
        - θ''(0) = alpha_0 * T^2
        - θ''(1) = alpha_f * T^2
        """
        # Default to zero if not specified
        alpha_0 = alpha_0 if alpha_0 is not None else 0.0
        alpha_f = alpha_f if alpha_f is not None else 0.0
        
        # Scale derivatives by T
        v0 = omega_0 * T
        vf = omega_f * T
        a0 = alpha_0 * T * T
        af = alpha_f * T * T
        
        # Solve for coefficients
        # θ(0) = a0 = theta_0
        # θ(1) = a0 + a1 + a2 + a3 + a4 + a5 = theta_f
        # θ'(0) = a1 = v0
        # θ'(1) = a1 + 2*a2 + 3*a3 + 4*a4 + 5*a5 = vf
        # θ''(0) = 2*a2 = a0
        # θ''(1) = 2*a2 + 6*a3 + 12*a4 + 20*a5 = af
        
        c0 = theta_0
        c1 = v0
        c2 = a0 / 2.0
        
        # Remaining coefficients from linear system
        # [1,  1,  1 ] [a3]   [theta_f - c0 - c1 - c2]
        # [3,  4,  5 ] [a4] = [vf - c1 - 2*c2        ]
        # [6, 12, 20 ] [a5]   [af - 2*c2             ]
        
        A = np.array([
            [1, 1, 1],
            [3, 4, 5],
            [6, 12, 20]
        ], dtype=np.float64)
        
        b = np.array([
            theta_f - c0 - c1 - c2,
            vf - c1 - 2*c2,
            af - 2*c2
        ])
        
        x = np.linalg.solve(A, b)
        
        return np.array([c0, c1, c2, x[0], x[1], x[2]])
    
    def _compute_7th_order_coeffs(
        self,
        theta_0: float,
        theta_f: float,
        omega_0: float,
        omega_f: float,
        alpha_0: Optional[float],
        alpha_f: Optional[float],
        jerk_0: Optional[float],
        jerk_f: Optional[float],
        T: float
    ) -> NDArray[np.float64]:
        """
        Compute 7th order polynomial coefficients.
        
        θ(τ) = a0 + a1*τ + a2*τ^2 + a3*τ^3 + a4*τ^4 + a5*τ^5 + a6*τ^6 + a7*τ^7
        """
        # Default to zero
        alpha_0 = alpha_0 if alpha_0 is not None else 0.0
        alpha_f = alpha_f if alpha_f is not None else 0.0
        jerk_0 = jerk_0 if jerk_0 is not None else 0.0
        jerk_f = jerk_f if jerk_f is not None else 0.0
        
        # Scale derivatives
        v0 = omega_0 * T
        vf = omega_f * T
        a0 = alpha_0 * T * T
        af = alpha_f * T * T
        j0 = jerk_0 * T * T * T
        jf = jerk_f * T * T * T
        
        # First 4 coefficients from initial conditions
        c0 = theta_0
        c1 = v0
        c2 = a0 / 2.0
        c3 = j0 / 6.0
        
        # Remaining from final conditions - solve 4x4 system
        A = np.array([
            [1, 1, 1, 1],           # θ(1)
            [4, 5, 6, 7],           # θ'(1)
            [12, 20, 30, 42],       # θ''(1)
            [24, 60, 120, 210]      # θ'''(1)
        ], dtype=np.float64)
        
        b = np.array([
            theta_f - c0 - c1 - c2 - c3,
            vf - c1 - 2*c2 - 3*c3,
            af - 2*c2 - 6*c3,
            jf - 6*c3
        ])
        
        x = np.linalg.solve(A, b)
        
        return np.array([c0, c1, c2, c3, x[0], x[1], x[2], x[3]])
    
    def _eval_polynomial(
        self,
        coeffs: NDArray[np.float64],
        tau: float,
        total_angle: float
    ) -> float:
        """Evaluate polynomial at normalized time tau."""
        result = 0.0
        for i, c in enumerate(coeffs):
            result += c * (tau ** i)
        return result
    
    def _eval_polynomial_deriv(
        self,
        coeffs: NDArray[np.float64],
        tau: float,
        total_angle: float,
        T: float,
        order: int = 1
    ) -> float:
        """Evaluate polynomial derivative at normalized time tau."""
        if order == 1:
            # First derivative
            result = 0.0
            for i in range(1, len(coeffs)):
                result += i * coeffs[i] * (tau ** (i-1))
            return result / T  # Scale back to real time
        elif order == 2:
            # Second derivative
            result = 0.0
            for i in range(2, len(coeffs)):
                result += i * (i-1) * coeffs[i] * (tau ** (i-2))
            return result / (T * T)
        else:
            raise ValueError(f"Order {order} not supported")
    
    def _angle_axis_to_quaternion(
        self,
        angle: float,
        axis: NDArray[np.float64],
        q0: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Convert rotation by angle about axis to quaternion, starting from q0.
        
        q_result = q_rotation * q0
        """
        # Rotation quaternion from angle-axis
        half_angle = angle / 2.0
        q_rot = np.array([
            axis[0] * np.sin(half_angle),
            axis[1] * np.sin(half_angle),
            axis[2] * np.sin(half_angle),
            np.cos(half_angle)
        ])
        
        # Apply rotation: q_result = q_rot * q0
        q_result = self._quaternion_multiply(q_rot, q0)
        
        # Normalize
        q_result = q_result / np.linalg.norm(q_result)
        
        return q_result
    
    def _compute_inverse_dynamics(
        self,
        states: NDArray[np.float64],
        omega_profile: NDArray[np.float64],
        alpha_profile_mag: NDArray[np.float64],
        eigenaxis: NDArray[np.float64],
        J_inertia: NDArray[np.float64],
        u_max: NDArray[np.float64],
        dt: float,
        n_rw: int
    ) -> NDArray[np.float64]:
        """Compute feedforward torques using inverse dynamics."""
        N = len(states)
        n_controls = len(u_max)
        controls = np.zeros((N - 1, n_controls))
        
        J = np.array(J_inertia)
        
        for k in range(N - 1):
            omega_k = omega_profile[k]
            alpha_k = alpha_profile_mag[k] * eigenaxis  # Angular acceleration vector
            
            # Gyroscopic term
            h_rw = states[k, 7:7+n_rw] if n_rw > 0 else np.zeros(3)
            H_total = J @ omega_k
            if n_rw > 0 and len(h_rw) == 3:
                H_total = H_total + h_rw
            
            gyro_term = np.cross(omega_k, H_total)
            
            # Required torque
            tau_required = J @ alpha_k + gyro_term
            
            # Simple allocation to RWs
            if n_rw >= 3:
                controls[k, :3] = np.zeros(3)
                controls[k, 3:3+min(n_rw, 3)] = np.clip(
                    tau_required[:min(n_rw, 3)],
                    -u_max[3:3+min(n_rw, 3)],
                    u_max[3:3+min(n_rw, 3)]
                )
        
        return controls
    
    def _compute_eigenaxis(
        self,
        q0: NDArray[np.float64],
        q_goal: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], float]:
        """Compute eigenaxis and angle for rotation from q0 to q_goal."""
        q0_inv = np.array([-q0[0], -q0[1], -q0[2], q0[3]])
        q_rel = self._quaternion_multiply(q_goal, q0_inv)
        
        if q_rel[3] < 0:
            q_rel = -q_rel
        
        sin_half_angle = np.linalg.norm(q_rel[:3])
        cos_half_angle = q_rel[3]
        
        if sin_half_angle < 1e-10:
            eigenaxis = np.array([0.0, 0.0, 1.0])
            angle = 0.0
        else:
            eigenaxis = q_rel[:3] / sin_half_angle
            angle = 2.0 * np.arctan2(sin_half_angle, cos_half_angle)
        
        return eigenaxis, angle
    
    @staticmethod
    def _quaternion_multiply(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> NDArray[np.float64]:
        """Multiply two quaternions q1 * q2."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])
    
    @staticmethod
    def _quaternion_angle(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> float:
        """Compute angle between two quaternions in radians."""
        dot = np.abs(np.dot(q1, q2))
        dot = np.clip(dot, -1.0, 1.0)
        return 2.0 * np.arccos(dot)
