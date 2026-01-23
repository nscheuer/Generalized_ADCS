"""
Eigenaxis + Trapezoidal Velocity Profile Trajectory Planner.

This is the industry-standard baseline method for spacecraft attitude maneuvers.
It computes the shortest rotation path (eigenaxis) and plans a trapezoidal
angular velocity profile to smoothly accelerate, coast, and decelerate.

This method is:
- Fast and deterministic
- Flight-proven on many missions
- Simple to implement and verify
- Does NOT optimize for control effort or handle complex constraints

References:
    Wie, B. "Space Vehicle Dynamics and Control", Chapter 7
    Wertz, J.R. "Spacecraft Attitude Determination and Control", Chapter 18
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple
from dataclasses import dataclass

from .base_planner import BasePlanner, PlannerResult, PlannerConfig, DynamicsModel


@dataclass
class TrapezoidalConfig(PlannerConfig):
    """Configuration specific to eigenaxis trapezoidal planner."""
    
    # Angular velocity limits (rad/s)
    omega_max: float = 0.05          # Maximum angular velocity magnitude
    
    # Angular acceleration limits (rad/s^2)
    alpha_max: float = 0.01          # Maximum angular acceleration
    
    # Minimum coast time (seconds) - ensures smooth profile
    min_coast_fraction: float = 0.0  # Fraction of maneuver in coast phase
    
    # Time margin factor
    time_margin: float = 1.1         # Add 10% margin to computed maneuver time
    
    # Use bang-coast-bang (True) or smooth trapezoid (False)
    use_bang_bang: bool = False


class EigenaxisTrapezoidalPlanner(BasePlanner):
    """
    Eigenaxis rotation with trapezoidal velocity profile.
    
    This planner:
    1. Computes the eigenaxis (axis of rotation) from initial to goal quaternion
    2. Computes the rotation angle about this axis
    3. Plans a trapezoidal angular velocity profile
    4. Integrates quaternion kinematics to generate the trajectory
    5. Computes feedforward torques using inverse dynamics
    
    The trapezoidal profile has three phases:
    - Acceleration: omega increases linearly from 0 to omega_max
    - Coast: omega stays constant at omega_max  
    - Deceleration: omega decreases linearly from omega_max to 0
    """
    
    def __init__(self, config: Optional[TrapezoidalConfig] = None):
        """Initialize planner with configuration."""
        if config is None:
            config = TrapezoidalConfig()
        super().__init__(config)
        self.config: TrapezoidalConfig = config
        self._name = "Eigenaxis+Trapezoidal"
    
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
        Generate eigenaxis trajectory with trapezoidal velocity profile.
        
        Args:
            x0: Initial state [omega (3), quaternion (4), rw_momentum (n_rw)]
            x_goal: Goal state (same format)
            J_inertia: Spacecraft inertia matrix (3x3)
            u_max: Maximum control for each actuator
            B_field: Magnetic field (not used by this planner)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            PlannerResult with trajectory and metrics
        """
        import time
        start_time = time.perf_counter()
        
        # Extract initial and goal quaternions
        q0 = x0[3:7].copy()
        q_goal = x_goal[3:7].copy()
        omega0 = x0[:3].copy()
        
        # Normalize quaternions
        q0 = q0 / np.linalg.norm(q0)
        q_goal = q_goal / np.linalg.norm(q_goal)
        
        # Determine number of RWs from state dimension
        n_rw = len(x0) - 7
        has_rw = n_rw > 0
        
        # Compute eigenaxis and angle
        eigenaxis, angle = self._compute_eigenaxis(q0, q_goal)
        
        # Compute trapezoidal profile parameters
        t_accel, t_coast, t_decel, omega_peak = self._compute_trapezoid_params(
            angle, self.config.omega_max, self.config.alpha_max
        )
        
        total_time = (t_accel + t_coast + t_decel) * self.config.time_margin
        
        # Override with config horizon if specified and longer
        if self.config.horizon > total_time:
            total_time = self.config.horizon
        
        # Generate time array
        N = int(np.ceil(total_time / self.config.dt)) + 1
        times = np.linspace(0, total_time, N)
        dt = times[1] - times[0] if N > 1 else self.config.dt
        
        # Generate angular velocity profile
        omega_profile = self._generate_omega_profile(
            times, eigenaxis, t_accel, t_coast, t_decel, omega_peak
        )
        
        # Add initial angular velocity (if nonzero, blend it)
        # For simplicity, we start from the planned profile
        # In practice, you might want to handle nonzero initial omega
        
        # Integrate quaternion kinematics
        states = np.zeros((N, len(x0)))
        states[0] = x0.copy()
        states[0, 3:7] = q0  # Use normalized q0
        
        for k in range(N - 1):
            omega_k = omega_profile[k]
            q_k = states[k, 3:7]
            
            # RK4 integration of quaternion kinematics
            q_next = self._integrate_quaternion_rk4(q_k, omega_k, dt)
            
            # Update state
            states[k+1, :3] = omega_profile[k+1] if k+1 < len(omega_profile) else omega_profile[-1]
            states[k+1, 3:7] = q_next
            
            # RW momentum stays constant (no momentum management in this planner)
            if has_rw:
                states[k+1, 7:] = states[k, 7:]
        
        # Compute feedforward controls using inverse dynamics
        controls = self._compute_inverse_dynamics(
            states, omega_profile, J_inertia, u_max, dt, n_rw
        )
        
        solve_time = time.perf_counter() - start_time
        
        # Compute final errors
        q_final = states[-1, 3:7]
        omega_final = states[-1, :3]
        angle_error = self._quaternion_angle(q_final, q_goal)
        omega_error = np.linalg.norm(omega_final - x_goal[:3])
        
        return PlannerResult(
            times=times,
            states=states,
            controls=controls,
            solve_time=solve_time,
            converged=True,  # Kinematic planner always "converges"
            iterations=1,    # Single pass
            final_cost=angle_error + omega_error,  # Proxy for cost
            max_constraint_violation=0.0,  # Doesn't check constraints
            solver_info={
                "eigenaxis": eigenaxis.tolist(),
                "rotation_angle_deg": np.degrees(angle),
                "omega_peak": omega_peak,
                "t_accel": t_accel,
                "t_coast": t_coast,
                "t_decel": t_decel,
                "final_angle_error_deg": np.degrees(angle_error),
                "final_omega_error": omega_error,
            }
        )
    
    def _compute_eigenaxis(
        self,
        q0: NDArray[np.float64],
        q_goal: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], float]:
        """
        Compute eigenaxis and angle for rotation from q0 to q_goal.
        
        The eigenaxis is the axis about which the shortest rotation occurs.
        
        Uses scalar-first convention: q = [qw, qx, qy, qz]
        
        Args:
            q0: Initial quaternion [qw, qx, qy, qz] (scalar-first)
            q_goal: Goal quaternion [qw, qx, qy, qz] (scalar-first)
            
        Returns:
            (eigenaxis, angle) where eigenaxis is unit vector and angle in radians
        """
        # Compute relative quaternion: q_rel = q_goal * q0^(-1)
        # q0^(-1) = conjugate for unit quaternion = [qw, -qx, -qy, -qz]
        q0_inv = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
        
        # Quaternion multiplication: q_rel = q_goal * q0_inv
        q_rel = self._quaternion_multiply(q_goal, q0_inv)
        
        # Ensure shortest path (q and -q represent same rotation)
        if q_rel[0] < 0:  # scalar part is at index 0
            q_rel = -q_rel
        
        # Extract axis and angle
        # q = [cos(θ/2), sin(θ/2)*axis] for scalar-first
        sin_half_angle = np.linalg.norm(q_rel[1:4])  # vector part
        cos_half_angle = q_rel[0]  # scalar part
        
        # Handle small angles
        if sin_half_angle < 1e-10:
            # Near-identity rotation
            eigenaxis = np.array([0.0, 0.0, 1.0])  # Arbitrary axis
            angle = 0.0
        else:
            eigenaxis = q_rel[1:4] / sin_half_angle  # vector part is [1:4]
            angle = 2.0 * np.arctan2(sin_half_angle, cos_half_angle)
        
        return eigenaxis, angle
    
    def _compute_trapezoid_params(
        self,
        angle: float,
        omega_max: float,
        alpha_max: float
    ) -> Tuple[float, float, float, float]:
        """
        Compute trapezoidal velocity profile parameters.
        
        For rotation angle θ with max velocity ω_max and max acceleration α_max:
        - If we can reach ω_max: triangular or trapezoidal profile
        - If angle is too small: triangular profile (no coast phase)
        
        Args:
            angle: Total rotation angle in radians
            omega_max: Maximum angular velocity (rad/s)
            alpha_max: Maximum angular acceleration (rad/s^2)
            
        Returns:
            (t_accel, t_coast, t_decel, omega_peak)
        """
        # Time to accelerate to omega_max
        t_accel_full = omega_max / alpha_max
        
        # Angle covered during full acceleration + deceleration (no coast)
        angle_accel_decel = omega_max * t_accel_full  # = omega_max^2 / alpha_max
        
        if angle <= angle_accel_decel:
            # Triangular profile - can't reach omega_max
            # angle = 0.5 * omega_peak * t_accel + 0.5 * omega_peak * t_decel
            # angle = omega_peak * t_accel (symmetric)
            # omega_peak = alpha_max * t_accel
            # angle = alpha_max * t_accel^2
            t_accel = np.sqrt(angle / alpha_max)
            t_coast = 0.0
            t_decel = t_accel
            omega_peak = alpha_max * t_accel
        else:
            # Trapezoidal profile
            t_accel = t_accel_full
            t_decel = t_accel_full
            omega_peak = omega_max
            
            # Angle during coast phase
            angle_coast = angle - angle_accel_decel
            t_coast = angle_coast / omega_max
        
        return t_accel, t_coast, t_decel, omega_peak
    
    def _generate_omega_profile(
        self,
        times: NDArray[np.float64],
        eigenaxis: NDArray[np.float64],
        t_accel: float,
        t_coast: float,
        t_decel: float,
        omega_peak: float
    ) -> NDArray[np.float64]:
        """
        Generate angular velocity profile along eigenaxis.
        
        Args:
            times: Time array
            eigenaxis: Unit vector for rotation axis
            t_accel: Acceleration phase duration
            t_coast: Coast phase duration
            t_decel: Deceleration phase duration
            omega_peak: Peak angular velocity magnitude
            
        Returns:
            (N, 3) angular velocity array
        """
        N = len(times)
        omega_profile = np.zeros((N, 3))
        
        t_end_accel = t_accel
        t_end_coast = t_accel + t_coast
        t_end_decel = t_accel + t_coast + t_decel
        
        alpha = omega_peak / t_accel if t_accel > 0 else 0.0
        
        for i, t in enumerate(times):
            if t < 0:
                omega_mag = 0.0
            elif t < t_end_accel:
                # Acceleration phase
                omega_mag = alpha * t
            elif t < t_end_coast:
                # Coast phase
                omega_mag = omega_peak
            elif t < t_end_decel:
                # Deceleration phase
                t_in_decel = t - t_end_coast
                omega_mag = omega_peak - alpha * t_in_decel
            else:
                # After maneuver complete
                omega_mag = 0.0
            
            omega_mag = max(0.0, omega_mag)  # Ensure non-negative
            omega_profile[i] = omega_mag * eigenaxis
        
        return omega_profile
    
    def _integrate_quaternion_rk4(
        self,
        q: NDArray[np.float64],
        omega: NDArray[np.float64],
        dt: float
    ) -> NDArray[np.float64]:
        """
        Integrate quaternion kinematics using RK4.
        
        Uses scalar-first convention: q = [w, x, y, z]
        q_dot = 0.5 * q ⊗ [0, omega]
        """
        def q_dot(q_val, w):
            # For scalar-first: q = [qw, qx, qy, qz]
            qw, qx, qy, qz = q_val
            wx, wy, wz = w
            return 0.5 * np.array([
                -qx*wx - qy*wy - qz*wz,  # w_dot
                qw*wx + qy*wz - qz*wy,   # x_dot  
                qw*wy + qz*wx - qx*wz,   # y_dot
                qw*wz + qx*wy - qy*wx    # z_dot
            ])
        
        k1 = q_dot(q, omega)
        k2 = q_dot(q + 0.5*dt*k1, omega)
        k3 = q_dot(q + 0.5*dt*k2, omega)
        k4 = q_dot(q + dt*k3, omega)
        
        q_next = q + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Normalize
        q_next = q_next / np.linalg.norm(q_next)
        
        return q_next
    
    def _compute_inverse_dynamics(
        self,
        states: NDArray[np.float64],
        omega_profile: NDArray[np.float64],
        J_inertia: NDArray[np.float64],
        u_max: NDArray[np.float64],
        dt: float,
        n_rw: int
    ) -> NDArray[np.float64]:
        """
        Compute feedforward torques using inverse dynamics.
        
        tau = J * omega_dot + omega x (J * omega + h_rw)
        
        For simplicity, we ignore RW momentum in cross product and
        compute the required total torque, then allocate to actuators.
        """
        N = len(states)
        n_controls = len(u_max)
        controls = np.zeros((N - 1, n_controls))
        
        J = np.array(J_inertia)
        
        for k in range(N - 1):
            omega_k = omega_profile[k]
            omega_kp1 = omega_profile[k + 1] if k + 1 < len(omega_profile) else omega_profile[-1]
            
            # Angular acceleration
            omega_dot = (omega_kp1 - omega_k) / dt
            
            # Gyroscopic term
            h_rw = states[k, 7:7+n_rw] if n_rw > 0 else np.zeros(3)
            H_total = J @ omega_k
            if n_rw > 0 and len(h_rw) == 3:
                H_total = H_total + h_rw
            
            gyro_term = np.cross(omega_k, H_total)
            
            # Required torque
            tau_required = J @ omega_dot + gyro_term
            
            # Simple allocation: put all torque on RWs if available
            if n_rw >= 3:
                # Assume 3 orthogonal RWs, allocate directly
                controls[k, :3] = np.zeros(3)  # No MTQ torque
                controls[k, 3:3+min(n_rw, 3)] = np.clip(
                    tau_required[:min(n_rw, 3)],
                    -u_max[3:3+min(n_rw, 3)],
                    u_max[3:3+min(n_rw, 3)]
                )
            else:
                # MTQ only - can't do proper allocation without B field
                # Just set to zero (this planner doesn't handle MTQ well)
                controls[k, :] = 0.0
        
        return controls
    
    @staticmethod
    def _quaternion_multiply(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> NDArray[np.float64]:
        """Multiply two quaternions q1 * q2. Uses scalar-first: [w, x, y, z]."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
            w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
            w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
            w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
        ])
    
    @staticmethod
    def _quaternion_angle(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> float:
        """Compute angle between two quaternions in radians."""
        dot = np.abs(np.dot(q1, q2))
        dot = np.clip(dot, -1.0, 1.0)
        return 2.0 * np.arccos(dot)
