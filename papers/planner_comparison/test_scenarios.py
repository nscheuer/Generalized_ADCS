"""
Test Scenarios for Trajectory Planner Comparison.

This module defines standardized test scenarios for comparing planner performance
across different maneuver types and conditions.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class TestScenario:
    """
    Container for a trajectory planning test scenario.
    
    Each scenario defines:
    - Initial and goal states
    - Spacecraft parameters (inertia, actuators)
    - Environmental conditions
    - Success criteria
    """
    
    name: str
    description: str
    
    # Initial state [omega (3), quaternion (4), rw_momentum (n_rw)]
    x0: NDArray[np.float64]
    
    # Goal state
    x_goal: NDArray[np.float64]
    
    # Spacecraft parameters
    J_inertia: NDArray[np.float64]  # 3x3 inertia matrix
    u_max: NDArray[np.float64]       # Control limits per actuator
    
    # Time parameters
    horizon: float = 60.0            # Planning horizon (seconds)
    dt: float = 1.0                  # Time step (seconds)
    
    # Environmental conditions (optional)
    B_field: Optional[NDArray[np.float64]] = None  # Magnetic field (N, 3) or (3,)
    
    # Success criteria
    angle_tolerance_deg: float = 1.0       # Final angle error threshold
    omega_tolerance: float = 0.001         # Final omega error threshold (rad/s)
    max_solve_time: float = 10.0           # Maximum acceptable solve time (seconds)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    
    @property
    def rotation_angle_deg(self) -> float:
        """Compute rotation angle between initial and goal attitudes."""
        q0 = self.x0[3:7]
        q_goal = self.x_goal[3:7]
        
        # Quaternion error
        q0_inv = np.array([-q0[0], -q0[1], -q0[2], q0[3]])
        
        # q_rel = q_goal * q0_inv
        w1 = q_goal[3]
        x1, y1, z1 = q_goal[:3]
        w2 = q0_inv[3]
        x2, y2, z2 = q0_inv[:3]
        
        q_rel = np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])
        
        if q_rel[3] < 0:
            q_rel = -q_rel
        
        angle = 2.0 * np.arctan2(np.linalg.norm(q_rel[:3]), q_rel[3])
        return np.degrees(angle)
    
    @property
    def n_rw(self) -> int:
        """Number of reaction wheels (inferred from state dimension)."""
        return len(self.x0) - 7
    
    @property
    def n_controls(self) -> int:
        """Number of control inputs."""
        return len(self.u_max)


def normalize_quaternion(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize quaternion to unit magnitude."""
    return q / np.linalg.norm(q)


def angle_axis_to_quaternion(angle_deg: float, axis: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert angle-axis to quaternion.
    
    Uses scalar-first convention [qw, qx, qy, qz] to match ADCS codebase.
    
    Args:
        angle_deg: Rotation angle in degrees
        axis: Rotation axis (will be normalized)
        
    Returns:
        Quaternion [qw, qx, qy, qz] (scalar-first convention)
    """
    axis = np.array(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    
    angle_rad = np.radians(angle_deg)
    half_angle = angle_rad / 2.0
    
    # Scalar-first: [w, x, y, z]
    return np.array([
        np.cos(half_angle),
        axis[0] * np.sin(half_angle),
        axis[1] * np.sin(half_angle),
        axis[2] * np.sin(half_angle),
    ])


class ScenarioLibrary:
    """
    Library of standard test scenarios for planner comparison.
    """
    
    @staticmethod
    def get_default_inertia() -> NDArray[np.float64]:
        """Standard 4kg CubeSat-like inertia matrix."""
        return np.diag([0.04, 0.04, 0.02])  # 3U CubeSat approximation
    
    @staticmethod
    def get_default_u_max(n_mtq: int = 3, n_rw: int = 3) -> NDArray[np.float64]:
        """
        Default control limits.
        
        MTQ: 0.2 Am² (magnetic dipole)
        RW: 0.004 Nm (reaction wheel torque)
        """
        mtq_max = 0.2 * np.ones(n_mtq)
        rw_max = 0.004 * np.ones(n_rw)
        return np.concatenate([mtq_max, rw_max])
    
    @staticmethod
    def create_rest_to_rest(
        angle_deg: float,
        axis: NDArray[np.float64] = None,
        horizon: float = 60.0,
        name: str = None
    ) -> TestScenario:
        """
        Create a rest-to-rest maneuver scenario.
        
        Args:
            angle_deg: Rotation angle in degrees
            axis: Rotation axis (default: z-axis)
            horizon: Planning horizon in seconds
            name: Scenario name (auto-generated if None)
            
        Returns:
            TestScenario for rest-to-rest maneuver
        """
        if axis is None:
            axis = np.array([0, 0, 1])
        axis = np.array(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        
        # Initial state: identity quaternion, zero angular velocity, zero RW momentum
        # Using scalar-first convention: [w, x, y, z] = [1, 0, 0, 0]
        q0 = np.array([1, 0, 0, 0], dtype=np.float64)
        omega0 = np.zeros(3)
        h_rw0 = np.zeros(3)  # 3 reaction wheels
        x0 = np.concatenate([omega0, q0, h_rw0])
        
        # Goal state: rotated quaternion, zero angular velocity
        q_goal = angle_axis_to_quaternion(angle_deg, axis)
        omega_goal = np.zeros(3)
        h_rw_goal = np.zeros(3)
        x_goal = np.concatenate([omega_goal, q_goal, h_rw_goal])
        
        if name is None:
            name = f"RestToRest_{int(angle_deg)}deg"
        
        return TestScenario(
            name=name,
            description=f"Rest-to-rest {angle_deg}° rotation about {axis}",
            x0=x0,
            x_goal=x_goal,
            J_inertia=ScenarioLibrary.get_default_inertia(),
            u_max=ScenarioLibrary.get_default_u_max(),
            horizon=horizon,
            tags=["rest-to-rest", f"{int(angle_deg)}deg"],
        )
    
    @staticmethod
    def create_with_initial_rate(
        angle_deg: float,
        omega0_mag: float = 0.01,
        axis: NDArray[np.float64] = None,
        horizon: float = 60.0
    ) -> TestScenario:
        """
        Create scenario with non-zero initial angular velocity.
        
        Args:
            angle_deg: Target rotation angle
            omega0_mag: Initial angular velocity magnitude (rad/s)
            axis: Rotation axis (default: z-axis)
            horizon: Planning horizon
            
        Returns:
            TestScenario with initial angular velocity
        """
        if axis is None:
            axis = np.array([0, 0, 1])
        axis = np.array(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        
        # Initial state with angular velocity
        # Using scalar-first convention: [w, x, y, z] = [1, 0, 0, 0]
        q0 = np.array([1, 0, 0, 0], dtype=np.float64)
        omega0 = omega0_mag * np.array([1, 0.5, 0.2])  # Non-aligned with axis
        omega0 = omega0 / np.linalg.norm(omega0) * omega0_mag
        h_rw0 = np.zeros(3)
        x0 = np.concatenate([omega0, q0, h_rw0])
        
        # Goal: rotated and at rest
        q_goal = angle_axis_to_quaternion(angle_deg, axis)
        omega_goal = np.zeros(3)
        h_rw_goal = np.zeros(3)
        x_goal = np.concatenate([omega_goal, q_goal, h_rw_goal])
        
        return TestScenario(
            name=f"WithRate_{int(angle_deg)}deg_w{omega0_mag:.3f}",
            description=f"{angle_deg}° rotation with initial rate {omega0_mag} rad/s",
            x0=x0,
            x_goal=x_goal,
            J_inertia=ScenarioLibrary.get_default_inertia(),
            u_max=ScenarioLibrary.get_default_u_max(),
            horizon=horizon,
            tags=["with-rate", f"{int(angle_deg)}deg"],
        )
    
    @staticmethod
    def create_large_angle(
        angle_deg: float = 180.0,
        horizon: float = 120.0
    ) -> TestScenario:
        """
        Create a large angle (flip) maneuver scenario.
        
        Args:
            angle_deg: Rotation angle (default 180°)
            horizon: Planning horizon (longer for large maneuvers)
            
        Returns:
            TestScenario for large angle maneuver
        """
        # Choose a challenging axis (not aligned with principal axes)
        axis = np.array([1, 1, 1], dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        
        return ScenarioLibrary.create_rest_to_rest(
            angle_deg=angle_deg,
            axis=axis,
            horizon=horizon,
            name=f"LargeAngle_{int(angle_deg)}deg"
        )
    
    @staticmethod
    def create_small_angle(
        angle_deg: float = 5.0,
        horizon: float = 30.0
    ) -> TestScenario:
        """Create small angle maneuver (fine pointing adjustment)."""
        axis = np.array([0, 1, 0])
        scenario = ScenarioLibrary.create_rest_to_rest(
            angle_deg=angle_deg,
            axis=axis,
            horizon=horizon,
            name=f"SmallAngle_{int(angle_deg)}deg"
        )
        scenario.angle_tolerance_deg = 0.1  # Tighter tolerance for small angles
        return scenario
    
    @staticmethod
    def create_asymmetric_inertia(
        angle_deg: float = 45.0,
        horizon: float = 60.0
    ) -> TestScenario:
        """Create scenario with asymmetric (non-diagonal) inertia."""
        # Asymmetric inertia (off-diagonal terms)
        J = np.array([
            [0.05, 0.01, 0.005],
            [0.01, 0.04, 0.008],
            [0.005, 0.008, 0.02]
        ])
        
        scenario = ScenarioLibrary.create_rest_to_rest(
            angle_deg=angle_deg,
            axis=np.array([1, 0, 0]),
            horizon=horizon,
            name=f"AsymmetricInertia_{int(angle_deg)}deg"
        )
        scenario.J_inertia = J
        scenario.tags.append("asymmetric-inertia")
        return scenario
    
    @staticmethod
    def create_constrained_actuator(
        angle_deg: float = 45.0,
        u_scale: float = 0.5,
        horizon: float = 90.0
    ) -> TestScenario:
        """
        Create scenario with reduced actuator authority.
        
        Args:
            angle_deg: Rotation angle
            u_scale: Fraction of normal control limits (e.g., 0.5 = 50%)
            horizon: Planning horizon (longer due to reduced authority)
        """
        scenario = ScenarioLibrary.create_rest_to_rest(
            angle_deg=angle_deg,
            axis=np.array([0, 0, 1]),
            horizon=horizon,
            name=f"ConstrainedActuator_{int(angle_deg)}deg_{int(u_scale*100)}pct"
        )
        scenario.u_max = scenario.u_max * u_scale
        scenario.tags.append("constrained-actuator")
        return scenario
    
    @staticmethod
    def create_tracking_maneuver(
        rate_deg_s: float = 0.5,
        duration: float = 60.0
    ) -> TestScenario:
        """
        Create a tracking maneuver (constant angular velocity).
        
        Args:
            rate_deg_s: Target angular velocity in deg/s
            duration: Tracking duration
            
        Returns:
            Scenario where goal is to maintain angular velocity
        """
        rate_rad_s = np.radians(rate_deg_s)
        axis = np.array([0, 0, 1])
        
        # Initial: at rest
        # Using scalar-first convention: [w, x, y, z] = [1, 0, 0, 0]
        q0 = np.array([1, 0, 0, 0], dtype=np.float64)
        omega0 = np.zeros(3)
        h_rw0 = np.zeros(3)
        x0 = np.concatenate([omega0, q0, h_rw0])
        
        # Goal: spinning at constant rate
        # Final quaternion after spinning for 'duration' seconds
        final_angle = rate_rad_s * duration
        q_goal = angle_axis_to_quaternion(np.degrees(final_angle), axis)
        omega_goal = rate_rad_s * axis
        h_rw_goal = np.zeros(3)
        x_goal = np.concatenate([omega_goal, q_goal, h_rw_goal])
        
        return TestScenario(
            name=f"Tracking_{rate_deg_s}degps",
            description=f"Track {rate_deg_s}°/s rotation rate",
            x0=x0,
            x_goal=x_goal,
            J_inertia=ScenarioLibrary.get_default_inertia(),
            u_max=ScenarioLibrary.get_default_u_max(),
            horizon=duration,
            omega_tolerance=np.radians(0.1),  # 0.1 deg/s tolerance
            tags=["tracking", "constant-rate"],
        )
    
    @staticmethod
    def get_standard_scenarios() -> List[TestScenario]:
        """
        Get the standard set of scenarios for comprehensive testing.
        
        Returns:
            List of TestScenario objects covering various conditions
        """
        scenarios = [
            # Basic rest-to-rest at different angles
            ScenarioLibrary.create_rest_to_rest(10.0, horizon=30.0),
            ScenarioLibrary.create_rest_to_rest(30.0, horizon=45.0),
            ScenarioLibrary.create_rest_to_rest(45.0, horizon=60.0),
            ScenarioLibrary.create_rest_to_rest(90.0, horizon=90.0),
            
            # Large angle maneuver
            ScenarioLibrary.create_large_angle(180.0, horizon=150.0),
            
            # Small angle (fine pointing)
            ScenarioLibrary.create_small_angle(5.0, horizon=20.0),
            
            # With initial angular velocity
            ScenarioLibrary.create_with_initial_rate(45.0, omega0_mag=0.02, horizon=60.0),
            
            # Constrained actuator
            ScenarioLibrary.create_constrained_actuator(45.0, u_scale=0.5, horizon=90.0),
        ]
        
        return scenarios
    
    @staticmethod
    def get_quick_scenarios() -> List[TestScenario]:
        """Get a minimal set of scenarios for quick testing."""
        return [
            ScenarioLibrary.create_rest_to_rest(30.0, horizon=45.0),
            ScenarioLibrary.create_rest_to_rest(90.0, horizon=90.0),
        ]
    
    @staticmethod
    def get_stress_test_scenarios() -> List[TestScenario]:
        """Get challenging scenarios for stress testing."""
        return [
            ScenarioLibrary.create_large_angle(180.0, horizon=180.0),
            ScenarioLibrary.create_with_initial_rate(90.0, omega0_mag=0.05, horizon=90.0),
            ScenarioLibrary.create_constrained_actuator(90.0, u_scale=0.3, horizon=150.0),
            ScenarioLibrary.create_asymmetric_inertia(60.0, horizon=75.0),
        ]
    
    @staticmethod
    def create_mtq_only(
        angle_deg: float = 30.0,
        horizon: float = 300.0,  # MTQ-only needs longer horizon
        B_field: Optional[NDArray[np.float64]] = None
    ) -> TestScenario:
        """
        Create underactuated scenario with magnetorquers only (no reaction wheels).
        
        MTQ-only is challenging because:
        1. Can only produce torque perpendicular to B-field
        2. Controllability varies with orbit position
        3. Much slower than RW-based control
        
        Args:
            angle_deg: Target rotation angle
            horizon: Planning horizon (should be ~1/4 to 1/2 orbit for MTQ)
            B_field: Magnetic field trajectory (N,3) or constant (3,)
        """
        axis = np.array([0, 0, 1])
        
        # No RW momentum in state (MTQ only has 7-element state)
        q0 = np.array([1, 0, 0, 0], dtype=np.float64)
        omega0 = np.zeros(3)
        x0 = np.concatenate([omega0, q0])  # No h_rw
        
        q_goal = angle_axis_to_quaternion(angle_deg, axis)
        omega_goal = np.zeros(3)
        x_goal = np.concatenate([omega_goal, q_goal])  # No h_rw
        
        # Default B-field if not provided (typical LEO field ~30 μT)
        if B_field is None:
            B_field = np.array([2e-5, 1e-5, 3e-5])  # Body frame
        
        # MTQ-only actuator limits (3 MTQs, typical 0.2 Am²)
        u_max = np.array([0.2, 0.2, 0.2])
        
        return TestScenario(
            name=f"MTQ_Only_{int(angle_deg)}deg",
            description=f"Underactuated MTQ-only {angle_deg}° maneuver",
            x0=x0,
            x_goal=x_goal,
            J_inertia=ScenarioLibrary.get_default_inertia(),
            u_max=u_max,
            B_field=B_field,
            horizon=horizon,
            dt=1.0,  # Coarser timestep OK for slow MTQ maneuvers
            angle_tolerance_deg=1.0,  # Looser tolerance for underactuated
            tags=["underactuated", "mtq-only"],
        )
    
    @staticmethod
    def create_desaturation(
        h_rw_initial: NDArray[np.float64] = None,
        angle_deg: float = 30.0,
        horizon: float = 180.0
    ) -> TestScenario:
        """
        Create scenario requiring RW desaturation (momentum dumping).
        
        The satellite starts with reaction wheels near saturation and must
        simultaneously point to a new attitude while dumping momentum using MTQs.
        
        Args:
            h_rw_initial: Initial RW momentum (defaults to 80% of max)
            angle_deg: Target pointing angle change
            horizon: Planning horizon (needs time for momentum dumping)
        """
        axis = np.array([0, 0, 1])
        
        # Initial state with saturated RWs
        q0 = np.array([1, 0, 0, 0], dtype=np.float64)
        omega0 = np.zeros(3)
        
        # Default to 80% saturated RWs
        h_max = 0.05  # Typical small sat RW max momentum (Nms)
        if h_rw_initial is None:
            h_rw_initial = 0.8 * h_max * np.array([1.0, -0.5, 0.3])
        
        x0 = np.concatenate([omega0, q0, h_rw_initial])
        
        # Goal: new attitude AND zero RW momentum (desaturated)
        q_goal = angle_axis_to_quaternion(angle_deg, axis)
        omega_goal = np.zeros(3)
        h_rw_goal = np.zeros(3)  # Target: fully desaturated
        x_goal = np.concatenate([omega_goal, q_goal, h_rw_goal])
        
        # Hybrid actuator set: MTQ for momentum dumping + RW for pointing
        # [MTQ_x, MTQ_y, MTQ_z, RW_x, RW_y, RW_z]
        u_max = np.array([0.2, 0.2, 0.2, 0.004, 0.004, 0.004])
        
        # Time-varying B-field (simulates orbit)
        N = int(horizon) + 1
        t = np.linspace(0, horizon, N)
        B_mag = 3e-5
        B_field = np.zeros((N, 3))
        for i, ti in enumerate(t):
            # Simple rotating B-field model
            phase = 2 * np.pi * ti / 5400  # ~90 min orbit
            B_field[i] = B_mag * np.array([np.cos(phase), np.sin(phase), 0.3])
        
        return TestScenario(
            name=f"Desaturation_{int(angle_deg)}deg",
            description=f"RW desaturation during {angle_deg}° slew",
            x0=x0,
            x_goal=x_goal,
            J_inertia=ScenarioLibrary.get_default_inertia(),
            u_max=u_max,
            B_field=B_field,
            horizon=horizon,
            tags=["desaturation", "momentum-management", "hybrid-actuator"],
        )
    
    @staticmethod
    def create_with_disturbance(
        angle_deg: float = 30.0,
        disturbance_torque: NDArray[np.float64] = None,
        horizon: float = 60.0
    ) -> TestScenario:
        """
        Create scenario with external disturbance torque.
        
        Tests planner's ability to handle persistent disturbances like:
        - Gravity gradient
        - Aerodynamic drag
        - Solar radiation pressure
        
        Args:
            angle_deg: Target rotation angle
            disturbance_torque: Constant disturbance in body frame (Nm)
            horizon: Planning horizon
        """
        axis = np.array([0, 0, 1])
        
        q0 = np.array([1, 0, 0, 0], dtype=np.float64)
        omega0 = np.zeros(3)
        h_rw0 = np.zeros(3)
        x0 = np.concatenate([omega0, q0, h_rw0])
        
        q_goal = angle_axis_to_quaternion(angle_deg, axis)
        omega_goal = np.zeros(3)
        h_rw_goal = np.zeros(3)
        x_goal = np.concatenate([omega_goal, q_goal, h_rw_goal])
        
        # Default disturbance: ~1e-6 Nm (typical for small LEO sat)
        if disturbance_torque is None:
            disturbance_torque = np.array([1e-6, -0.5e-6, 0.8e-6])
        
        scenario = TestScenario(
            name=f"WithDisturbance_{int(angle_deg)}deg",
            description=f"{angle_deg}° slew with external disturbance",
            x0=x0,
            x_goal=x_goal,
            J_inertia=ScenarioLibrary.get_default_inertia(),
            u_max=ScenarioLibrary.get_default_u_max(),
            horizon=horizon,
            tags=["disturbance", "external-torque"],
        )
        scenario.disturbance_torque = disturbance_torque
        return scenario
    
    @staticmethod
    def create_pointing_constraint(
        angle_deg: float = 45.0,
        keep_out_cone_deg: float = 30.0,
        keep_out_axis: NDArray[np.float64] = None,
        horizon: float = 90.0
    ) -> TestScenario:
        """
        Create scenario with pointing constraints (keep-out zones).
        
        Example: Don't point sensor at sun, or keep antenna within comm cone.
        
        Args:
            angle_deg: Target rotation angle
            keep_out_cone_deg: Half-angle of keep-out cone
            keep_out_axis: Axis to avoid (ECI frame)
            horizon: Planning horizon
        """
        axis = np.array([0, 0, 1])
        
        q0 = np.array([1, 0, 0, 0], dtype=np.float64)
        omega0 = np.zeros(3)
        h_rw0 = np.zeros(3)
        x0 = np.concatenate([omega0, q0, h_rw0])
        
        q_goal = angle_axis_to_quaternion(angle_deg, axis)
        omega_goal = np.zeros(3)
        h_rw_goal = np.zeros(3)
        x_goal = np.concatenate([omega_goal, q_goal, h_rw_goal])
        
        if keep_out_axis is None:
            keep_out_axis = np.array([1, 0, 0])  # Don't point at +X (e.g., sun)
        
        scenario = TestScenario(
            name=f"PointingConstrained_{int(angle_deg)}deg",
            description=f"{angle_deg}° slew avoiding {keep_out_cone_deg}° cone",
            x0=x0,
            x_goal=x_goal,
            J_inertia=ScenarioLibrary.get_default_inertia(),
            u_max=ScenarioLibrary.get_default_u_max(),
            horizon=horizon,
            tags=["constrained", "keep-out", "path-planning"],
        )
        scenario.keep_out_cone_deg = keep_out_cone_deg
        scenario.keep_out_axis = keep_out_axis
        return scenario
    
    @staticmethod 
    def create_reduced_attitude(
        target_vector_eci: NDArray[np.float64] = None,
        boresight_body: NDArray[np.float64] = None,
        horizon: float = 60.0
    ) -> TestScenario:
        """
        Create 2-DOF reduced attitude scenario (point boresight at target).
        
        This is the classic "point camera at Earth" problem where roll around
        boresight is unconstrained. Only 2 of 3 attitude DOF are specified.
        
        Args:
            target_vector_eci: Target direction in ECI frame
            boresight_body: Boresight direction in body frame
            horizon: Planning horizon
        """
        if target_vector_eci is None:
            target_vector_eci = np.array([0, 0, -1])  # Nadir pointing
        if boresight_body is None:
            boresight_body = np.array([0, 0, 1])  # +Z is boresight
        
        target_vector_eci = target_vector_eci / np.linalg.norm(target_vector_eci)
        boresight_body = boresight_body / np.linalg.norm(boresight_body)
        
        # Initial state: identity
        q0 = np.array([1, 0, 0, 0], dtype=np.float64)
        omega0 = np.zeros(3)
        h_rw0 = np.zeros(3)
        x0 = np.concatenate([omega0, q0, h_rw0])
        
        # For reduced attitude, goal is a 3-vector (pointing direction) not quaternion
        # The planner should understand this is 2-DOF control
        x_goal = np.concatenate([np.zeros(3), target_vector_eci, h_rw0])
        
        scenario = TestScenario(
            name="ReducedAttitude_Nadir",
            description="2-DOF boresight pointing (roll-free)",
            x0=x0,
            x_goal=x_goal,  # Note: goal[3:6] is target vector, not quaternion!
            J_inertia=ScenarioLibrary.get_default_inertia(),
            u_max=ScenarioLibrary.get_default_u_max(),
            horizon=horizon,
            tags=["reduced-attitude", "2-dof", "pointing"],
        )
        scenario.boresight_body = boresight_body
        scenario.is_reduced_attitude = True
        return scenario
    
    @staticmethod
    def get_altro_showcase_scenarios() -> List[TestScenario]:
        """
        Get scenarios that highlight ALTRO's advantages over simpler planners.
        
        These test:
        1. Underactuated control (MTQ-only)
        2. Momentum management (desaturation)
        3. Disturbance rejection
        4. Path constraints
        """
        return [
            ScenarioLibrary.create_mtq_only(30.0, horizon=300.0),
            ScenarioLibrary.create_desaturation(angle_deg=30.0, horizon=180.0),
            ScenarioLibrary.create_with_disturbance(30.0, horizon=60.0),
            ScenarioLibrary.create_pointing_constraint(45.0, keep_out_cone_deg=30.0),
            ScenarioLibrary.create_reduced_attitude(),
        ]
