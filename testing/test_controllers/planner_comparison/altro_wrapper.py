"""
ALTRO Planner Wrapper.

Wraps the existing C++ ALTRO implementation to conform to the BasePlanner interface
for comparison testing.
"""
from __future__ import annotations

import sys
import os
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from .base_planner import BasePlanner, PlannerResult, PlannerConfig


@dataclass
class ALTROConfig(PlannerConfig):
    """Configuration for ALTRO planner wrapper."""
    
    # ALTRO-specific settings
    dt_tp: float = 10.0              # Trajectory planner timestep (coarse)
    dt_tvlqr: float = 1.0            # TVLQR timestep (fine)
    
    # Convergence - Pass 1
    pass1_max_outer_iter: int = 15
    pass1_max_inner_iter: int = 80
    pass1_grad_tol: float = 1e-3
    pass1_cost_tol: float = 1e-2
    
    # Convergence - Pass 2
    pass2_max_outer_iter: int = 10
    pass2_max_inner_iter: int = 40
    pass2_grad_tol: float = 1e-4
    pass2_cost_tol: float = 1e-3
    
    # Cost weights
    angle_weight: float = 1e8
    angle_weight_terminal: float = 1e12
    ang_vel_weight: float = 1e3
    ang_vel_weight_terminal: float = 1e6
    
    # Actuator weights
    mtq_control_weight: float = 1e3
    rw_control_weight: float = 1e5
    
    # Full attitude control (3-DOF quaternion) vs 2-DOF pointing
    use_quaternion_goal: bool = True  # If True, use full 3-DOF attitude control
    
    # Quaternion to 3-vec mode for cost computation
    # 0 = MRP with qe0>0, 1 = MRP, 2 = Cayley, 3 = qev w/ qe0>0, 4 = qev
    quat_to_3vec_mode: int = 2  # Cayley is recommended
    
    # Regularization
    use_dynamics_hess: bool = False
    use_full_cost_hess: bool = False
    
    # Bdot mode: 0=off, 1=on, 2=smart
    bdot_on: int = 0


class ALTROWrapper(BasePlanner):
    """
    Wrapper for ALTRO trajectory planner.
    
    This class wraps the existing C++ ALTRO implementation to provide
    a consistent interface for comparison with other planners.
    """
    
    def __init__(self, config: Optional[ALTROConfig] = None):
        """Initialize ALTRO wrapper."""
        if config is None:
            config = ALTROConfig()
        super().__init__(config)
        self.config: ALTROConfig = config
        self._name = "ALTRO"
        
        # Will be set when solve() is called with satellite info
        self._controller = None
        self._planner_settings = None
    
    def solve(
        self,
        x0: NDArray[np.float64],
        x_goal: NDArray[np.float64],
        J_inertia: NDArray[np.float64],
        u_max: NDArray[np.float64],
        B_field: Optional[NDArray[np.float64]] = None,
        satellite: Any = None,
        orbital_state: Any = None,
        goals: Any = None,
        **kwargs
    ) -> PlannerResult:
        """
        Solve trajectory optimization using ALTRO.
        
        This method requires additional satellite infrastructure objects
        that the other planners don't need.
        
        Args:
            x0: Initial state
            x_goal: Goal state (used to construct goals if not provided)
            J_inertia: Inertia matrix
            u_max: Control limits
            B_field: Magnetic field (optional, from orbital state)
            satellite: EstimatedSatellite object (required)
            orbital_state: Orbital_State object (required)
            goals: GoalList object (optional, constructed from x_goal if not provided)
            
        Returns:
            PlannerResult with trajectory
        """
        import time
        start_time = time.perf_counter()
        
        if satellite is None:
            raise ValueError("ALTRO wrapper requires 'satellite' (EstimatedSatellite) argument")
        if orbital_state is None:
            raise ValueError("ALTRO wrapper requires 'orbital_state' (Orbital_State) argument")
        
        # Import ADCS components
        from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
        from ADCS.controller.helpers import PlannerSettings
        from ADCS.CONOPS.goals import ECI_Goal, Fixed_Attitude_Goal
        from ADCS.CONOPS.goallist import GoalList
        
        # Setup planner settings
        planner_settings = PlannerSettings(
            est_sat=satellite,
            bdot_on=self.config.bdot_on,
            dt_tp=self.config.dt_tp,
            dt_tvlqr=self.config.dt_tvlqr,
        )
        
        # Configure Pass 1
        planner_settings.pass1.convergence.max_outer_iter = self.config.pass1_max_outer_iter
        planner_settings.pass1.convergence.max_inner_iter = self.config.pass1_max_inner_iter
        planner_settings.pass1.convergence.grad_tol = self.config.pass1_grad_tol
        planner_settings.pass1.convergence.ilqr_cost_tol = self.config.pass1_cost_tol
        
        # Configure Pass 2
        planner_settings.pass2.convergence.max_outer_iter = self.config.pass2_max_outer_iter
        planner_settings.pass2.convergence.max_inner_iter = self.config.pass2_max_inner_iter
        planner_settings.pass2.convergence.grad_tol = self.config.pass2_grad_tol
        planner_settings.pass2.convergence.ilqr_cost_tol = self.config.pass2_cost_tol
        
        # Configure costs
        planner_settings.cost_main.angle = self.config.angle_weight
        planner_settings.cost_main.angle_N = self.config.angle_weight_terminal
        planner_settings.cost_main.ang_vel = self.config.ang_vel_weight
        planner_settings.cost_main.ang_vel_N = self.config.ang_vel_weight_terminal
        
        # Configure Hessians
        planner_settings.cost_main.use_full_cost_hessian = self.config.use_full_cost_hess
        planner_settings.pass1.regularization.use_dynamics_hess = 1 if self.config.use_dynamics_hess else 0
        planner_settings.pass2.regularization.use_dynamics_hess = 1 if self.config.use_dynamics_hess else 0
        
        # Create controller
        controller = Plan_and_Track_LQR(est_sat=satellite, planner_settings=planner_settings)
        
        # Set quaternion to 3-vec mode for cost computation
        # Mode 2 (Cayley) is recommended for quaternion goals
        if self.config.use_quaternion_goal:
            controller.planner.setquaternionTo3VecMode(self.config.quat_to_3vec_mode)
        
        # Setup goals
        if goals is None:
            q_goal = x_goal[3:7]
            q_goal = q_goal / np.linalg.norm(q_goal)
            
            if self.config.use_quaternion_goal:
                # Use Fixed_Attitude_Goal for full 3-DOF quaternion control
                goal = Fixed_Attitude_Goal(q_goal)
            else:
                # Fall back to 2-DOF pointing control
                goal_vec = self._quaternion_to_pointing_vector(q_goal, satellite.boresight)
                goal = ECI_Goal(goal_vec)
            goals = GoalList({orbital_state.J2000: goal})
        
        t_start = orbital_state.J2000
        duration = self.config.horizon
        
        # Calculate trajectory
        try:
            traj = controller.calculate_trajectory(
                t_start=t_start,
                duration=duration,
                x_0=x0,
                os_0=orbital_state,
                goals=goals,
                verbose=self.config.verbose,
            )
            converged = True
            
            # Extract trajectory data
            times = np.array(traj.times) if hasattr(traj, 'times') else np.linspace(0, duration, len(traj.states))
            states = np.array(traj.states).T if hasattr(traj, 'states') else traj.states
            controls = np.array(traj.controls).T if hasattr(traj, 'controls') else traj.controls
            gains = np.array(traj.gains) if hasattr(traj, 'gains') else None
            
            # Ensure correct shape
            if states.ndim == 1:
                states = states.reshape(-1, 1).T
            if len(states.shape) == 2 and states.shape[0] < states.shape[1]:
                states = states.T
            if controls.ndim == 1:
                controls = controls.reshape(-1, 1).T
            if len(controls.shape) == 2 and controls.shape[0] < controls.shape[1]:
                controls = controls.T
                
        except Exception as e:
            converged = False
            # Return empty result on failure
            N = int(duration / self.config.dt_tvlqr) + 1
            times = np.linspace(0, duration, N)
            states = np.zeros((N, len(x0)))
            states[0] = x0
            controls = np.zeros((N-1, len(u_max)))
            gains = None
        
        solve_time = time.perf_counter() - start_time
        
        # Compute metrics
        q_final = states[-1, 3:7] if states.shape[0] > 0 else x0[3:7]
        q_goal = x_goal[3:7]
        angle_error = self._quaternion_angle(q_final, q_goal)
        omega_error = np.linalg.norm(states[-1, :3] - x_goal[:3]) if states.shape[0] > 0 else 0.0
        
        # Compute control effort
        control_effort = np.sum(np.abs(controls)) if controls.size > 0 else 0.0
        
        return PlannerResult(
            times=times,
            states=states,
            controls=controls,
            solve_time=solve_time,
            converged=converged,
            iterations=-1,  # ALTRO doesn't expose iteration count easily
            final_cost=angle_error + omega_error,
            max_constraint_violation=0.0,  # Would need to extract from ALTRO
            gains=gains,
            solver_info={
                "final_angle_error_deg": np.degrees(angle_error),
                "final_omega_error": omega_error,
                "control_effort": control_effort,
                "dt_tp": self.config.dt_tp,
                "dt_tvlqr": self.config.dt_tvlqr,
            }
        )
    
    def solve_standalone(
        self,
        x0: NDArray[np.float64],
        x_goal: NDArray[np.float64],
        J_inertia: NDArray[np.float64],
        u_max: NDArray[np.float64],
        B_field: Optional[NDArray[np.float64]] = None,
        **kwargs
    ) -> PlannerResult:
        """
        Standalone solve without full ADCS infrastructure.
        
        This creates minimal satellite and orbit objects for testing.
        Use the full solve() method for production comparisons.
        """
        import time
        start_time = time.perf_counter()
        
        # Create minimal satellite (EstimatedSatellite required for planner)
        from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
        from ADCS.satellite_hardware.sensors import MTM
        from ADCS.satellite_hardware.actuators import RW, MTQ
        from ADCS.helpers.math_constants import MathConstants
        from ADCS.orbits.ephemeris import Ephemeris
        from ADCS.orbits.orbital_state import Orbital_State
        from ADCS.CONOPS.goals import ECI_Goal, Fixed_Attitude_Goal
        from ADCS.CONOPS.goallist import GoalList
        
        # Determine actuator configuration from u_max
        n_controls = len(u_max)
        n_rw = len(x0) - 7
        n_mtq = n_controls - n_rw
        
        # Create actuators
        actuators = []
        for i, axis in enumerate(MathConstants.unitvecs[:n_mtq]):
            actuators.append(MTQ(axis=axis, max_torque=u_max[i]))
        for i, axis in enumerate(MathConstants.unitvecs[:n_rw]):
            actuators.append(RW(axis=axis, max_torque=u_max[n_mtq + i], J=0.001, h=x0[7+i] if i < n_rw else 0.0, h_max=0.05))
        
        sensors = [MTM(axis=j) for j in MathConstants.unitvecs]
        
        satellite = EstimatedSatellite(
            mass=4.0,
            J_0=J_inertia,
            actuators=actuators,
            sensors=sensors,
            boresight=np.array([0, 0, 1])
        )
        
        # Create orbital state
        ephem = Ephemeris()
        orbital_state = Orbital_State(
            ephem=ephem,
            J2000=0.22,
            R=6778 * np.array([1, 0, 0]),
            V=np.array([0, 7.67, 0]),
            B=B_field[:3] if B_field is not None else np.array([2e-5, 1e-5, 3e-5]),
            S=np.array([1e5, 0, 0]),
            rho=0.0
        )
        
        # Create goal
        q_goal = x_goal[3:7]
        q_goal = q_goal / np.linalg.norm(q_goal)
        
        if self.config.use_quaternion_goal:
            goal = Fixed_Attitude_Goal(q_goal)
        else:
            goal_vec = self._quaternion_to_pointing_vector(q_goal, np.array([0, 0, 1]))
            goal = ECI_Goal(goal_vec)
        goals = GoalList({orbital_state.J2000: goal})
        
        return self.solve(
            x0=x0,
            x_goal=x_goal,
            J_inertia=J_inertia,
            u_max=u_max,
            B_field=B_field,
            satellite=satellite,
            orbital_state=orbital_state,
            goals=goals,
            **kwargs
        )
    
    def _quaternion_to_pointing_vector(
        self,
        q: NDArray[np.float64],
        boresight: NDArray[np.float64] = None
    ) -> NDArray[np.float64]:
        """
        Convert goal quaternion to pointing vector.
        
        For ALTRO, we need to specify where the satellite's boresight should point
        in the ECI frame. Given a goal quaternion q_goal that represents the
        desired body-to-ECI rotation, we compute where the boresight points.
        
        The goal quaternion rotates from body frame to ECI frame.
        If boresight is [0,0,1] in body frame, the goal vector in ECI is R @ [0,0,1].
        
        Args:
            q: Goal quaternion [qx, qy, qz, qw] (body-to-ECI rotation)
            boresight: Boresight vector in body frame (default [0,0,1])
            
        Returns:
            Goal pointing vector in ECI frame
        """
        if boresight is None:
            boresight = np.array([0, 0, 1])
        
        # Normalize
        q = q / np.linalg.norm(q)
        boresight = boresight / np.linalg.norm(boresight)
        
        # Quaternion to rotation matrix (body-to-ECI)
        qx, qy, qz, qw = q
        
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])
        
        # Transform boresight from body to ECI
        goal_eci = R @ boresight
        
        return goal_eci / np.linalg.norm(goal_eci)
    
    @staticmethod
    def _quaternion_angle(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> float:
        """Compute angle between two quaternions in radians."""
        dot = np.abs(np.dot(q1, q2))
        dot = np.clip(dot, -1.0, 1.0)
        return 2.0 * np.arccos(dot)
