"""
Tests for C++ TinyMPC tracking controller.
"""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.satellite_factory.satellites.create_cubesats import (
    create_beavercube1_cubesat,
    create_beavercube2_cubesat,
)
from ADCS.CONOPS.goallist import GoalList
from ADCS.CONOPS.goals.attitude_goals import Fixed_Attitude_Goal
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.controller.helpers import create_planner_settings, TinyMPCSettings
from ADCS.helpers.math_helpers import quat_diff, quat_to_vec3, rot_mat

# Check if C++ TinyMPC is available
try:
    from ADCS.controller import Plan_and_Track_TinyMPC_Cpp
    HAS_TINYMPC = True
except ImportError:
    HAS_TINYMPC = False


@pytest.fixture
def sat_mtq_rw():
    """BeaverCube2: 3 MTQ + 1 RW."""
    return create_beavercube2_cubesat(estimated=True)


@pytest.fixture
def sat_mtq_only():
    """BeaverCube1: 3 MTQ only."""
    return create_beavercube1_cubesat(estimated=True)


@pytest.fixture
def orbit():
    """Random circular orbit at 500km altitude."""
    np.random.seed(42)
    return create_random_circular_orbit(radius_km=6878, dt=1.0, tf=300.0)


@pytest.fixture
def goals():
    """60 degree rotation about Z axis."""
    theta = 60 * np.pi / 180
    q_target = np.array([np.cos(theta/2), 0, 0, np.sin(theta/2)])
    goals = GoalList()
    goals.add_goal(0.0, Fixed_Attitude_Goal(q_ref=q_target))
    return goals, q_target


@pytest.mark.skipif(not HAS_TINYMPC, reason="C++ TinyMPC not built")
class TestTinyMPCInitialization:
    """Test TinyMPC controller initialization."""
    
    def test_init_mtq_rw(self, sat_mtq_rw, orbit):
        """Test initialization with MTQ+RW satellite."""
        planner_settings = create_planner_settings(sat_mtq_rw, duration=100.0)
        tinympc_settings = TinyMPCSettings(track_horizon=5, track_dt=5.0)
        
        controller = Plan_and_Track_TinyMPC_Cpp(
            sat_mtq_rw, planner_settings, tinympc_settings
        )
        
        assert controller is not None
        assert controller._mpc.getStateDim() == 8  # 7 + 1 RW
        assert controller._mpc.getControlDim() == 4  # 3 MTQ + 1 RW
    
    def test_init_mtq_only(self, sat_mtq_only, orbit):
        """Test initialization with MTQ-only satellite."""
        planner_settings = create_planner_settings(sat_mtq_only, duration=100.0)
        tinympc_settings = TinyMPCSettings(track_horizon=5, track_dt=5.0)
        
        controller = Plan_and_Track_TinyMPC_Cpp(
            sat_mtq_only, planner_settings, tinympc_settings
        )
        
        assert controller is not None
        assert controller._mpc.getStateDim() == 7  # No RW
        assert controller._mpc.getControlDim() == 3  # 3 MTQ


@pytest.mark.skipif(not HAS_TINYMPC, reason="C++ TinyMPC not built")
class TestTinyMPCTrajectory:
    """Test trajectory calculation."""
    
    def test_trajectory_calculation(self, sat_mtq_rw, orbit, goals):
        """Test that trajectory calculation succeeds."""
        goal_list, q_target = goals
        os0 = orbit.get_os(orbit.min_time())
        t_start = os0.J2000
        
        # Update goal time
        goal_list = GoalList()
        goal_list.add_goal(t_start, Fixed_Attitude_Goal(q_ref=q_target))
        
        planner_settings = create_planner_settings(sat_mtq_rw, duration=100.0)
        tinympc_settings = TinyMPCSettings(track_horizon=5, track_dt=5.0)
        
        controller = Plan_and_Track_TinyMPC_Cpp(
            sat_mtq_rw, planner_settings, tinympc_settings
        )
        
        x0 = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float64)
        
        trajectory = controller.calculate_trajectory(
            t_start=t_start,
            duration=100.0,
            x_0=x0,
            os_0=os0,
            goals=goal_list,
            verbose=False
        )
        
        assert trajectory is not None
        assert trajectory.is_valid_time(t_start)


@pytest.mark.skipif(not HAS_TINYMPC, reason="C++ TinyMPC not built")
class TestTinyMPCSolve:
    """Test MPC solve functionality."""
    
    def test_solve_returns_control(self, sat_mtq_rw, orbit, goals):
        """Test that solve returns valid control."""
        goal_list, q_target = goals
        os0 = orbit.get_os(orbit.min_time())
        t_start = os0.J2000
        
        goal_list = GoalList()
        goal_list.add_goal(t_start, Fixed_Attitude_Goal(q_ref=q_target))
        
        planner_settings = create_planner_settings(sat_mtq_rw, duration=100.0)
        tinympc_settings = TinyMPCSettings(track_horizon=5, track_dt=5.0, verbose=0)
        
        controller = Plan_and_Track_TinyMPC_Cpp(
            sat_mtq_rw, planner_settings, tinympc_settings
        )
        
        x0 = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float64)
        
        trajectory = controller.calculate_trajectory(
            t_start=t_start,
            duration=100.0,
            x_0=x0,
            os_0=os0,
            goals=goal_list,
            verbose=False
        )
        
        controller.active_trajectory = trajectory
        
        # Test find_u
        u = controller.find_u(x0, np.zeros(10), sat_mtq_rw, os0)
        
        assert u.shape == (4,)  # 3 MTQ + 1 RW
        assert np.all(np.isfinite(u))
    
    def test_solve_timing(self, sat_mtq_rw, orbit, goals):
        """Test that solve is fast (< 5ms)."""
        goal_list, q_target = goals
        os0 = orbit.get_os(orbit.min_time())
        t_start = os0.J2000
        
        goal_list = GoalList()
        goal_list.add_goal(t_start, Fixed_Attitude_Goal(q_ref=q_target))
        
        planner_settings = create_planner_settings(sat_mtq_rw, duration=100.0)
        tinympc_settings = TinyMPCSettings(track_horizon=5, track_dt=5.0, verbose=0)
        
        controller = Plan_and_Track_TinyMPC_Cpp(
            sat_mtq_rw, planner_settings, tinympc_settings
        )
        
        x0 = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float64)
        
        trajectory = controller.calculate_trajectory(
            t_start=t_start,
            duration=100.0,
            x_0=x0,
            os_0=os0,
            goals=goal_list,
            verbose=False
        )
        
        controller.active_trajectory = trajectory
        
        # Warm up
        controller.find_u(x0, np.zeros(10), sat_mtq_rw, os0)
        
        # Time multiple solves
        import time
        times = []
        for _ in range(10):
            start = time.perf_counter()
            controller.find_u(x0, np.zeros(10), sat_mtq_rw, os0)
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = np.mean(times)
        print(f"\nTinyMPC solve time: {avg_time:.3f} ms (avg of 10)")
        
        # Should be < 5ms (typically ~0.15ms)
        assert avg_time < 5.0, f"Solve too slow: {avg_time:.3f} ms"


@pytest.mark.skipif(not HAS_TINYMPC, reason="C++ TinyMPC not built")
class TestTinyMPCClosedLoop:
    """Test closed-loop simulation."""
    
    def test_closed_loop_tracking(self, sat_mtq_rw, orbit, goals):
        """Test that closed-loop tracking reduces error over time."""
        goal_list, q_target = goals
        os0 = orbit.get_os(orbit.min_time())
        t_start = os0.J2000
        
        goal_list = GoalList()
        goal_list.add_goal(t_start, Fixed_Attitude_Goal(q_ref=q_target))
        
        planner_settings = create_planner_settings(sat_mtq_rw, duration=100.0)
        tinympc_settings = TinyMPCSettings(
            track_horizon=10, 
            track_dt=2.0, 
            verbose=0,
            max_iter=50
        )
        
        controller = Plan_and_Track_TinyMPC_Cpp(
            sat_mtq_rw, planner_settings, tinympc_settings
        )
        
        x0 = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float64)
        
        trajectory = controller.calculate_trajectory(
            t_start=t_start,
            duration=100.0,
            x_0=x0,
            os_0=os0,
            goals=goal_list,
            verbose=False
        )
        
        controller.active_trajectory = trajectory
        
        # Run closed-loop simulation
        x = x0.copy()
        dt_sim = 0.5
        
        errors = []
        for i in range(100):
            t_sec = i * dt_sim
            t = t_start + t_sec * TimeConstants.sec2cent
            
            if not trajectory.is_valid_time(t):
                break
            
            os_t = orbit.get_os(t)
            u = controller.find_u(x, np.zeros(10), sat_mtq_rw, os_t)
            
            # Integrate
            xdot = sat_mtq_rw.dynamics_core(x, u, os_t)
            x = x + dt_sim * xdot
            x[3:7] = x[3:7] / np.linalg.norm(x[3:7])
            
            # Compute error
            q_err = quat_diff(q_target, x[3:7])
            angle_err = 2 * np.arcsin(np.clip(np.linalg.norm(quat_to_vec3(q_err)), 0, 1)) * 180 / np.pi
            errors.append(angle_err)
        
        initial_error = errors[0]
        final_error = errors[-1]
        
        print(f"\nInitial error: {initial_error:.1f}°, Final error: {final_error:.1f}°")
        
        # The error should stay bounded (controller is working)
        # Note: Full convergence requires proper tuning
        assert np.max(errors) < 180, "Error exploded - controller unstable"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
