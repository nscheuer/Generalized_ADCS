"""
Comprehensive tests for Computed Torque and MPC tracking controllers.

Tests verify:
1. Controller initialization for MTQ-only and MTQ+RW systems
2. Trajectory calculation with C++ and Python planners
3. Control computation correctness
4. Timing performance
5. Comparison with TVLQR baseline
6. Closed-loop simulation accuracy
"""
import pytest
import numpy as np
import sys
import time

sys.path.insert(0, '/home/pmckeen/Generalized_ADCS')
sys.path.insert(0, '/home/pmckeen/Generalized_ADCS/papers/Planner')

from scipy.spatial.transform import Rotation

from ADCS.CONOPS.goals import ECI_Goal, Fixed_Attitude_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_factory.satellites.create_cubesats import (
    create_beavercube1_cubesat,  # MTQ-only
    create_beavercube2_cubesat,  # 3MTQ + 1RW
)
from ADCS.controller import (
    Plan_and_Track_ComputedTorque,
    Plan_and_Track_MPC,
    Plan_and_Track_LQR,
    MPCParams,
)
from ADCS.helpers.math_helpers import normalize, rot_mat, quat_diff


def angle_between_quats(q1: np.ndarray, q2: np.ndarray) -> float:
    """Compute angle between two quaternions in radians."""
    q_err = quat_diff(q1, q2)
    # Ensure positive scalar part
    if q_err[0] < 0:
        q_err = -q_err
    # angle = 2 * arccos(|q0|), clamped for numerical stability
    return 2 * np.arccos(np.clip(abs(q_err[0]), -1, 1))
from ADCS.orbits.universal_constants import TimeConstants

# Import planner settings
from mc_planner_settings import create_optimized_planner_settings


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def orbit():
    """Create a test orbit (shared across tests for speed)."""
    np.random.seed(42)
    orb = create_random_circular_orbit(
        radius_km=7000.0, dt=0.5, tf=250, use_J2=True, fast=True
    )
    orb.populate_environment(compute_B=True, compute_S=True)
    return orb


@pytest.fixture
def mtq_only_sat():
    """BeaverCube1: MTQ-only satellite."""
    return create_beavercube1_cubesat(estimated=False)


@pytest.fixture
def mtq_rw_sat():
    """BeaverCube2: 3MTQ + 1RW satellite."""
    return create_beavercube2_cubesat(estimated=False)


@pytest.fixture
def initial_state_mtq():
    """Initial state for MTQ-only (7D)."""
    q0 = normalize(np.array([1.0, 0.3, 0.2, 0.1]))
    return np.concatenate([[0.002, -0.001, 0.001], q0])


@pytest.fixture
def initial_state_rw():
    """Initial state for MTQ+RW (8D)."""
    q0 = normalize(np.array([1.0, 0.3, 0.2, 0.1]))
    return np.concatenate([[0.002, -0.001, 0.001], q0, [0.0]])


@pytest.fixture
def eci_goal():
    """Simple ECI pointing goal."""
    return GoalList({0.22: ECI_Goal(np.array([1.0, 0.0, 0.0]))})


@pytest.fixture
def quat_goal():
    """Fixed quaternion goal."""
    q_goal = normalize(np.array([1.0, 0.0, 0.0, 0.0]))
    return GoalList({0.22: Fixed_Attitude_Goal(q_goal)})


# =============================================================================
# Initialization Tests
# =============================================================================

class TestInitialization:
    """Test controller initialization."""
    
    def test_computed_torque_init_mtq_only(self, mtq_only_sat, orbit):
        """Test ComputedTorque initialization for MTQ-only system."""
        os0 = orbit.get_os(0.22)
        settings = create_optimized_planner_settings(mtq_only_sat, duration=100)
        
        ctrl = Plan_and_Track_ComputedTorque(mtq_only_sat, settings)
        
        assert ctrl._n_mtq == 3
        assert ctrl._n_rw == 0
        assert ctrl._has_rw == False
        assert ctrl._m_max > 0
        assert ctrl._J.shape == (3, 3)
    
    def test_computed_torque_init_mtq_rw(self, mtq_rw_sat, orbit):
        """Test ComputedTorque initialization for MTQ+RW system."""
        os0 = orbit.get_os(0.22)
        settings = create_optimized_planner_settings(mtq_rw_sat, duration=100)
        
        ctrl = Plan_and_Track_ComputedTorque(mtq_rw_sat, settings)
        
        assert ctrl._n_mtq == 3
        assert ctrl._n_rw == 1
        assert ctrl._has_rw == True
        assert ctrl._rw_axes is not None
        assert ctrl._rw_u_max is not None
    
    def test_mpc_init_mtq_only(self, mtq_only_sat, orbit):
        """Test MPC initialization for MTQ-only system."""
        settings = create_optimized_planner_settings(mtq_only_sat, duration=100)
        
        ctrl = Plan_and_Track_MPC(mtq_only_sat, settings)
        
        assert ctrl._n_mtq == 3
        assert ctrl._n_rw == 0
        assert ctrl._J is not None
    
    def test_mpc_init_mtq_rw(self, mtq_rw_sat, orbit):
        """Test MPC initialization for MTQ+RW system."""
        settings = create_optimized_planner_settings(mtq_rw_sat, duration=100)
        
        ctrl = Plan_and_Track_MPC(mtq_rw_sat, settings)
        
        assert ctrl._n_mtq == 3
        assert ctrl._n_rw == 1
    
    def test_mpc_params_presets(self):
        """Test MPCParams preset methods."""
        fast = MPCParams.fast()
        assert fast.max_iter == 20
        assert fast.use_tvlqr_weights == False
        
        accurate = MPCParams.accurate()
        assert accurate.max_iter == 100
        
        balanced = MPCParams.balanced()
        assert balanced.Q_attitude == 100.0
        assert balanced.use_tvlqr_weights == True


# =============================================================================
# Trajectory Calculation Tests
# =============================================================================

class TestTrajectoryCalculation:
    """Test trajectory planning functionality."""
    
    def test_computed_torque_trajectory_mtq_only(
        self, mtq_only_sat, orbit, initial_state_mtq, eci_goal
    ):
        """Test trajectory calculation for MTQ-only with ComputedTorque."""
        os0 = orbit.get_os(0.22)
        settings = create_optimized_planner_settings(mtq_only_sat, duration=100)
        
        ctrl = Plan_and_Track_ComputedTorque(mtq_only_sat, settings)
        traj = ctrl.calculate_trajectory(
            0.22, 100, initial_state_mtq, os0, eci_goal, verbose=False
        )
        
        assert traj is not None
        # Trajectory created successfully
        assert traj.states.shape[0] == 7  # MTQ-only state dim
        assert traj.controls.shape[0] == 3  # MTQ controls
        assert traj.gains is not None
    
    def test_computed_torque_trajectory_mtq_rw(
        self, mtq_rw_sat, orbit, initial_state_rw, eci_goal
    ):
        """Test trajectory calculation for MTQ+RW with ComputedTorque."""
        os0 = orbit.get_os(0.22)
        settings = create_optimized_planner_settings(mtq_rw_sat, duration=100)
        
        ctrl = Plan_and_Track_ComputedTorque(mtq_rw_sat, settings)
        traj = ctrl.calculate_trajectory(
            0.22, 100, initial_state_rw, os0, eci_goal, verbose=False
        )
        
        assert traj is not None
        # Trajectory created successfully
        assert traj.states.shape[0] == 8  # MTQ+RW state dim
        assert traj.controls.shape[0] == 4  # 3 MTQ + 1 RW
    
    def test_mpc_trajectory_mtq_only(
        self, mtq_only_sat, orbit, initial_state_mtq, eci_goal
    ):
        """Test trajectory calculation for MTQ-only with MPC."""
        os0 = orbit.get_os(0.22)
        settings = create_optimized_planner_settings(mtq_only_sat, duration=100)
        
        ctrl = Plan_and_Track_MPC(mtq_only_sat, settings)
        traj = ctrl.calculate_trajectory(
            0.22, 100, initial_state_mtq, os0, eci_goal, verbose=False
        )
        
        assert traj is not None
        # Trajectory created successfully
    
    def test_trajectory_with_quat_goal(
        self, mtq_only_sat, orbit, initial_state_mtq, quat_goal
    ):
        """Test trajectory with fixed quaternion goal."""
        os0 = orbit.get_os(0.22)
        settings = create_optimized_planner_settings(mtq_only_sat, duration=100)
        
        ctrl = Plan_and_Track_ComputedTorque(mtq_only_sat, settings)
        traj = ctrl.calculate_trajectory(
            0.22, 100, initial_state_mtq, os0, quat_goal, verbose=False
        )
        
        assert traj is not None
        # Trajectory created successfully


# =============================================================================
# Control Computation Tests
# =============================================================================

class TestControlComputation:
    """Test control computation (find_u)."""
    
    def test_computed_torque_find_u_mtq_only(
        self, mtq_only_sat, orbit, initial_state_mtq, eci_goal
    ):
        """Test find_u for MTQ-only with ComputedTorque."""
        os0 = orbit.get_os(0.22)
        settings = create_optimized_planner_settings(mtq_only_sat, duration=100)
        
        ctrl = Plan_and_Track_ComputedTorque(mtq_only_sat, settings)
        traj = ctrl.calculate_trajectory(
            0.22, 100, initial_state_mtq, os0, eci_goal, verbose=False
        )
        ctrl.set_active_trajectory(traj)
        
        # Compute B_body
        q = initial_state_mtq[3:7]
        R = rot_mat(q)
        B_body = R.T @ os0.B
        
        u = ctrl.find_u(initial_state_mtq, None, mtq_only_sat, os0, B_body=B_body)
        
        assert u.shape == (3,)
        assert np.all(np.abs(u) <= ctrl._m_max * 1.01)  # Within limits (small tolerance)
    
    def test_computed_torque_find_u_mtq_rw(
        self, mtq_rw_sat, orbit, initial_state_rw, eci_goal
    ):
        """Test find_u for MTQ+RW with ComputedTorque."""
        os0 = orbit.get_os(0.22)
        settings = create_optimized_planner_settings(mtq_rw_sat, duration=100)
        
        ctrl = Plan_and_Track_ComputedTorque(mtq_rw_sat, settings)
        traj = ctrl.calculate_trajectory(
            0.22, 100, initial_state_rw, os0, eci_goal, verbose=False
        )
        ctrl.set_active_trajectory(traj)
        
        q = initial_state_rw[3:7]
        R = rot_mat(q)
        B_body = R.T @ os0.B
        
        u = ctrl.find_u(initial_state_rw, None, mtq_rw_sat, os0, B_body=B_body)
        
        assert u.shape == (4,)  # 3 MTQ + 1 RW
        assert np.all(np.abs(u[:3]) <= ctrl._m_max * 1.01)
    
    def test_mpc_find_u_mtq_only(
        self, mtq_only_sat, orbit, initial_state_mtq, eci_goal
    ):
        """Test find_u for MTQ-only with MPC."""
        os0 = orbit.get_os(0.22)
        settings = create_optimized_planner_settings(mtq_only_sat, duration=100)
        
        ctrl = Plan_and_Track_MPC(mtq_only_sat, settings)
        traj = ctrl.calculate_trajectory(
            0.22, 100, initial_state_mtq, os0, eci_goal, verbose=False
        )
        ctrl.set_active_trajectory(traj)
        
        q = initial_state_mtq[3:7]
        R = rot_mat(q)
        B_body = R.T @ os0.B
        
        u = ctrl.find_u(initial_state_mtq, None, mtq_only_sat, os0, B_body=B_body)
        
        assert u.shape == (3,)
        assert np.all(np.abs(u) <= ctrl._m_max * 1.01)
    
    def test_mpc_find_u_mtq_rw(
        self, mtq_rw_sat, orbit, initial_state_rw, eci_goal
    ):
        """Test find_u for MTQ+RW with MPC."""
        os0 = orbit.get_os(0.22)
        settings = create_optimized_planner_settings(mtq_rw_sat, duration=100)
        
        ctrl = Plan_and_Track_MPC(mtq_rw_sat, settings)
        traj = ctrl.calculate_trajectory(
            0.22, 100, initial_state_rw, os0, eci_goal, verbose=False
        )
        ctrl.set_active_trajectory(traj)
        
        q = initial_state_rw[3:7]
        R = rot_mat(q)
        B_body = R.T @ os0.B
        
        u = ctrl.find_u(initial_state_rw, None, mtq_rw_sat, os0, B_body=B_body)
        
        assert u.shape == (4,)
    
    def test_find_u_without_trajectory_returns_zeros(
        self, mtq_only_sat, orbit, initial_state_mtq
    ):
        """Test that find_u returns zeros when no trajectory is set."""
        os0 = orbit.get_os(0.22)
        settings = create_optimized_planner_settings(mtq_only_sat, duration=100)
        
        ctrl = Plan_and_Track_ComputedTorque(mtq_only_sat, settings)
        # Don't set trajectory
        
        u = ctrl.find_u(initial_state_mtq, None, mtq_only_sat, os0)
        
        assert u.shape == (3,)
        assert np.allclose(u, 0)


# =============================================================================
# Timing Performance Tests
# =============================================================================

class TestTimingPerformance:
    """Test computational performance of controllers."""
    
    def test_computed_torque_timing(
        self, mtq_only_sat, orbit, initial_state_mtq, eci_goal
    ):
        """Verify ComputedTorque is fast (<500 µs)."""
        os0 = orbit.get_os(0.22)
        settings = create_optimized_planner_settings(mtq_only_sat, duration=100)
        
        ctrl = Plan_and_Track_ComputedTorque(mtq_only_sat, settings)
        traj = ctrl.calculate_trajectory(
            0.22, 100, initial_state_mtq, os0, eci_goal, verbose=False
        )
        ctrl.set_active_trajectory(traj)
        
        q = initial_state_mtq[3:7]
        R = rot_mat(q)
        B_body = R.T @ os0.B
        
        # Warm up
        for _ in range(10):
            ctrl.find_u(initial_state_mtq, None, mtq_only_sat, os0, B_body=B_body)
        
        # Time
        N = 100
        t0 = time.perf_counter()
        for _ in range(N):
            ctrl.find_u(initial_state_mtq, None, mtq_only_sat, os0, B_body=B_body)
        elapsed = (time.perf_counter() - t0) / N * 1e6  # µs
        
        print(f"ComputedTorque timing: {elapsed:.1f} µs")
        assert elapsed < 500, f"ComputedTorque too slow: {elapsed:.1f} µs"
    
    def test_mpc_timing(
        self, mtq_only_sat, orbit, initial_state_mtq, eci_goal
    ):
        """Verify MPC timing is reasonable (<10 ms)."""
        os0 = orbit.get_os(0.22)
        settings = create_optimized_planner_settings(mtq_only_sat, duration=100)
        
        ctrl = Plan_and_Track_MPC(mtq_only_sat, settings)
        traj = ctrl.calculate_trajectory(
            0.22, 100, initial_state_mtq, os0, eci_goal, verbose=False
        )
        ctrl.set_active_trajectory(traj)
        
        q = initial_state_mtq[3:7]
        R = rot_mat(q)
        B_body = R.T @ os0.B
        
        # Warm up
        for _ in range(5):
            ctrl.find_u(initial_state_mtq, None, mtq_only_sat, os0, B_body=B_body)
        
        # Time
        N = 20
        t0 = time.perf_counter()
        for _ in range(N):
            ctrl.find_u(initial_state_mtq, None, mtq_only_sat, os0, B_body=B_body)
        elapsed = (time.perf_counter() - t0) / N * 1e3  # ms
        
        print(f"MPC timing: {elapsed:.2f} ms")
        assert elapsed < 10, f"MPC too slow: {elapsed:.2f} ms"
    
    def test_tvlqr_timing_baseline(
        self, mtq_only_sat, orbit, initial_state_mtq, eci_goal
    ):
        """Measure TVLQR timing as baseline."""
        os0 = orbit.get_os(0.22)
        settings = create_optimized_planner_settings(mtq_only_sat, duration=100)
        
        ctrl = Plan_and_Track_LQR(mtq_only_sat, settings)
        traj = ctrl.calculate_trajectory(
            0.22, 100, initial_state_mtq, os0, eci_goal, verbose=False
        )
        ctrl.set_active_trajectory(traj)
        
        # Warm up
        for _ in range(10):
            ctrl.find_u(initial_state_mtq, None, mtq_only_sat, os0)
        
        # Time
        N = 100
        t0 = time.perf_counter()
        for _ in range(N):
            ctrl.find_u(initial_state_mtq, None, mtq_only_sat, os0)
        elapsed = (time.perf_counter() - t0) / N * 1e6  # µs
        
        print(f"TVLQR timing: {elapsed:.1f} µs")
        assert elapsed < 500, f"TVLQR too slow: {elapsed:.1f} µs"


# =============================================================================
# Closed-Loop Simulation Tests
# =============================================================================

class TestClosedLoopSimulation:
    """Test closed-loop tracking accuracy."""
    
    def _simulate(self, ctrl, sat, orbit, x0, t_start, duration, dt_sim=1.0):
        """Run a simple closed-loop simulation."""
        t = t_start
        x = x0.copy()
        t_end = t_start + duration * TimeConstants.sec2cent
        
        trajectory_errors = []
        
        while t < t_end:
            os = orbit.get_os(t)
            
            # Get reference
            x_ref = ctrl.active_trajectory.get_state_at(t)
            
            # Compute error
            q_curr = x[3:7]
            q_ref = x_ref[3:7]
            angle_err = angle_between_quats(q_curr, q_ref) * 180 / np.pi
            trajectory_errors.append(angle_err)
            
            # Compute control
            q = x[3:7]
            R = rot_mat(q)
            B_body = R.T @ os.B
            
            u = ctrl.find_u(x, None, sat, os, B_body=B_body)
            
            # Simple Euler integration
            xdot = sat.dynamics_core(x, u, os)
            x = x + xdot * dt_sim
            x[3:7] = x[3:7] / np.linalg.norm(x[3:7])  # Normalize quaternion
            
            t += dt_sim * TimeConstants.sec2cent
        
        return np.array(trajectory_errors)
    
    def test_computed_torque_tracking_mtq_only(
        self, mtq_only_sat, orbit, initial_state_mtq, eci_goal
    ):
        """Test closed-loop tracking with ComputedTorque for MTQ-only."""
        os0 = orbit.get_os(0.22)
        settings = create_optimized_planner_settings(mtq_only_sat, duration=100)
        
        ctrl = Plan_and_Track_ComputedTorque(mtq_only_sat, settings)
        traj = ctrl.calculate_trajectory(
            0.22, 100, initial_state_mtq, os0, eci_goal, verbose=False
        )
        ctrl.set_active_trajectory(traj)
        
        errors = self._simulate(ctrl, mtq_only_sat, orbit, initial_state_mtq, 0.22, 100)
        
        final_error = errors[-1]
        max_error = np.max(errors)
        
        print(f"ComputedTorque MTQ-only: final={final_error:.1f}°, max={max_error:.1f}°")
        
        # Should track reasonably well
        assert final_error < 30, f"Final error too large: {final_error:.1f}°"
    
    def test_computed_torque_tracking_mtq_rw(
        self, mtq_rw_sat, orbit, initial_state_rw, eci_goal
    ):
        """Test closed-loop tracking with ComputedTorque for MTQ+RW."""
        os0 = orbit.get_os(0.22)
        settings = create_optimized_planner_settings(mtq_rw_sat, duration=100)
        
        ctrl = Plan_and_Track_ComputedTorque(mtq_rw_sat, settings)
        traj = ctrl.calculate_trajectory(
            0.22, 100, initial_state_rw, os0, eci_goal, verbose=False
        )
        ctrl.set_active_trajectory(traj)
        
        errors = self._simulate(ctrl, mtq_rw_sat, orbit, initial_state_rw, 0.22, 100)
        
        final_error = errors[-1]
        max_error = np.max(errors)
        
        print(f"ComputedTorque MTQ+RW: final={final_error:.1f}°, max={max_error:.1f}°")
        
        # MTQ+RW should track better
        assert final_error < 20, f"Final error too large: {final_error:.1f}°"
    
    def test_mpc_tracking_mtq_only(
        self, mtq_only_sat, orbit, initial_state_mtq, eci_goal
    ):
        """Test closed-loop tracking with MPC for MTQ-only."""
        os0 = orbit.get_os(0.22)
        settings = create_optimized_planner_settings(mtq_only_sat, duration=100)
        
        ctrl = Plan_and_Track_MPC(mtq_only_sat, settings)
        traj = ctrl.calculate_trajectory(
            0.22, 100, initial_state_mtq, os0, eci_goal, verbose=False
        )
        ctrl.set_active_trajectory(traj)
        
        errors = self._simulate(ctrl, mtq_only_sat, orbit, initial_state_mtq, 0.22, 100)
        
        final_error = errors[-1]
        max_error = np.max(errors)
        
        print(f"MPC MTQ-only: final={final_error:.1f}°, max={max_error:.1f}°")
        
        assert final_error < 30, f"Final error too large: {final_error:.1f}°"


# =============================================================================
# Comparison Tests
# =============================================================================

class TestControllerComparison:
    """Compare different controllers."""
    
    def test_computed_torque_vs_tvlqr_mtq_only(
        self, mtq_only_sat, orbit, initial_state_mtq, eci_goal
    ):
        """Compare ComputedTorque vs TVLQR for MTQ-only."""
        os0 = orbit.get_os(0.22)
        settings = create_optimized_planner_settings(mtq_only_sat, duration=100)
        
        # Create controllers
        ctrl_ct = Plan_and_Track_ComputedTorque(mtq_only_sat, settings)
        ctrl_lqr = Plan_and_Track_LQR(mtq_only_sat, settings)
        
        # Plan trajectory (use same for both)
        traj = ctrl_ct.calculate_trajectory(
            0.22, 100, initial_state_mtq, os0, eci_goal, verbose=False
        )
        ctrl_ct.set_active_trajectory(traj)
        ctrl_lqr.set_active_trajectory(traj)
        
        # Compute controls at same state
        q = initial_state_mtq[3:7]
        R = rot_mat(q)
        B_body = R.T @ os0.B
        
        u_ct = ctrl_ct.find_u(initial_state_mtq, None, mtq_only_sat, os0, B_body=B_body)
        u_lqr = ctrl_lqr.find_u(initial_state_mtq, None, mtq_only_sat, os0)
        
        print(f"ComputedTorque: {u_ct}")
        print(f"TVLQR:          {u_lqr}")
        print(f"Difference:     {u_ct - u_lqr}")
        
        # They should be different (CT uses actual B-field)
        # But both should be reasonable
        assert np.all(np.abs(u_ct) <= ctrl_ct._m_max * 1.01)
        assert np.all(np.abs(u_lqr) <= ctrl_ct._m_max * 1.01)
    
    def test_mpc_vs_computed_torque(
        self, mtq_only_sat, orbit, initial_state_mtq, eci_goal
    ):
        """Compare MPC vs ComputedTorque."""
        os0 = orbit.get_os(0.22)
        settings = create_optimized_planner_settings(mtq_only_sat, duration=100)
        
        ctrl_mpc = Plan_and_Track_MPC(mtq_only_sat, settings)
        ctrl_ct = Plan_and_Track_ComputedTorque(mtq_only_sat, settings)
        
        # Plan trajectory
        traj = ctrl_ct.calculate_trajectory(
            0.22, 100, initial_state_mtq, os0, eci_goal, verbose=False
        )
        ctrl_mpc.set_active_trajectory(traj)
        ctrl_ct.set_active_trajectory(traj)
        
        q = initial_state_mtq[3:7]
        R = rot_mat(q)
        B_body = R.T @ os0.B
        
        u_mpc = ctrl_mpc.find_u(initial_state_mtq, None, mtq_only_sat, os0, B_body=B_body)
        u_ct = ctrl_ct.find_u(initial_state_mtq, None, mtq_only_sat, os0, B_body=B_body)
        
        print(f"MPC:            {u_mpc}")
        print(f"ComputedTorque: {u_ct}")
        print(f"Difference:     {u_mpc - u_ct}")
        
        # Both should produce valid controls
        assert np.all(np.abs(u_mpc) <= ctrl_mpc._m_max * 1.01)
        assert np.all(np.abs(u_ct) <= ctrl_ct._m_max * 1.01)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_expired_trajectory_raises_error(
        self, mtq_only_sat, orbit, initial_state_mtq, eci_goal
    ):
        """Test that expired trajectory raises RuntimeError."""
        os0 = orbit.get_os(0.22)
        settings = create_optimized_planner_settings(mtq_only_sat, duration=50)
        
        ctrl = Plan_and_Track_ComputedTorque(mtq_only_sat, settings)
        traj = ctrl.calculate_trajectory(
            0.22, 50, initial_state_mtq, os0, eci_goal, verbose=False
        )
        ctrl.set_active_trajectory(traj)
        
        # Get orbital state way past trajectory end
        os_late = orbit.get_os(0.22 + 200 * TimeConstants.sec2cent)
        
        with pytest.raises(RuntimeError, match="expired"):
            ctrl.find_u(initial_state_mtq, None, mtq_only_sat, os_late)
    
    def test_auto_compute_b_body(
        self, mtq_only_sat, orbit, initial_state_mtq, eci_goal
    ):
        """Test that B_body is auto-computed if not provided."""
        os0 = orbit.get_os(0.22)
        settings = create_optimized_planner_settings(mtq_only_sat, duration=100)
        
        ctrl = Plan_and_Track_ComputedTorque(mtq_only_sat, settings)
        traj = ctrl.calculate_trajectory(
            0.22, 100, initial_state_mtq, os0, eci_goal, verbose=False
        )
        ctrl.set_active_trajectory(traj)
        
        # Call without B_body - should auto-compute
        u = ctrl.find_u(initial_state_mtq, None, mtq_only_sat, os0)
        
        assert u.shape == (3,)
        assert not np.allclose(u, 0)  # Should produce non-zero control
    
    def test_custom_mpc_params(
        self, mtq_only_sat, orbit, initial_state_mtq, eci_goal
    ):
        """Test controller with custom MPCParams."""
        os0 = orbit.get_os(0.22)
        settings = create_optimized_planner_settings(mtq_only_sat, duration=100)
        
        custom_params = MPCParams(
            Q_omega=10.0,
            Q_attitude=1000.0,
            R_mtq=0.001,
            use_tvlqr_weights=False
        )
        
        ctrl = Plan_and_Track_ComputedTorque(
            mtq_only_sat, settings, mpc_params=custom_params
        )
        
        assert ctrl.params.Q_omega == 10.0
        assert ctrl.params.Q_attitude == 1000.0
        assert ctrl.params.use_tvlqr_weights == False


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
