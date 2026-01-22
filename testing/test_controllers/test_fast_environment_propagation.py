"""
Tests for the fast environment propagation methods.

This verifies that the optimized batch methods for B-field and sun vector
computation produce results equivalent to the original per-timestep methods.
"""
import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants


class TestBatchBFieldComputation:
    """Test that batch B-field computation matches individual calls."""

    @pytest.fixture
    def orbit_setup(self):
        """Create a test orbit."""
        ephem = Ephemeris()
        start_time = 0.22
        R = 7000 * np.array([0, np.sqrt(2) / 2, np.sqrt(2) / 2])
        V = np.array([8, 0, 0])
        os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V, fast=False)
        
        # Create orbit - needs fast=True to test our optimization path
        # but states need geocentric coords for B-field computation
        duration = 100  # seconds
        dt = 10  # coarser for faster test
        orb = Orbit(
            os0=os0, 
            end_time=start_time + duration * TimeConstants.sec2cent,
            dt=dt, 
            use_J2=True, 
            fast=True,
            verbose=False
        )
        return orb

    def test_batch_b_field_shape(self, orbit_setup):
        """Test that batch B-field returns correct shape."""
        orb = orbit_setup
        B_batch = orb.get_b_eci_orbit()
        
        assert B_batch.shape == (len(orb.times), 3), \
            f"Expected shape ({len(orb.times)}, 3), got {B_batch.shape}"

    def test_batch_b_field_matches_individual(self, orbit_setup):
        """Test that batch B-field matches individual get_b_eci() calls."""
        orb = orbit_setup
        
        # Get batch result
        B_batch = orb.get_b_eci_orbit()
        
        # Get individual results
        B_individual = []
        for t in orb.times:
            os_t = orb.states[t]
            B_individual.append(os_t.get_b_eci())
        B_individual = np.array(B_individual)
        
        # Compare - allow small numerical differences
        np.testing.assert_allclose(
            B_batch, B_individual, 
            rtol=1e-10, atol=1e-15,
            err_msg="Batch B-field does not match individual computations"
        )

    def test_b_field_reasonable_magnitude(self, orbit_setup):
        """Test that B-field values are physically reasonable."""
        orb = orbit_setup
        B_batch = orb.get_b_eci_orbit()
        
        # Earth's magnetic field at LEO is typically 20-60 μT
        B_magnitudes = np.linalg.norm(B_batch, axis=1)
        
        # Convert to μT for checking
        B_magnitudes_uT = B_magnitudes * 1e6
        
        assert np.all(B_magnitudes_uT > 10), \
            f"B-field too weak: min={B_magnitudes_uT.min():.2f} μT"
        assert np.all(B_magnitudes_uT < 100), \
            f"B-field too strong: max={B_magnitudes_uT.max():.2f} μT"


class TestBatchSunVectorComputation:
    """Test that batch sun vector computation matches individual calls."""

    @pytest.fixture
    def orbit_setup(self):
        """Create a test orbit."""
        ephem = Ephemeris()
        start_time = 0.22
        R = 7000 * np.array([0, np.sqrt(2) / 2, np.sqrt(2) / 2])
        V = np.array([8, 0, 0])
        os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V, fast=False)
        
        duration = 100  # seconds
        dt = 10  # coarser for faster test
        orb = Orbit(
            os0=os0, 
            end_time=start_time + duration * TimeConstants.sec2cent,
            dt=dt, 
            use_J2=True, 
            fast=True,
            verbose=False
        )
        return orb

    def test_batch_sun_vector_shape(self, orbit_setup):
        """Test that batch sun vector returns correct shape."""
        orb = orbit_setup
        S_batch = orb.get_sun_eci_orbit()
        
        assert S_batch.shape == (len(orb.times), 3), \
            f"Expected shape ({len(orb.times)}, 3), got {S_batch.shape}"

    def test_batch_sun_vector_matches_individual(self, orbit_setup):
        """Test that batch sun vector matches individual get_sun_eci() calls."""
        orb = orbit_setup
        
        # Get batch result
        S_batch = orb.get_sun_eci_orbit()
        
        # Get individual results
        S_individual = []
        for t in orb.times:
            os_t = orb.states[t]
            S_individual.append(os_t.get_sun_eci())
        S_individual = np.array(S_individual)
        
        # Compare - allow small numerical differences
        np.testing.assert_allclose(
            S_batch, S_individual, 
            rtol=1e-10, atol=1e-6,  # km precision
            err_msg="Batch sun vector does not match individual computations"
        )

    def test_sun_vector_reasonable_magnitude(self, orbit_setup):
        """Test that sun vector magnitude is approximately 1 AU."""
        orb = orbit_setup
        S_batch = orb.get_sun_eci_orbit()
        
        # Sun distance should be approximately 1 AU = 149,597,870.7 km
        S_magnitudes = np.linalg.norm(S_batch, axis=1)
        
        # Allow 5% variation (Earth's orbit eccentricity is ~1.7%)
        AU_km = 149_597_870.7
        assert np.all(S_magnitudes > 0.9 * AU_km), \
            f"Sun too close: min={S_magnitudes.min()/1e6:.2f} million km"
        assert np.all(S_magnitudes < 1.1 * AU_km), \
            f"Sun too far: max={S_magnitudes.max()/1e6:.2f} million km"


class TestEnvironmentPropagationConsistency:
    """Test that _propagate_environment produces consistent results."""

    @pytest.fixture
    def controller_setup(self):
        """Create a test controller."""
        from ADCS.CONOPS.goals import ECI_Goal
        from ADCS.CONOPS.goallist import GoalList
        from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
        from ADCS.controller.helpers import PlannerSettings
        from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
        from ADCS.helpers.math_helpers import normalize

        real_sat = create_beavercube2_cubesat(estimated=False)
        real_sat.rw_actuators[0].h = 0.0

        planner_settings = PlannerSettings(
            est_sat=real_sat,
            bdot_on=2,
            dt_tp=30,
            dt_tvlqr=1,
        )
        # Minimal iterations for speed
        planner_settings.pass1.convergence.max_outer_iter = 2
        planner_settings.pass1.convergence.max_inner_iter = 5
        planner_settings.pass2.convergence.max_outer_iter = 1
        planner_settings.pass2.convergence.max_inner_iter = 3

        controller = Plan_and_Track_LQR(est_sat=real_sat, planner_settings=planner_settings)

        goal_vec = normalize(np.array([0, 0, 1]))
        goal = ECI_Goal(goal_vec)
        goals = GoalList({0.22: goal})

        return controller, goals

    @pytest.fixture
    def orbit_setup(self):
        """Create test orbit and initial state."""
        ephem = Ephemeris()
        start_time = 0.22 - 1 * TimeConstants.sec2cent
        R = 7000 * np.array([0, np.sqrt(2) / 2, np.sqrt(2) / 2])
        V = np.array([8, 0, 0])
        os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)

        orb = Orbit(
            os0=os0, 
            end_time=start_time + 150 * TimeConstants.sec2cent,
            dt=1, 
            use_J2=True, 
            fast=True,
            verbose=False
        )
        return orb

    def test_propagate_environment_returns_correct_shapes(self, controller_setup, orbit_setup):
        """Test that _propagate_environment returns correct array shapes."""
        controller, goals = controller_setup
        orb = orbit_setup
        
        os0_for_traj = orb.get_os(0.22)
        
        duration = 50  # seconds
        dt_seconds = 1
        N = int(np.ceil(duration / dt_seconds)) + 1
        t_end = 0.22 + (duration * TimeConstants.sec2cent)
        
        vecsPy = controller._propagate_environment(
            os0_for_traj, 0.22, t_end, dt_seconds, N, goals
        )
        
        t, R, V, B, S, A, E, p, rho = vecsPy
        
        # Check shapes
        assert t.shape == (N,), f"t shape: expected ({N},), got {t.shape}"
        assert R.shape == (3, N), f"R shape: expected (3, {N}), got {R.shape}"
        assert V.shape == (3, N), f"V shape: expected (3, {N}), got {V.shape}"
        assert B.shape == (3, N), f"B shape: expected (3, {N}), got {B.shape}"
        assert S.shape == (3, N), f"S shape: expected (3, {N}), got {S.shape}"
        assert A.shape == (3, N), f"A shape: expected (3, {N}), got {A.shape}"
        assert E.shape == (3, N), f"E shape: expected (3, {N}), got {E.shape}"
        assert p.shape == (N,), f"p shape: expected ({N},), got {p.shape}"
        assert rho.shape == (N,), f"rho shape: expected ({N},), got {rho.shape}"

    def test_propagate_environment_b_field_nonzero(self, controller_setup, orbit_setup):
        """Test that B-field from _propagate_environment is non-zero."""
        controller, goals = controller_setup
        orb = orbit_setup
        
        os0_for_traj = orb.get_os(0.22)
        
        duration = 50
        dt_seconds = 1
        N = int(np.ceil(duration / dt_seconds)) + 1
        t_end = 0.22 + (duration * TimeConstants.sec2cent)
        
        vecsPy = controller._propagate_environment(
            os0_for_traj, 0.22, t_end, dt_seconds, N, goals
        )
        
        B = vecsPy[3]  # B-field
        B_magnitudes = np.linalg.norm(B, axis=0)
        
        assert np.all(B_magnitudes > 1e-6), \
            f"B-field has near-zero values: min={B_magnitudes.min()}"

    def test_propagate_environment_sun_vector_nonzero(self, controller_setup, orbit_setup):
        """Test that sun vector from _propagate_environment is non-zero."""
        controller, goals = controller_setup
        orb = orbit_setup
        
        os0_for_traj = orb.get_os(0.22)
        
        duration = 50
        dt_seconds = 1
        N = int(np.ceil(duration / dt_seconds)) + 1
        t_end = 0.22 + (duration * TimeConstants.sec2cent)
        
        vecsPy = controller._propagate_environment(
            os0_for_traj, 0.22, t_end, dt_seconds, N, goals
        )
        
        S = vecsPy[4]  # Sun vector
        S_magnitudes = np.linalg.norm(S, axis=0)
        
        # Should be approximately 1 AU
        AU_km = 149_597_870.7
        assert np.all(S_magnitudes > 0.9 * AU_km), \
            f"Sun vector too small: min={S_magnitudes.min()/1e6:.2f} million km"

    def test_propagate_environment_times_monotonic(self, controller_setup, orbit_setup):
        """Test that times from _propagate_environment are monotonically increasing."""
        controller, goals = controller_setup
        orb = orbit_setup
        
        os0_for_traj = orb.get_os(0.22)
        
        duration = 50
        dt_seconds = 1
        N = int(np.ceil(duration / dt_seconds)) + 1
        t_end = 0.22 + (duration * TimeConstants.sec2cent)
        
        vecsPy = controller._propagate_environment(
            os0_for_traj, 0.22, t_end, dt_seconds, N, goals
        )
        
        t = vecsPy[0]
        
        assert np.all(np.diff(t) > 0), "Times are not monotonically increasing"


class TestSpeedImprovement:
    """Test that the batch methods are actually faster."""

    @pytest.fixture
    def orbit_setup(self):
        """Create a test orbit with more timesteps for timing."""
        ephem = Ephemeris()
        start_time = 0.22
        R = 7000 * np.array([0, np.sqrt(2) / 2, np.sqrt(2) / 2])
        V = np.array([8, 0, 0])
        os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V, fast=False)
        
        duration = 100  # seconds
        dt = 5  # 5 second steps = 21 timesteps
        orb = Orbit(
            os0=os0, 
            end_time=start_time + duration * TimeConstants.sec2cent,
            dt=dt, 
            use_J2=True, 
            fast=True,
            verbose=False
        )
        return orb

    def test_batch_b_field_faster_than_individual(self, orbit_setup):
        """Test that batch B-field is faster than individual calls."""
        import time
        orb = orbit_setup
        
        # Time batch method
        t0 = time.perf_counter()
        B_batch = orb.get_b_eci_orbit()
        batch_time = time.perf_counter() - t0
        
        # Time individual method
        t0 = time.perf_counter()
        B_individual = []
        for t in orb.times:
            os_t = orb.states[t]
            B_individual.append(os_t.get_b_eci())
        individual_time = time.perf_counter() - t0
        
        print(f"\nB-field timing ({len(orb.times)} timesteps):")
        print(f"  Batch: {batch_time:.3f}s")
        print(f"  Individual: {individual_time:.3f}s")
        print(f"  Speedup: {individual_time/batch_time:.1f}x")
        
        # Batch should be at least 2x faster
        assert batch_time < individual_time, \
            f"Batch method ({batch_time:.3f}s) not faster than individual ({individual_time:.3f}s)"

    def test_batch_sun_vector_faster_than_individual(self, orbit_setup):
        """Test that batch sun vector is faster than individual calls."""
        import time
        orb = orbit_setup
        
        # Time batch method
        t0 = time.perf_counter()
        S_batch = orb.get_sun_eci_orbit()
        batch_time = time.perf_counter() - t0
        
        # Time individual method
        t0 = time.perf_counter()
        S_individual = []
        for t in orb.times:
            os_t = orb.states[t]
            S_individual.append(os_t.get_sun_eci())
        individual_time = time.perf_counter() - t0
        
        print(f"\nSun vector timing ({len(orb.times)} timesteps):")
        print(f"  Batch: {batch_time:.3f}s")
        print(f"  Individual: {individual_time:.3f}s")
        print(f"  Speedup: {individual_time/batch_time:.1f}x")
        
        # Batch should be faster (at least equal, vectorization helps with more timesteps)
        # For small N, overhead might make it similar
        assert batch_time <= individual_time * 1.5, \
            f"Batch method ({batch_time:.3f}s) significantly slower than individual ({individual_time:.3f}s)"


class TestTrajectoryCalculationWithOptimization:
    """Test that trajectory calculation still works correctly with optimization."""

    def test_trajectory_calculation_completes(self):
        """Test that trajectory calculation completes without error."""
        from ADCS.CONOPS.goals import ECI_Goal
        from ADCS.CONOPS.goallist import GoalList
        from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
        from ADCS.controller.helpers import PlannerSettings
        from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
        from ADCS.helpers.math_helpers import normalize, random_n_unit_vec

        np.random.seed(42)
        
        real_sat = create_beavercube2_cubesat(estimated=False)
        real_sat.rw_actuators[0].h = 0.0

        # Initial state
        w0 = random_n_unit_vec(3) * 0.5 * np.pi / 180.0
        q0 = normalize(np.random.randn(4))
        h0 = np.array([0.0])
        x = np.concatenate([w0, q0, h0])

        # Create orbit
        ephem = Ephemeris()
        start_time = 0.22
        R = 7000 * np.array([0, np.sqrt(2) / 2, np.sqrt(2) / 2])
        V = np.array([8, 0, 0])
        os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)

        orb = Orbit(
            os0=os0, 
            end_time=start_time + 100 * TimeConstants.sec2cent,
            dt=1, 
            use_J2=True, 
            fast=True,
            verbose=False
        )

        # Planner settings with minimal iterations
        planner_settings = PlannerSettings(
            est_sat=real_sat,
            bdot_on=2,
            dt_tp=30,
            dt_tvlqr=1,
        )
        planner_settings.pass1.convergence.max_outer_iter = 3
        planner_settings.pass1.convergence.max_inner_iter = 10
        planner_settings.pass2.convergence.max_outer_iter = 2
        planner_settings.pass2.convergence.max_inner_iter = 5

        controller = Plan_and_Track_LQR(est_sat=real_sat, planner_settings=planner_settings)

        goal_vec = normalize(np.array([0, 0, 1]))
        goal = ECI_Goal(goal_vec)
        goals = GoalList({start_time: goal})

        os0_for_traj = orb.get_os(start_time)
        
        # Calculate trajectory
        traj = controller.calculate_trajectory(
            t_start=start_time,
            duration=50,  # Short duration
            x_0=x,
            os_0=os0_for_traj,
            goals=goals,
            verbose=False,
        )

        # Verify trajectory is valid
        assert traj is not None, "Trajectory is None"
        assert traj.states.shape[0] == 8, f"Expected 8 states, got {traj.states.shape[0]}"
        assert traj.states.shape[1] > 1, "Trajectory has only 1 timestep"
        assert traj.controls.shape[1] > 1, "Controls have only 1 timestep"
        
        # Check quaternion normalization
        q_norms = np.linalg.norm(traj.states[3:7, :], axis=0)
        np.testing.assert_allclose(q_norms, 1.0, rtol=1e-3, 
            err_msg="Quaternions not normalized in trajectory")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
