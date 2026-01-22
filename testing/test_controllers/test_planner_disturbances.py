"""
Tests for planner surface-based disturbance integration.

These tests verify:
1. The planner can be initialized with SRP and drag disturbances
2. Surface geometry data is correctly passed from Python to C++
3. The physics calculations in C++ match the expected values
4. The planner runs successfully with disturbances enabled
"""

import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.CONOPS.goals import ECI_Goal, No_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import PlannerSettings
from ADCS.controller.helpers.build_csat import build_cpp_satellite
from ADCS.controller.helpers.planner_subsettings import CostWeights
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants, EarthConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.disturbances import (
    SRP_Disturbance, Drag_Disturbance, GeometryConfig, GeometryFace
)
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def create_mtq_satellite_with_srp():
    """Create MTQ satellite with SRP disturbance using a simple geometry."""
    # Single face on +Y with centroid at +X
    face = GeometryFace(
        area=1.0,
        centroid=np.array([1.0, 0.0, 0.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        eta_a=1.0,  # fully absorptive
        eta_d=0.0,
        eta_s=0.0
    )
    config = GeometryConfig(geometry_faces=[face])
    srp = SRP_Disturbance(config=config)

    mtq_max = 0.2
    mtqs = [MTQ(axis=j, max_torque=mtq_max) for j in MathConstants.unitvecs]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]

    return Satellite(
        mass=4.0,
        J_0=np.diagflat([0.1, 0.12, 0.15]),
        actuators=mtqs,
        sensors=mtms,
        disturbances=[srp],
        boresight=np.array([0, 0, 1]),
        COM=np.array([0.0, 0.0, 0.0])
    )


def create_mtq_satellite_with_drag():
    """Create MTQ satellite with drag disturbance using a simple geometry."""
    face = GeometryFace(
        area=1.0,
        centroid=np.array([1.0, 0.0, 0.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        CD=2.2
    )
    config = GeometryConfig(geometry_faces=[face])
    drag = Drag_Disturbance(config=config)

    mtq_max = 0.2
    mtqs = [MTQ(axis=j, max_torque=mtq_max) for j in MathConstants.unitvecs]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]

    return Satellite(
        mass=4.0,
        J_0=np.diagflat([0.1, 0.12, 0.15]),
        actuators=mtqs,
        sensors=mtms,
        disturbances=[drag],
        boresight=np.array([0, 0, 1]),
        COM=np.array([0.0, 0.0, 0.0])
    )


def create_mtq_satellite_with_both():
    """Create MTQ satellite with both SRP and drag disturbances."""
    # SRP face
    srp_face = GeometryFace(
        area=1.0,
        centroid=np.array([1.0, 0.0, 0.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        eta_a=0.3,
        eta_d=0.3,
        eta_s=0.4
    )
    srp_config = GeometryConfig(geometry_faces=[srp_face])
    srp = SRP_Disturbance(config=srp_config)

    # Drag face
    drag_face = GeometryFace(
        area=0.5,
        centroid=np.array([0.0, 1.0, 0.0]),
        normal=np.array([1.0, 0.0, 0.0]),
        CD=2.0
    )
    drag_config = GeometryConfig(geometry_faces=[drag_face])
    drag = Drag_Disturbance(config=drag_config)

    mtq_max = 0.2
    mtqs = [MTQ(axis=j, max_torque=mtq_max) for j in MathConstants.unitvecs]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]

    return Satellite(
        mass=4.0,
        J_0=np.diagflat([0.1, 0.12, 0.15]),
        actuators=mtqs,
        sensors=mtms,
        disturbances=[srp, drag],
        boresight=np.array([0, 0, 1]),
        COM=np.array([0.0, 0.0, 0.0])
    )


def create_orbit_with_density(duration: int = 200):
    """Create orbit with non-zero atmospheric density for drag testing."""
    ephem = Ephemeris()
    R = 7000 * np.array([1.0, 0.0, 0.0])  # Along +X
    V = np.array([0.0, 7.5, 0.0])  # Along +Y (orbital velocity)
    S = np.array([7000.0, 1e9, 0.0])  # Sun along +Y (for SRP)
    rho = 1e-12  # Typical LEO density

    os0 = Orbital_State(
        ephem=ephem,
        J2000=0.22,
        R=R, V=V,
        B=np.array([0.0, 3e-5, 2e-5]),
        S=S,
        rho=rho
    )

    orbs = [os0.copy() for _ in range(duration + 10)]
    for j in range(len(orbs)):
        orbs[j].J2000 = os0.J2000 + j * TimeConstants.sec2cent
    return Orbit(orbs), os0


# ==========================================
# TESTS: PLANNER SETTINGS SURFACE DATA EXTRACTION
# ==========================================

class TestPlannerSettingsSurfaceData:
    """Test that PlannerSettings correctly extracts surface data from disturbances."""

    def test_srp_surfaces_extracted(self):
        """Test SRP surface data extraction."""
        sat = create_mtq_satellite_with_srp()
        est_sat = EstimatedSatellite(
            mass=sat.mass, J_0=sat.J_0, actuators=sat.actuators,
            sensors=sat.sensors, disturbances=sat.disturbances, COM=sat.COM
        )

        settings = PlannerSettings(
            est_sat=est_sat,
            include_srp=True,
            include_drag=False
        )

        assert settings.srp_surfaces is not None
        assert settings.drag_surfaces is None

        # Verify surface geometry was extracted correctly
        # Note: Python has (N, 3), we transpose to (3, N) for C++
        assert settings.srp_surfaces['normals'].shape == (3, 1)
        assert settings.srp_surfaces['centroids'].shape == (3, 1)
        assert len(settings.srp_surfaces['areas']) == 1
        assert len(settings.srp_surfaces['eta_s']) == 1
        assert len(settings.srp_surfaces['eta_d']) == 1
        assert len(settings.srp_surfaces['eta_a']) == 1

        # Check actual values
        np.testing.assert_array_almost_equal(
            settings.srp_surfaces['normals'][:, 0],
            [0.0, 1.0, 0.0]
        )
        np.testing.assert_array_almost_equal(
            settings.srp_surfaces['centroids'][:, 0],
            [1.0, 0.0, 0.0]
        )
        assert settings.srp_surfaces['areas'][0] == 1.0
        assert settings.srp_surfaces['eta_a'][0] == 1.0

    def test_drag_surfaces_extracted(self):
        """Test drag surface data extraction."""
        sat = create_mtq_satellite_with_drag()
        est_sat = EstimatedSatellite(
            mass=sat.mass, J_0=sat.J_0, actuators=sat.actuators,
            sensors=sat.sensors, disturbances=sat.disturbances, COM=sat.COM
        )

        settings = PlannerSettings(
            est_sat=est_sat,
            include_srp=False,
            include_drag=True
        )

        assert settings.srp_surfaces is None
        assert settings.drag_surfaces is not None

        assert settings.drag_surfaces['normals'].shape == (3, 1)
        assert settings.drag_surfaces['centroids'].shape == (3, 1)
        assert len(settings.drag_surfaces['areas']) == 1
        assert len(settings.drag_surfaces['CDs']) == 1

        np.testing.assert_array_almost_equal(
            settings.drag_surfaces['normals'][:, 0],
            [0.0, 1.0, 0.0]
        )
        assert settings.drag_surfaces['CDs'][0] == 2.2

    def test_both_disturbances_extracted(self):
        """Test both SRP and drag surface data extraction."""
        sat = create_mtq_satellite_with_both()
        est_sat = EstimatedSatellite(
            mass=sat.mass, J_0=sat.J_0, actuators=sat.actuators,
            sensors=sat.sensors, disturbances=sat.disturbances, COM=sat.COM
        )

        settings = PlannerSettings(
            est_sat=est_sat,
            include_srp=True,
            include_drag=True
        )

        assert settings.srp_surfaces is not None
        assert settings.drag_surfaces is not None

    def test_no_extraction_when_disabled(self):
        """Test that surface data is not extracted when disturbances disabled."""
        sat = create_mtq_satellite_with_both()
        est_sat = EstimatedSatellite(
            mass=sat.mass, J_0=sat.J_0, actuators=sat.actuators,
            sensors=sat.sensors, disturbances=sat.disturbances, COM=sat.COM
        )

        settings = PlannerSettings(
            est_sat=est_sat,
            include_srp=False,
            include_drag=False
        )

        assert settings.srp_surfaces is None
        assert settings.drag_surfaces is None


# ==========================================
# TESTS: C++ SATELLITE CONSTRUCTION
# ==========================================

class TestCppSatelliteConstruction:
    """Test that C++ satellite is correctly constructed with surface data."""

    def test_cpp_satellite_with_srp(self):
        """Test C++ satellite construction with SRP surfaces."""
        sat = create_mtq_satellite_with_srp()
        est_sat = EstimatedSatellite(
            mass=sat.mass, J_0=sat.J_0, actuators=sat.actuators,
            sensors=sat.sensors, disturbances=sat.disturbances, COM=sat.COM
        )

        settings = PlannerSettings(
            est_sat=est_sat,
            include_srp=True,
            include_drag=False
        )

        # Build C++ satellite - should not raise
        csat = build_cpp_satellite(est_sat, settings)
        assert csat is not None

    def test_cpp_satellite_with_drag(self):
        """Test C++ satellite construction with drag surfaces."""
        sat = create_mtq_satellite_with_drag()
        est_sat = EstimatedSatellite(
            mass=sat.mass, J_0=sat.J_0, actuators=sat.actuators,
            sensors=sat.sensors, disturbances=sat.disturbances, COM=sat.COM
        )

        settings = PlannerSettings(
            est_sat=est_sat,
            include_srp=False,
            include_drag=True
        )

        csat = build_cpp_satellite(est_sat, settings)
        assert csat is not None

    def test_cpp_satellite_with_both(self):
        """Test C++ satellite construction with both disturbances."""
        sat = create_mtq_satellite_with_both()
        est_sat = EstimatedSatellite(
            mass=sat.mass, J_0=sat.J_0, actuators=sat.actuators,
            sensors=sat.sensors, disturbances=sat.disturbances, COM=sat.COM
        )

        settings = PlannerSettings(
            est_sat=est_sat,
            include_srp=True,
            include_drag=True
        )

        csat = build_cpp_satellite(est_sat, settings)
        assert csat is not None


# ==========================================
# TESTS: PHYSICS CALCULATIONS
# ==========================================

class TestDisturbancePhysics:
    """Test that the Python and expected physics match."""

    def test_srp_torque_direction(self):
        """Test SRP torque direction for simple geometry."""
        # Single face with +Y normal at +X position, sun along +Y
        # Expected torque should be along -Z (negative due to SRP formula)
        face = GeometryFace(
            area=1.0,
            centroid=np.array([1.0, 0.0, 0.0]),
            normal=np.array([0.0, 1.0, 0.0]),
            eta_a=1.0, eta_d=0.0, eta_s=0.0
        )
        config = GeometryConfig(geometry_faces=[face])
        srp = SRP_Disturbance(config=config)
        sat = Satellite(disturbances=[srp], COM=np.array([0.0, 0.0, 0.0]))

        # Identity quaternion in scalar-first Hamilton convention: [q0, q1, q2, q3] = [1, 0, 0, 0]
        # State x = [wx, wy, wz, q0, q1, q2, q3]
        x = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        ephem = Ephemeris()
        os = Orbital_State(
            ephem=ephem, J2000=0.22,
            R=np.array([7000.0, 0.0, 0.0]),
            V=np.array([0.0, 7.5, 0.0]),
            S=np.array([7000.0, 1e9, 0.0])  # Sun along +Y
        )

        torque = srp.torque(sat=sat, x=x, os=os)

        # Torque should be along -Z axis
        # r x s = [1,0,0] x [0,1,0] = [0,0,1]
        # With SRP formula: T = -P * A * (eta_a + eta_d) * (r x s)
        # So torque should be negative Z
        assert torque[2] < 0, f"SRP torque Z component should be negative, got {torque[2]}"
        assert abs(torque[0]) < 1e-15, f"SRP torque X should be ~0, got {torque[0]}"
        assert abs(torque[1]) < 1e-15, f"SRP torque Y should be ~0, got {torque[1]}"

    def test_drag_torque_direction(self):
        """Test drag torque direction for simple geometry."""
        # Single face with +Y normal at +X position, velocity along +Y
        face = GeometryFace(
            area=1.0,
            centroid=np.array([1.0, 0.0, 0.0]),
            normal=np.array([0.0, 1.0, 0.0]),
            CD=2.2
        )
        config = GeometryConfig(geometry_faces=[face])
        drag = Drag_Disturbance(config=config)
        sat = Satellite(disturbances=[drag], COM=np.array([0.0, 0.0, 0.0]))

        # Identity quaternion in scalar-first Hamilton convention
        x = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        ephem = Ephemeris()
        os = Orbital_State(
            ephem=ephem, J2000=0.22,
            R=np.array([7000.0, 0.0, 0.0]),
            V=np.array([0.0, 7.5, 0.0]),  # +Y velocity
            rho=1e-12
        )

        torque = drag.torque(sat=sat, x=x, os=os)

        # r x V = [1,0,0] x [0,V,0] = [0,0,V]
        # Drag torque = -0.5 * rho * CD * A * (n.V) * (r x V)
        # So torque should be negative Z
        assert torque[2] < 0, f"Drag torque Z component should be negative, got {torque[2]}"
        assert abs(torque[0]) < 1e-20, f"Drag torque X should be ~0, got {torque[0]}"
        assert abs(torque[1]) < 1e-20, f"Drag torque Y should be ~0, got {torque[1]}"

    def test_back_facing_no_torque(self):
        """Test that back-facing surfaces produce no torque."""
        # Face with -Y normal (away from sun/velocity)
        srp_face = GeometryFace(
            area=1.0,
            centroid=np.array([1.0, 0.0, 0.0]),
            normal=np.array([0.0, -1.0, 0.0]),  # Facing away
            eta_a=1.0, eta_d=0.0, eta_s=0.0
        )
        drag_face = GeometryFace(
            area=1.0,
            centroid=np.array([1.0, 0.0, 0.0]),
            normal=np.array([0.0, -1.0, 0.0]),
            CD=2.2
        )

        srp = SRP_Disturbance(GeometryConfig([srp_face]))
        drag = Drag_Disturbance(GeometryConfig([drag_face]))
        sat = Satellite(disturbances=[srp, drag], COM=np.array([0.0, 0.0, 0.0]))

        # Identity quaternion in scalar-first Hamilton convention: [q0, q1, q2, q3] = [1, 0, 0, 0]
        # State x = [wx, wy, wz, q0, q1, q2, q3]
        x = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        ephem = Ephemeris()
        os = Orbital_State(
            ephem=ephem, J2000=0.22,
            R=np.array([7000.0, 0.0, 0.0]),
            V=np.array([0.0, 7.5, 0.0]),
            S=np.array([7000.0, 1e9, 0.0]),
            rho=1e-12
        )

        srp_torque = srp.torque(sat=sat, x=x, os=os)
        drag_torque = drag.torque(sat=sat, x=x, os=os)

        np.testing.assert_array_almost_equal(srp_torque, [0, 0, 0])
        np.testing.assert_array_almost_equal(drag_torque, [0, 0, 0])


# ==========================================
# TESTS: PLANNER INTEGRATION
# ==========================================

class TestPlannerIntegration:
    """Test full planner integration with disturbances."""

    @pytest.mark.slow
    def test_planner_runs_with_srp(self):
        """Test that planner runs successfully with SRP enabled."""
        sat = create_mtq_satellite_with_srp()
        est_sat = EstimatedSatellite(
            mass=sat.mass, J_0=sat.J_0, actuators=sat.actuators,
            sensors=sat.sensors, disturbances=sat.disturbances, COM=sat.COM
        )
        orbit, os0 = create_orbit_with_density(200)

        # Initial state: identity quaternion (scalar-first Hamilton convention)
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        w0 = np.array([0.01, 0.0, 0.0])
        x0 = np.concatenate([w0, q0])

        # Goal - point along +X
        eci_goal = ECI_Goal(eci_vector=np.array([1, 0, 0]))
        goals = GoalList({os0.J2000: eci_goal})

        settings = PlannerSettings(
            est_sat=est_sat,
            include_srp=True,
            include_drag=False,
            dt_tp=10.0,
            dt_tvlqr=1.0,
        )

        controller = Plan_and_Track_LQR(
            est_sat=est_sat,
            planner_settings=settings,
        )

        # Calculate trajectory - should not raise
        traj = controller.calculate_trajectory(
            t_start=os0.J2000,
            duration=60.0,
            x_0=x0,
            os_0=os0,
            goals=goals,
            verbose=False
        )

        assert traj is not None
        assert traj.states is not None
        assert traj.controls is not None

    @pytest.mark.slow
    def test_planner_runs_with_drag(self):
        """Test that planner runs successfully with drag enabled."""
        sat = create_mtq_satellite_with_drag()
        est_sat = EstimatedSatellite(
            mass=sat.mass, J_0=sat.J_0, actuators=sat.actuators,
            sensors=sat.sensors, disturbances=sat.disturbances, COM=sat.COM
        )
        orbit, os0 = create_orbit_with_density(200)

        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        w0 = np.array([0.01, 0.0, 0.0])
        x0 = np.concatenate([w0, q0])

        eci_goal = ECI_Goal(eci_vector=np.array([1, 0, 0]))
        goals = GoalList({os0.J2000: eci_goal})

        settings = PlannerSettings(
            est_sat=est_sat,
            include_srp=False,
            include_drag=True,
            dt_tp=10.0,
            dt_tvlqr=1.0,
        )

        controller = Plan_and_Track_LQR(
            est_sat=est_sat,
            planner_settings=settings,
        )

        traj = controller.calculate_trajectory(
            t_start=os0.J2000,
            duration=60.0,
            x_0=x0,
            os_0=os0,
            goals=goals,
            verbose=False
        )

        assert traj is not None
        assert traj.states is not None

    @pytest.mark.slow
    def test_planner_runs_with_both_disturbances(self):
        """Test that planner runs with both SRP and drag enabled."""
        sat = create_mtq_satellite_with_both()
        est_sat = EstimatedSatellite(
            mass=sat.mass, J_0=sat.J_0, actuators=sat.actuators,
            sensors=sat.sensors, disturbances=sat.disturbances, COM=sat.COM
        )
        orbit, os0 = create_orbit_with_density(200)

        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        w0 = np.array([0.01, 0.0, 0.0])
        x0 = np.concatenate([w0, q0])

        eci_goal = ECI_Goal(eci_vector=np.array([1, 0, 0]))
        goals = GoalList({os0.J2000: eci_goal})

        settings = PlannerSettings(
            est_sat=est_sat,
            include_srp=True,
            include_drag=True,
            dt_tp=10.0,
            dt_tvlqr=1.0,
        )

        controller = Plan_and_Track_LQR(
            est_sat=est_sat,
            planner_settings=settings,
        )

        traj = controller.calculate_trajectory(
            t_start=os0.J2000,
            duration=60.0,
            x_0=x0,
            os_0=os0,
            goals=goals,
            verbose=False
        )

        assert traj is not None
        assert traj.states is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
