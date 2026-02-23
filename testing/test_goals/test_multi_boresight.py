"""
Multi-boresight goal alignment tests.

Tests verify that goals can correctly align with different named boresights
on the same spacecraft, and that boresight switching produces expected
angular errors.
"""

import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.CONOPS.goals import ECI_Goal, Coordinate_Goal
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_helpers import normalize, rot_mat
from ADCS.helpers.math_constants import MathConstants


class TestMultiBoresightGoals:
    """Test suite for multi-boresight goal alignment."""

    @pytest.fixture
    def multi_boresight_sat(self):
        """Create a satellite with multiple named boresights."""
        return Satellite(
            mass=4.0,
            J_0=np.diagflat([0.1, 0.1, 0.1]),
            boresight={
                "camera": np.array([0.0, 0.0, 1.0]),      # +Z axis
                "solar_panel": np.array([1.0, 0.0, 0.0]), # +X axis
                "antenna": np.array([0.0, 1.0, 0.0]),     # +Y axis
            }
        )

    @pytest.fixture
    def orbital_state(self):
        """Create a standard orbital state for testing."""
        ephem = Ephemeris()
        R = 6778 * np.array([1.0, 0.0, 0.0])  # ~400km altitude, over equator
        V = np.array([0.0, 7.67, 0.0])        # ~circular orbit velocity
        B = np.array([0.0, 0.0, 1e-5])        # Small B-field
        return Orbital_State(
            ephem=ephem,
            J2000=0.22,
            R=R,
            V=V,
            B=B,
            S=np.array([1e5, 0.0, 0.0]),
            rho=0.0
        )

    def test_multi_boresight_satellite_creation(self, multi_boresight_sat):
        """Test that multi-boresight satellite is created correctly."""
        sat = multi_boresight_sat

        # Check all boresights are present
        assert isinstance(sat.boresight, dict)
        assert "camera" in sat.boresight
        assert "solar_panel" in sat.boresight
        assert "antenna" in sat.boresight

        # Check retrieval by name
        camera = sat.get_boresight("camera")
        assert np.allclose(camera, [0.0, 0.0, 1.0])

        solar_panel = sat.get_boresight("solar_panel")
        assert np.allclose(solar_panel, [1.0, 0.0, 0.0])

        antenna = sat.get_boresight("antenna")
        assert np.allclose(antenna, [0.0, 1.0, 0.0])

    def test_get_boresight_default(self, multi_boresight_sat):
        """Test get_boresight() returns first boresight when name is None."""
        sat = multi_boresight_sat
        default = sat.get_boresight(None)
        first = sat.get_boresight("camera")  # First one in dict
        assert np.allclose(default, first)

    def test_eci_goal_with_boresight_name(self, multi_boresight_sat, orbital_state):
        """Test ECI_Goal with specific boresight name."""
        sat = multi_boresight_sat
        os0 = orbital_state

        # Create eci goal pointing at [1, 0, 0]
        goal = ECI_Goal(np.array([1.0, 0.0, 0.0]), boresight_name="camera")

        # Quaternion pointing +Z axis (body) toward +X (ECI)
        # q = [0, 1/√2, 0, 1/√2] (90° rotation about Y axis)
        q_body2eci = normalize(np.array([0.0, 1.0 / np.sqrt(2), 0.0, 1.0 / np.sqrt(2)]))

        # Compute error
        error = goal.error(q=q_body2eci, body_boresight=sat.get_boresight("camera"), os0=os0)

        # Camera (+Z body) should be aligned with +X target (ECI)
        # Error should be close to zero in terms of quaternion distance
        assert np.all(np.isfinite(error)), "Error should be finite"

    def test_boresight_switching_produces_angular_error(
        self, multi_boresight_sat, orbital_state
    ):
        """Test that switching boresights changes the computed error."""
        sat = multi_boresight_sat
        os0 = orbital_state

        # Identity quaternion (body = ECI)
        q = np.array([1.0, 0.0, 0.0, 0.0])

        # Goal: point camera at +X ECI
        goal_camera = ECI_Goal(np.array([1.0, 0.0, 0.0]), boresight_name="camera")

        # Goal: point solar panel at +X ECI
        goal_panel = ECI_Goal(np.array([1.0, 0.0, 0.0]), boresight_name="solar_panel")

        # Compute errors
        error_camera = goal_camera.error(
            q=q, body_boresight=sat.get_boresight("camera"), os0=os0
        )
        error_panel = goal_panel.error(
            q=q, body_boresight=sat.get_boresight("solar_panel"), os0=os0
        )

        # Camera is +Z, solar_panel is +X
        # Both goals want to point at +X ECI with identity quaternion
        # camera error: try to point +Z at +X → ~90° error
        # panel error: try to point +X at +X → ~0° error
        # Errors should be significantly different
        error_diff = np.linalg.norm(error_camera - error_panel)
        assert error_diff > 0.1, "Errors should differ significantly when boresights differ"

    def test_coordinate_goal_with_boresight(self, multi_boresight_sat, orbital_state):
        """Test Coordinate_Goal respects boresight_name parameter."""
        sat = multi_boresight_sat
        os0 = orbital_state

        # Nadir target (ground point at satellite sub-point)
        goal = Coordinate_Goal(lat=0.0, lon=0.0, alt=0.0, boresight_name="antenna")

        assert goal.boresight_name == "antenna"

        # Get reference
        r_ref, w_ref = goal.to_ref(os0)

        # Should return valid reference (nadir is downward)
        assert len(r_ref) == 4
        assert np.isnan(r_ref[0])  # Vector format
        assert np.linalg.norm(r_ref[1:4]) > 0  # Should have valid direction

    def test_goal_default_boresight_name(self, orbital_state):
        """Test that goals without explicit boresight_name have None."""
        goal = ECI_Goal(np.array([1.0, 0.0, 0.0]))
        assert goal.boresight_name is None

    def test_eci_goal_alignment_perfect(self, multi_boresight_sat, orbital_state):
        """Test perfect alignment between boresight and target."""
        sat = multi_boresight_sat
        os0 = orbital_state

        # Goal: point camera (+Z body) at +Z ECI
        goal = ECI_Goal(np.array([0.0, 0.0, 1.0]), boresight_name="camera")

        # Identity quaternion (body = ECI)
        q_identity = np.array([1.0, 0.0, 0.0, 0.0])

        error = goal.error(
            q=q_identity,
            body_boresight=sat.get_boresight("camera"),
            os0=os0
        )

        # With perfect alignment, error should be near zero (quaternion [1,0,0,0])
        # Small numerical tolerance
        assert np.linalg.norm(error[1:]) < 1e-10, "Error vector should be near zero"

    def test_multiple_boresights_independent_errors(
        self, multi_boresight_sat, orbital_state
    ):
        """Test that different goals can use different boresights independently."""
        sat = multi_boresight_sat
        os0 = orbital_state

        # Create three goals, each using different boresight
        goal_camera = ECI_Goal(
            np.array([0.0, 0.0, 1.0]), boresight_name="camera"
        )
        goal_panel = ECI_Goal(
            np.array([1.0, 0.0, 0.0]), boresight_name="solar_panel"
        )
        goal_antenna = ECI_Goal(
            np.array([0.0, 1.0, 0.0]), boresight_name="antenna"
        )

        # Same random quaternion
        q = normalize(np.array([0.5, 0.5, 0.5, 0.5]))

        error_camera = goal_camera.error(
            q=q, body_boresight=sat.get_boresight("camera"), os0=os0
        )
        error_panel = goal_panel.error(
            q=q, body_boresight=sat.get_boresight("solar_panel"), os0=os0
        )
        error_antenna = goal_antenna.error(
            q=q, body_boresight=sat.get_boresight("antenna"), os0=os0
        )

        # All three errors should be different
        assert not np.allclose(error_camera, error_panel)
        assert not np.allclose(error_panel, error_antenna)
        assert not np.allclose(error_camera, error_antenna)

    def test_coordinate_goal_nadir_pointing(self, multi_boresight_sat, orbital_state):
        """Test coordinate goal for nadir (downward) pointing."""
        sat = multi_boresight_sat
        os0 = orbital_state

        # Nadir: ground point at satellite sub-point
        goal = Coordinate_Goal(lat=0.0, lon=0.0, alt=0.0, boresight_name="solar_panel")

        r_ref, w_ref = goal.to_ref(os0)

        # Reference should be a vector (not quaternion)
        assert np.isnan(r_ref[0])

        # Direction should be roughly downward (negative radial from satellite)
        # Satellite is at +X, Earth center at origin
        # Nadir should point from satellite toward Earth center
        direction = r_ref[1:4]
        assert np.linalg.norm(direction) > 0

        # Should not be upward (positive radial)
        dot_product = np.dot(direction, np.array([1.0, 0.0, 0.0]))
        assert dot_product < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
