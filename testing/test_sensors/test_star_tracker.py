"""
Tests for star tracker sensor implementation.

Tests cover:
1. Star catalog initialization and coordinate conversion
2. Earth occlusion (stars behind Earth are not visible)
3. Sun exclusion (tracker blinded when sun near boresight)
4. Clean measurement model (b = A(q)^T @ s_ECI)
5. Jacobian correctness (finite difference verification)
6. Factory functions

References:
    [1] Vallado (2013), Section 5.3 for occlusion geometry
    [2] Markley & Crassidis (2014), Eq. 5.108 for measurement model
"""
import sys
import os
import numpy as np
import numdifftools as nd
import pytest

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.satellite_hardware.sensors import StarTracker, StarCatalog, NavigationStar
from ADCS.satellite_factory.sensors import create_bct_nst, create_terma_t1, create_generic_star_tracker
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_helpers import random_n_unit_vec, rot_mat


class TestStarCatalog:
    """Tests for the StarCatalog class."""

    def test_catalog_initialization(self):
        """Verify catalog loads approximately 30 stars."""
        catalog = StarCatalog()
        assert len(catalog.stars) >= 25
        assert len(catalog.stars) <= 35

    def test_star_vectors_are_unit(self):
        """Verify all star ECI vectors are unit vectors."""
        catalog = StarCatalog()
        for star in catalog.stars:
            norm = np.linalg.norm(star.s_eci)
            assert abs(norm - 1.0) < 1e-10, f"Star {star.name} has non-unit vector"

    def test_star_coordinates_valid(self):
        """Verify RA and Dec are in valid ranges."""
        catalog = StarCatalog()
        for star in catalog.stars:
            # RA should be 0 to 2*pi
            assert 0 <= star.ra_rad <= 2 * np.pi, f"{star.name} RA out of range"
            # Dec should be -pi/2 to pi/2
            assert -np.pi/2 <= star.dec_rad <= np.pi/2, f"{star.name} Dec out of range"

    def test_ra_dec_to_eci_conversion(self):
        """Verify RA/Dec to ECI conversion is correct."""
        catalog = StarCatalog()
        for star in catalog.stars:
            # Reconstruct ECI from RA/Dec
            expected = np.array([
                np.cos(star.dec_rad) * np.cos(star.ra_rad),
                np.cos(star.dec_rad) * np.sin(star.ra_rad),
                np.sin(star.dec_rad)
            ])
            np.testing.assert_allclose(star.s_eci, expected, rtol=1e-10)

    def test_brightest_stars_first(self):
        """Verify catalog is sorted by magnitude (brightest first)."""
        catalog = StarCatalog()
        # First star should be Sirius (brightest)
        assert catalog.stars[0].name == "Sirius"
        assert catalog.stars[0].vmag < -1.0

    def test_earth_occlusion(self):
        """Verify stars behind Earth are not visible.

        At 400 km altitude, Earth subtends about 141° of the sky.
        Stars within ~70.5° of nadir should be occluded.

        Reference: Vallado (2013), Section 5.3
        """
        catalog = StarCatalog()

        # Satellite at 400 km altitude on +X axis
        r_sat = np.array([6778.0, 0.0, 0.0])  # km

        # Point boresight toward nadir
        boresight_nadir = -r_sat / np.linalg.norm(r_sat)

        # Use very large FOV to not limit by FOV
        visible = catalog.get_visible_stars(
            boresight_eci=boresight_nadir,
            fov_rad=np.pi,  # 180 degrees
            r_sat_eci=r_sat
        )

        # Earth angular radius at 400 km
        R_EARTH = 6378.137
        r_norm = np.linalg.norm(r_sat)
        earth_angular_radius = np.arcsin(R_EARTH / r_norm)

        # All visible stars should be outside Earth disk
        nadir = -r_sat / r_norm
        for star in visible:
            angle_from_nadir = np.arccos(np.clip(np.dot(nadir, star.s_eci), -1, 1))
            assert angle_from_nadir > earth_angular_radius * 0.99, \
                f"Star {star.name} should be occluded by Earth"

    def test_sun_exclusion_blinds_tracker(self):
        """Verify tracker returns no stars when sun is in exclusion zone."""
        catalog = StarCatalog()
        r_sat = np.array([6778.0, 0.0, 0.0])  # 400 km altitude
        boresight = np.array([0.0, 1.0, 0.0])  # Looking +Y

        # Sun directly along boresight
        sun_eci = np.array([0.0, 1.5e8, 0.0])  # ~1 AU in +Y direction

        visible = catalog.get_visible_stars(
            boresight_eci=boresight,
            fov_rad=np.deg2rad(180),  # Very large FOV
            r_sat_eci=r_sat,
            sun_eci=sun_eci,
            sun_exclusion_rad=np.deg2rad(45)
        )

        # Tracker should be completely blinded
        assert len(visible) == 0, "Tracker should be blinded by sun"

    def test_sun_outside_exclusion_allows_stars(self):
        """Verify stars are visible when sun is outside exclusion zone."""
        catalog = StarCatalog()
        r_sat = np.array([6778.0, 0.0, 0.0])  # 400 km altitude
        boresight = np.array([0.0, 1.0, 0.0])  # Looking +Y

        # Sun perpendicular to boresight (90 degrees away)
        sun_eci = np.array([0.0, 0.0, 1.5e8])  # +Z direction

        visible = catalog.get_visible_stars(
            boresight_eci=boresight,
            fov_rad=np.deg2rad(180),
            r_sat_eci=r_sat,
            sun_eci=sun_eci,
            sun_exclusion_rad=np.deg2rad(45)
        )

        # Should have some visible stars (sun is 90° away, > 45° exclusion)
        assert len(visible) > 0, "Stars should be visible when sun is 90° from boresight"

    def test_fov_limits_visibility(self):
        """Verify FOV correctly limits visible stars."""
        catalog = StarCatalog()
        r_sat = np.array([6778.0, 0.0, 0.0])

        # Point at Sirius
        sirius = catalog.stars[0]
        boresight = sirius.s_eci.copy()

        # Very narrow FOV
        narrow_fov = np.deg2rad(1.0)
        visible_narrow = catalog.get_visible_stars(
            boresight_eci=boresight,
            fov_rad=narrow_fov,
            r_sat_eci=r_sat
        )

        # Wider FOV
        wide_fov = np.deg2rad(30.0)
        visible_wide = catalog.get_visible_stars(
            boresight_eci=boresight,
            fov_rad=wide_fov,
            r_sat_eci=r_sat
        )

        # Wider FOV should include at least as many stars
        assert len(visible_wide) >= len(visible_narrow)


class TestStarTracker:
    """Tests for the StarTracker sensor class."""

    @pytest.fixture
    def wide_fov_tracker(self):
        """Create a star tracker with very wide FOV for testing."""
        return StarTracker(
            boresight=np.array([0.0, 0.0, 1.0]),
            fov=np.deg2rad(170),  # Very wide to ensure star visibility
            cross_noise_std=1e-8,  # Effectively noiseless for clean tests
            roll_noise_std=1e-8,
            sun_exclusion=np.deg2rad(25)
        )

    @pytest.fixture
    def orbital_state(self):
        """Create a test orbital state."""
        ephem = Ephemeris()
        # Position satellite so boresight (+Z) points away from Earth/Sun
        os = Orbital_State(
            ephem=ephem,
            J2000=0.22,
            R=np.array([7000.0, 0.0, 0.0]),  # On +X axis
            V=np.array([0.0, 7.5, 0.0])      # Circular orbit velocity
        )
        return os

    def test_output_is_unit_vector(self, wide_fov_tracker, orbital_state):
        """Verify clean reading returns a unit vector."""
        # Use identity quaternion
        x = np.zeros(7)
        x[3:7] = [1.0, 0.0, 0.0, 0.0]

        b = wide_fov_tracker.clean_reading(x, orbital_state)

        if not np.any(np.isnan(b)):
            norm = np.linalg.norm(b)
            assert abs(norm - 1.0) < 1e-10, f"Output not unit vector: norm={norm}"

    def test_measurement_matches_dcm_transform(self, wide_fov_tracker, orbital_state):
        """Verify b = A(q)^T @ s_ECI.

        Reference: Markley & Crassidis (2014), Eq. 5.108
        """
        # Random quaternion
        q = random_n_unit_vec(4)
        x = np.concatenate([np.zeros(3), q])

        b = wide_fov_tracker.clean_reading(x, orbital_state)

        if not np.any(np.isnan(b)) and wide_fov_tracker.current_star is not None:
            # Manually compute expected result
            A = rot_mat(q)
            s_eci = wide_fov_tracker.current_star.s_eci
            expected = A.T @ s_eci

            np.testing.assert_allclose(b, expected, rtol=1e-10)

    def test_no_star_returns_nan(self):
        """Verify NaN returned when no star is visible."""
        # Very narrow FOV pointing away from all stars
        tracker = StarTracker(
            boresight=np.array([1.0, 0.0, 0.0]),
            fov=np.deg2rad(0.001),  # Extremely narrow
            cross_noise_std=1e-8,
            roll_noise_std=1e-8
        )

        ephem = Ephemeris()
        os = Orbital_State(
            ephem=ephem,
            J2000=0.22,
            R=np.array([7000.0, 0.0, 0.0]),
            V=np.array([0.0, 7.5, 0.0])
        )

        x = np.zeros(7)
        x[3:7] = [1.0, 0.0, 0.0, 0.0]

        b = tracker.clean_reading(x, os)

        # Should return NaN when no star visible
        assert np.all(np.isnan(b)), "Should return NaN when no star visible"

    def test_jacobian_finite_difference(self, wide_fov_tracker, orbital_state):
        """Verify analytical Jacobian matches finite difference.

        Reference: Shuster (1993), Eq. 168
        """
        # Use a specific quaternion
        q = np.array([0.9, 0.2, 0.3, 0.1])
        q = q / np.linalg.norm(q)
        x = np.concatenate([np.zeros(3), q])

        # Get clean reading to set current_star
        b0 = wide_fov_tracker.clean_reading(x, orbital_state)

        if np.any(np.isnan(b0)):
            pytest.skip("No star visible for Jacobian test")

        # Analytical Jacobian
        J_ana = wide_fov_tracker.basestate_jac(x, orbital_state)

        # Save reference to current star
        current_star = wide_fov_tracker.current_star

        # Finite difference Jacobian using numdifftools
        # Note: We do NOT renormalize the quaternion in the finite difference
        # because the analytical Jacobian drotmatTvecdq computes the derivative
        # without the unit quaternion constraint.
        def measurement_func(state):
            q_test = state[3:7]  # Do not renormalize for unconstrained derivative
            A = rot_mat(q_test)
            return A.T @ current_star.s_eci

        J_fd = nd.Jacobian(measurement_func)(x)

        # Compare quaternion part (rows 3-6)
        np.testing.assert_allclose(
            J_ana[3:7, :],
            J_fd[:, 3:7].T,
            rtol=1e-4,
            atol=1e-8
        )

    def test_omega_jacobian_is_zero(self, wide_fov_tracker, orbital_state):
        """Verify Jacobian w.r.t. angular velocity is zero.

        Measurement only depends on quaternion, not angular velocity.
        """
        x = np.zeros(7)
        x[3:7] = [1.0, 0.0, 0.0, 0.0]

        wide_fov_tracker.clean_reading(x, orbital_state)
        J = wide_fov_tracker.basestate_jac(x, orbital_state)

        # First 3 rows (omega) should be zero
        np.testing.assert_allclose(J[0:3, :], np.zeros((3, 3)), atol=1e-15)

    def test_bias_jacobian_is_empty(self, wide_fov_tracker, orbital_state):
        """Verify bias Jacobian has correct shape (no bias states)."""
        x = np.zeros(7)
        x[3:7] = [1.0, 0.0, 0.0, 0.0]

        J_bias = wide_fov_tracker.bias_jac(x, orbital_state)

        assert J_bias.shape == (0, 3)

    def test_noise_covariance_shape(self, wide_fov_tracker):
        """Verify noise covariance has correct shape."""
        R = wide_fov_tracker.noise_covariance
        assert R.shape == (3, 3)

    def test_noise_covariance_positive_definite(self, wide_fov_tracker):
        """Verify noise covariance is positive definite."""
        R = wide_fov_tracker.noise_covariance
        eigenvalues = np.linalg.eigvalsh(R)
        assert np.all(eigenvalues > 0), "Noise covariance should be positive definite"


class TestStarTrackerFactory:
    """Tests for star tracker factory functions."""

    def test_create_bct_nst(self):
        """Test BCT NST factory function."""
        tracker = create_bct_nst()

        assert tracker.fov == pytest.approx(np.deg2rad(10.0))
        assert tracker.sun_exclusion == pytest.approx(np.deg2rad(45.0))
        # 6 arcsec cross noise
        assert tracker.cross_noise_std == pytest.approx(6.0 * np.pi / (180 * 3600))
        # 40 arcsec roll noise
        assert tracker.roll_noise_std == pytest.approx(40.0 * np.pi / (180 * 3600))

    def test_create_terma_t1(self):
        """Test Terma T1 factory function."""
        tracker = create_terma_t1()

        assert tracker.fov == pytest.approx(np.deg2rad(22.0))
        assert tracker.sun_exclusion == pytest.approx(np.deg2rad(30.0))
        # 2 arcsec cross noise
        assert tracker.cross_noise_std == pytest.approx(2.0 * np.pi / (180 * 3600))
        # 15 arcsec roll noise
        assert tracker.roll_noise_std == pytest.approx(15.0 * np.pi / (180 * 3600))

    def test_create_generic_star_tracker(self):
        """Test generic star tracker factory function."""
        tracker = create_generic_star_tracker(
            cross_arcsec=8.0,
            roll_arcsec=30.0,
            fov_deg=12.0,
            sun_exclusion_deg=40.0
        )

        assert tracker.fov == pytest.approx(np.deg2rad(12.0))
        assert tracker.sun_exclusion == pytest.approx(np.deg2rad(40.0))
        assert tracker.cross_noise_std == pytest.approx(8.0 * np.pi / (180 * 3600))
        assert tracker.roll_noise_std == pytest.approx(30.0 * np.pi / (180 * 3600))

    def test_custom_boresight(self):
        """Test factory function with custom boresight."""
        boresight = np.array([0.0, 1.0, 0.0])
        tracker = create_bct_nst(boresight=boresight)

        np.testing.assert_allclose(
            tracker.boresight,
            boresight / np.linalg.norm(boresight),
            rtol=1e-10
        )


class TestAnisotropicNoise:
    """Tests for the anisotropic noise model."""

    def test_noise_aligned_with_boresight_z(self):
        """Verify noise model when boresight is +Z axis."""
        tracker = StarTracker(
            boresight=np.array([0.0, 0.0, 1.0]),
            fov=np.deg2rad(20),
            cross_noise_std=1e-4,  # 0.1 mrad
            roll_noise_std=5e-4   # 0.5 mrad
        )

        R = tracker.noise_covariance

        # For boresight along z, noise should be diagonal
        # with cross noise on x,y and roll noise on z
        expected = np.diag([1e-4**2, 1e-4**2, 5e-4**2])
        np.testing.assert_allclose(R, expected, rtol=1e-10)

    def test_noise_rotation_for_arbitrary_boresight(self):
        """Verify noise covariance is properly rotated for arbitrary boresight."""
        tracker = StarTracker(
            boresight=np.array([1.0, 0.0, 0.0]),  # +X boresight
            fov=np.deg2rad(20),
            cross_noise_std=1e-4,
            roll_noise_std=5e-4
        )

        R = tracker.noise_covariance

        # Should be positive definite
        eigvals = np.linalg.eigvalsh(R)
        assert np.all(eigvals > 0)

        # Eigenvalues should be the noise variances
        eigvals_sorted = np.sort(eigvals)
        expected_eigvals = np.sort([1e-4**2, 1e-4**2, 5e-4**2])
        np.testing.assert_allclose(eigvals_sorted, expected_eigvals, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
