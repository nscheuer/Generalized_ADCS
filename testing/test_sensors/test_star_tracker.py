import sys
import os
import numpy as np
import numdifftools as nd
import pytest

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.satellite_hardware.sensors import StarTracker
from ADCS.environment import StarCatalog, NavigationStar
from ADCS.satellite_factory.sensors import create_bct_nst, create_terma_t1, create_generic_star_tracker
from ADCS.satellite_hardware.actuators import AnisotropicNoise
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_helpers import random_n_unit_vec, rot_mat

def test_catalog_initialization():
    catalog = StarCatalog()
    assert len(catalog.stars) >= 25
    assert len(catalog.stars) <= 35

def test_star_vectors_are_unit():
    catalog = StarCatalog()
    for star in catalog.stars:
        norm = np.linalg.norm(star.s_eci)
        assert abs(norm - 1.0) < 1e-10

def test_star_coordinates_valid():
    catalog = StarCatalog()
    for star in catalog.stars:
        assert 0 <= star.ra_rad <= 2 * np.pi
        assert -np.pi/2 <= star.dec_rad <= np.pi/2

def test_ra_dec_to_eci_conversion():
    catalog = StarCatalog()
    for star in catalog.stars:
        expected = np.array([
            np.cos(star.dec_rad) * np.cos(star.ra_rad),
            np.cos(star.dec_rad) * np.sin(star.ra_rad),
            np.sin(star.dec_rad)
        ])
        np.testing.assert_allclose(star.s_eci, expected, rtol=1e-10)

def test_brightest_stars_first():
    catalog = StarCatalog()
    assert catalog.stars[0].name == "Sirius"
    assert catalog.stars[0].vmag < -1.0

def test_earth_occlusion():
    catalog = StarCatalog()
    r_sat = np.array([6778.0, 0.0, 0.0])
    boresight_nadir = -r_sat / np.linalg.norm(r_sat)

    visible = catalog.get_visible_stars(
        boresight_eci=boresight_nadir,
        fov_rad=np.pi,
        r_sat_eci=r_sat
    )

    R_EARTH = 6378.137
    r_norm = np.linalg.norm(r_sat)
    earth_angular_radius = np.arcsin(R_EARTH / r_norm)
    nadir = -r_sat / r_norm

    for star in visible:
        angle_from_nadir = np.arccos(np.clip(np.dot(nadir, star.s_eci), -1, 1))
        assert angle_from_nadir > earth_angular_radius * 0.99

def test_sun_exclusion_blinds_tracker():
    catalog = StarCatalog()
    r_sat = np.array([6778.0, 0.0, 0.0])
    boresight = np.array([0.0, 1.0, 0.0])
    sun_eci = np.array([0.0, 1.5e8, 0.0])

    visible = catalog.get_visible_stars(
        boresight_eci=boresight,
        fov_rad=np.deg2rad(180),
        r_sat_eci=r_sat,
        sun_eci=sun_eci,
        sun_exclusion_rad=np.deg2rad(45)
    )
    assert len(visible) == 0

def test_sun_outside_exclusion_allows_stars():
    catalog = StarCatalog()
    r_sat = np.array([6778.0, 0.0, 0.0])
    boresight = np.array([0.0, 1.0, 0.0])
    sun_eci = np.array([0.0, 0.0, 1.5e8])

    visible = catalog.get_visible_stars(
        boresight_eci=boresight,
        fov_rad=np.deg2rad(180),
        r_sat_eci=r_sat,
        sun_eci=sun_eci,
        sun_exclusion_rad=np.deg2rad(45)
    )
    assert len(visible) > 0

def test_fov_limits_visibility():
    catalog = StarCatalog()
    r_sat = np.array([6778.0, 0.0, 0.0])
    sirius = catalog.stars[0]
    boresight = sirius.s_eci.copy()

    visible_narrow = catalog.get_visible_stars(
        boresight_eci=boresight,
        fov_rad=np.deg2rad(1.0),
        r_sat_eci=r_sat
    )
    visible_wide = catalog.get_visible_stars(
        boresight_eci=boresight,
        fov_rad=np.deg2rad(30.0),
        r_sat_eci=r_sat
    )
    assert len(visible_wide) >= len(visible_narrow)

def test_output_is_unit_vector():
    noise = AnisotropicNoise(std_cross=1e-8, std_roll=1e-8)
    tracker = StarTracker(
        boresight=np.array([0.0, 0.0, 1.0]),
        fov=np.deg2rad(170),
        anisotropic_noise=noise,
        sun_exclusion=np.deg2rad(25)
    )
    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]))
    
    x = np.zeros(7)
    x[3:7] = [1.0, 0.0, 0.0, 0.0]

    b = tracker.clean_reading(x, os)

    if not np.any(np.isnan(b)):
        norm = np.linalg.norm(b)
        assert abs(norm - 1.0) < 1e-10

def test_measurement_matches_dcm_transform():
    noise = AnisotropicNoise(std_cross=1e-8, std_roll=1e-8)
    tracker = StarTracker(
        boresight=np.array([0.0, 0.0, 1.0]),
        fov=np.deg2rad(170),
        anisotropic_noise=noise,
        sun_exclusion=np.deg2rad(25)
    )
    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]))

    q = random_n_unit_vec(4)
    x = np.concatenate([np.zeros(3), q])

    b = tracker.clean_reading(x, os)

    if not np.any(np.isnan(b)) and tracker.current_star is not None:
        A = rot_mat(q)
        s_eci = tracker.current_star.s_eci
        expected = A.T @ s_eci
        np.testing.assert_allclose(b, expected, rtol=1e-10)

def test_no_star_returns_nan():
    noise = AnisotropicNoise(std_cross=1e-8, std_roll=1e-8)
    tracker = StarTracker(
        boresight=np.array([1.0, 0.0, 0.0]),
        fov=np.deg2rad(0.001),
        anisotropic_noise=noise
    )
    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]))

    x = np.zeros(7)
    x[3:7] = [1.0, 0.0, 0.0, 0.0]

    b = tracker.clean_reading(x, os)
    assert np.all(np.isnan(b))

def test_jacobian_finite_difference():
    noise = AnisotropicNoise(std_cross=1e-8, std_roll=1e-8)
    tracker = StarTracker(
        boresight=np.array([0.0, 0.0, 1.0]),
        fov=np.deg2rad(170),
        anisotropic_noise=noise,
        sun_exclusion=np.deg2rad(25)
    )
    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]))

    q = np.array([0.9, 0.2, 0.3, 0.1])
    q = q / np.linalg.norm(q)
    x = np.concatenate([np.zeros(3), q])

    b0 = tracker.clean_reading(x, os)
    if np.any(np.isnan(b0)):
        return

    J_ana = tracker.basestate_jac(x, os)
    current_star = tracker.current_star

    def measurement_func(state):
        q_test = state[3:7]
        A = rot_mat(q_test)
        return A.T @ current_star.s_eci

    J_fd = nd.Jacobian(measurement_func)(x)

    np.testing.assert_allclose(J_ana[3:7, :], J_fd[:, 3:7].T, rtol=1e-4, atol=1e-8)

def test_omega_jacobian_is_zero():
    noise = AnisotropicNoise(std_cross=1e-8, std_roll=1e-8)
    tracker = StarTracker(
        boresight=np.array([0.0, 0.0, 1.0]),
        fov=np.deg2rad(170),
        anisotropic_noise=noise,
        sun_exclusion=np.deg2rad(25)
    )
    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]))

    x = np.zeros(7)
    x[3:7] = [1.0, 0.0, 0.0, 0.0]

    tracker.clean_reading(x, os)
    J = tracker.basestate_jac(x, os)
    np.testing.assert_allclose(J[0:3, :], np.zeros((3, 3)), atol=1e-15)

def test_bias_jacobian_is_empty():
    noise = AnisotropicNoise(std_cross=1e-8, std_roll=1e-8)
    tracker = StarTracker(
        boresight=np.array([0.0, 0.0, 1.0]),
        fov=np.deg2rad(170),
        anisotropic_noise=noise
    )
    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]))
    
    x = np.zeros(7)
    x[3:7] = [1.0, 0.0, 0.0, 0.0]
    J_bias = tracker.bias_jac(x, os)
    assert J_bias.shape == (0, 3)

def test_noise_covariance_shape():
    noise = AnisotropicNoise(std_cross=1e-8, std_roll=1e-8)
    tracker = StarTracker(
        boresight=np.array([0.0, 0.0, 1.0]),
        fov=np.deg2rad(170),
        anisotropic_noise=noise
    )
    assert tracker.noise_covariance.shape == (3, 3)

def test_noise_covariance_positive_definite():
    noise = AnisotropicNoise(std_cross=1e-8, std_roll=1e-8)
    tracker = StarTracker(
        boresight=np.array([0.0, 0.0, 1.0]),
        fov=np.deg2rad(170),
        anisotropic_noise=noise
    )
    eigenvalues = np.linalg.eigvalsh(tracker.noise_covariance)
    assert np.all(eigenvalues > 0)

def test_create_bct_nst():
    tracker = create_bct_nst()
    assert tracker.fov == pytest.approx(np.deg2rad(10.0))
    assert tracker.sun_exclusion == pytest.approx(np.deg2rad(45.0))
    assert tracker.noise.std_cross == pytest.approx(6.0 * np.pi / (180 * 3600))
    assert tracker.noise.std_roll == pytest.approx(40.0 * np.pi / (180 * 3600))

def test_create_terma_t1():
    tracker = create_terma_t1()
    assert tracker.fov == pytest.approx(np.deg2rad(22.0))
    assert tracker.sun_exclusion == pytest.approx(np.deg2rad(30.0))
    assert tracker.noise.std_cross == pytest.approx(2.0 * np.pi / (180 * 3600))
    assert tracker.noise.std_roll == pytest.approx(15.0 * np.pi / (180 * 3600))

def test_create_generic_star_tracker():
    tracker = create_generic_star_tracker(
        cross_arcsec=8.0,
        roll_arcsec=30.0,
        fov_deg=12.0,
        sun_exclusion_deg=40.0
    )
    assert tracker.fov == pytest.approx(np.deg2rad(12.0))
    assert tracker.sun_exclusion == pytest.approx(np.deg2rad(40.0))
    assert tracker.noise.std_cross == pytest.approx(8.0 * np.pi / (180 * 3600))
    assert tracker.noise.std_roll == pytest.approx(30.0 * np.pi / (180 * 3600))

def test_custom_boresight():
    boresight = np.array([0.0, 1.0, 0.0])
    tracker = create_bct_nst(boresight=boresight)
    np.testing.assert_allclose(
        tracker.boresight,
        boresight / np.linalg.norm(boresight),
        rtol=1e-10
    )

def test_noise_aligned_with_boresight_z():
    noise = AnisotropicNoise(std_cross=1e-4, std_roll=5e-4)
    tracker = StarTracker(
        boresight=np.array([0.0, 0.0, 1.0]),
        fov=np.deg2rad(20),
        anisotropic_noise=noise
    )
    R = tracker.noise_covariance
    expected = np.diag([1e-4**2, 1e-4**2, 5e-4**2])
    np.testing.assert_allclose(R, expected, rtol=1e-10)

def test_noise_rotation_for_arbitrary_boresight():
    noise = AnisotropicNoise(std_cross=1e-4, std_roll=5e-4)
    tracker = StarTracker(
        boresight=np.array([1.0, 0.0, 0.0]),
        fov=np.deg2rad(20),
        anisotropic_noise=noise
    )
    R = tracker.noise_covariance
    eigvals = np.linalg.eigvalsh(R)
    assert np.all(eigvals > 0)
    eigvals_sorted = np.sort(eigvals)
    expected_eigvals = np.sort([1e-4**2, 1e-4**2, 5e-4**2])
    np.testing.assert_allclose(eigvals_sorted, expected_eigvals, rtol=1e-10)

if __name__ == "__main__":
    test_catalog_initialization()
    test_star_vectors_are_unit()
    test_star_coordinates_valid()
    test_ra_dec_to_eci_conversion()
    test_brightest_stars_first()
    test_earth_occlusion()
    test_sun_exclusion_blinds_tracker()
    test_sun_outside_exclusion_allows_stars()
    test_fov_limits_visibility()
    
    test_output_is_unit_vector()
    test_measurement_matches_dcm_transform()
    test_no_star_returns_nan()
    test_jacobian_finite_difference()
    test_omega_jacobian_is_zero()
    test_bias_jacobian_is_empty()
    test_noise_covariance_shape()
    test_noise_covariance_positive_definite()
    
    test_create_bct_nst()
    test_create_terma_t1()
    test_create_generic_star_tracker()
    test_custom_boresight()
    
    test_noise_aligned_with_boresight_z()
    test_noise_rotation_for_arbitrary_boresight()
    print("All tests passed.")