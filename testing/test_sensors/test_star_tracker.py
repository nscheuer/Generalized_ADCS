import numpy as np
import pytest

from ADCS.environment import StarCatalog
from ADCS.helpers.math_helpers import random_n_unit_vec, rot_mat
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_factory.sensors import create_bct_nst, create_generic_star_tracker, create_terma_t1
from ADCS.satellite_hardware.errors import AnisotropicNoise
from ADCS.satellite_hardware.sensors import StarTracker


def make_tracker(
    *,
    boresight: np.ndarray | None = None,
    fov_deg: float = 170.0,
    std_cross: float = 1e-8,
    std_roll: float = 1e-8,
    sun_exclusion_deg: float = 25.0,
):
    return StarTracker(
        boresight=np.array([0.0, 0.0, 1.0]) if boresight is None else boresight,
        fov=np.deg2rad(fov_deg),
        anisotropic_noise=AnisotropicNoise(std_cross=std_cross, std_roll=std_roll),
        sun_exclusion=np.deg2rad(sun_exclusion_deg),
    )


def make_orbital_state() -> Orbital_State:
    return Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]),
    )


def central_difference_jacobian(tracker: StarTracker, state: np.ndarray, orbital_state: Orbital_State, star_vector: np.ndarray, eps: float = 1e-7):
    def measurement(candidate):
        quaternion = candidate[3:7]
        return rot_mat(quaternion).T @ star_vector

    numeric = np.zeros((state.size, 3))
    for index in range(state.size):
        delta = np.zeros(state.size)
        delta[index] = eps
        numeric[index] = (measurement(state + delta) - measurement(state - delta)) / (2.0 * eps)
    return numeric


def test_star_catalog_has_expected_size_range():
    catalog = StarCatalog()
    assert 25 <= len(catalog.stars) <= 35


def test_star_catalog_vectors_are_unit_length():
    catalog = StarCatalog()
    for star in catalog.stars:
        assert abs(np.linalg.norm(star.s_eci) - 1.0) < 1e-10


def test_star_catalog_coordinates_are_in_valid_ranges():
    catalog = StarCatalog()
    for star in catalog.stars:
        assert 0.0 <= star.ra_rad <= 2.0 * np.pi
        assert -np.pi / 2.0 <= star.dec_rad <= np.pi / 2.0


def test_star_catalog_ra_dec_conversion_matches_eci_vectors():
    catalog = StarCatalog()
    for star in catalog.stars:
        expected = np.array(
            [
                np.cos(star.dec_rad) * np.cos(star.ra_rad),
                np.cos(star.dec_rad) * np.sin(star.ra_rad),
                np.sin(star.dec_rad),
            ]
        )
        np.testing.assert_allclose(star.s_eci, expected, rtol=1e-10)


def test_star_catalog_is_sorted_brightest_first():
    catalog = StarCatalog()
    assert catalog.stars[0].name == "Sirius"
    assert catalog.stars[0].vmag < -1.0


def test_visible_stars_exclude_earth_occlusion():
    catalog = StarCatalog()
    satellite_position = np.array([6778.0, 0.0, 0.0])
    boresight = -satellite_position / np.linalg.norm(satellite_position)
    visible = catalog.get_visible_stars(boresight_eci=boresight, fov_rad=np.pi, r_sat_eci=satellite_position)

    earth_radius = 6378.137
    earth_angular_radius = np.arcsin(earth_radius / np.linalg.norm(satellite_position))
    nadir = -satellite_position / np.linalg.norm(satellite_position)
    for star in visible:
        angle_from_nadir = np.arccos(np.clip(np.dot(nadir, star.s_eci), -1.0, 1.0))
        assert angle_from_nadir > earth_angular_radius * 0.99


def test_visible_stars_are_suppressed_by_sun_exclusion():
    catalog = StarCatalog()
    visible = catalog.get_visible_stars(
        boresight_eci=np.array([0.0, 1.0, 0.0]),
        fov_rad=np.deg2rad(180.0),
        r_sat_eci=np.array([6778.0, 0.0, 0.0]),
        sun_eci=np.array([0.0, 1.5e8, 0.0]),
        sun_exclusion_rad=np.deg2rad(45.0),
    )
    assert len(visible) == 0


def test_visible_stars_remain_when_sun_is_outside_exclusion_zone():
    catalog = StarCatalog()
    visible = catalog.get_visible_stars(
        boresight_eci=np.array([0.0, 1.0, 0.0]),
        fov_rad=np.deg2rad(180.0),
        r_sat_eci=np.array([6778.0, 0.0, 0.0]),
        sun_eci=np.array([0.0, 0.0, 1.5e8]),
        sun_exclusion_rad=np.deg2rad(45.0),
    )
    assert len(visible) > 0


def test_visible_star_count_grows_with_wider_fov():
    catalog = StarCatalog()
    boresight = catalog.stars[0].s_eci.copy()
    narrow = catalog.get_visible_stars(boresight_eci=boresight, fov_rad=np.deg2rad(1.0), r_sat_eci=np.array([6778.0, 0.0, 0.0]))
    wide = catalog.get_visible_stars(boresight_eci=boresight, fov_rad=np.deg2rad(30.0), r_sat_eci=np.array([6778.0, 0.0, 0.0]))
    assert len(wide) >= len(narrow)


def test_star_tracker_clean_reading_is_unit_vector_when_visible():
    tracker = make_tracker()
    reading = tracker.clean_reading(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), make_orbital_state())
    if not np.any(np.isnan(reading)):
        assert abs(np.linalg.norm(reading) - 1.0) < 1e-10


def test_star_tracker_clean_reading_matches_attitude_transform():
    tracker = make_tracker()
    orbital_state = make_orbital_state()
    quaternion = random_n_unit_vec(4)
    state = np.concatenate([np.zeros(3), quaternion])
    reading = tracker.clean_reading(state, orbital_state)
    if not np.any(np.isnan(reading)) and tracker.current_star is not None:
        expected = rot_mat(quaternion).T @ tracker.current_star.s_eci
        np.testing.assert_allclose(reading, expected, rtol=1e-10)


def test_star_tracker_returns_nan_when_no_star_is_visible():
    tracker = make_tracker(boresight=np.array([1.0, 0.0, 0.0]), fov_deg=0.001, sun_exclusion_deg=25.0)
    reading = tracker.clean_reading(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), make_orbital_state())
    assert np.all(np.isnan(reading))


def test_star_tracker_jacobian_matches_finite_difference():
    tracker = make_tracker()
    orbital_state = make_orbital_state()
    quaternion = np.array([0.9, 0.2, 0.3, 0.1])
    quaternion = quaternion / np.linalg.norm(quaternion)
    state = np.concatenate([np.zeros(3), quaternion])
    reading = tracker.clean_reading(state, orbital_state)
    if not np.any(np.isnan(reading)) and tracker.current_star is not None:
        analytic = tracker.basestate_jac(state, orbital_state)
        numeric = central_difference_jacobian(tracker, state, orbital_state, tracker.current_star.s_eci)
        np.testing.assert_allclose(analytic[3:7, :], numeric[3:7, :], rtol=1e-4, atol=1e-8)


def test_star_tracker_jacobian_has_zero_omega_block():
    tracker = make_tracker()
    jacobian = tracker.basestate_jac(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), make_orbital_state())
    np.testing.assert_allclose(jacobian[0:3, :], np.zeros((3, 3)), atol=1e-15)


def test_star_tracker_bias_jacobian_is_empty():
    jacobian = make_tracker().bias_jac(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), make_orbital_state())
    assert jacobian.shape == (0, 3)


def test_star_tracker_noise_covariance_has_expected_shape():
    assert make_tracker().noise_covariance.shape == (3, 3)


def test_star_tracker_noise_covariance_is_positive_definite():
    eigenvalues = np.linalg.eigvalsh(make_tracker().noise_covariance)
    assert np.all(eigenvalues > 0)


def test_bct_factory_sets_expected_tracker_properties():
    tracker = create_bct_nst()
    assert tracker.fov == pytest.approx(np.deg2rad(10.0))
    assert tracker.sun_exclusion == pytest.approx(np.deg2rad(45.0))
    assert tracker.noise.std_cross == pytest.approx(6.0 * np.pi / (180.0 * 3600.0))
    assert tracker.noise.std_roll == pytest.approx(40.0 * np.pi / (180.0 * 3600.0))


def test_terma_factory_sets_expected_tracker_properties():
    tracker = create_terma_t1()
    assert tracker.fov == pytest.approx(np.deg2rad(22.0))
    assert tracker.sun_exclusion == pytest.approx(np.deg2rad(30.0))
    assert tracker.noise.std_cross == pytest.approx(2.0 * np.pi / (180.0 * 3600.0))
    assert tracker.noise.std_roll == pytest.approx(15.0 * np.pi / (180.0 * 3600.0))


def test_generic_factory_sets_expected_tracker_properties():
    tracker = create_generic_star_tracker(cross_arcsec=8.0, roll_arcsec=30.0, fov_deg=12.0, sun_exclusion_deg=40.0)
    assert tracker.fov == pytest.approx(np.deg2rad(12.0))
    assert tracker.sun_exclusion == pytest.approx(np.deg2rad(40.0))
    assert tracker.noise.std_cross == pytest.approx(8.0 * np.pi / (180.0 * 3600.0))
    assert tracker.noise.std_roll == pytest.approx(30.0 * np.pi / (180.0 * 3600.0))


def test_tracker_factory_normalizes_custom_boresight():
    boresight = np.array([0.0, 1.0, 0.0])
    tracker = create_bct_nst(boresight=boresight)
    np.testing.assert_allclose(tracker.boresight, boresight / np.linalg.norm(boresight), rtol=1e-10)


def test_star_tracker_noise_covariance_matches_body_z_alignment_case():
    tracker = make_tracker(std_cross=1e-4, std_roll=5e-4)
    expected = np.diag([1e-4**2, 1e-4**2, 5e-4**2])
    np.testing.assert_allclose(tracker.noise_covariance, expected, rtol=1e-10)


def test_star_tracker_noise_covariance_eigenvalues_match_for_rotated_boresight():
    tracker = make_tracker(boresight=np.array([1.0, 0.0, 0.0]), fov_deg=20.0, std_cross=1e-4, std_roll=5e-4)
    expected_eigenvalues = np.sort([1e-4**2, 1e-4**2, 5e-4**2])
    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(tracker.noise_covariance)), expected_eigenvalues, rtol=1e-10)
