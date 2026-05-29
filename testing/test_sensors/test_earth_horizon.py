import numpy as np
import numdifftools as nd
import pytest

from ADCS.satellite_hardware.sensors import EarthHorizonSensor
from ADCS.satellite_factory.sensors import (
    create_generic_earth_horizon,
    create_irst_horizon_sensor,
)
from ADCS.satellite_hardware.errors import ErrorMode, Noise
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_helpers import random_n_unit_vec, rot_mat


def _make_sensor(fov_deg=90.0, noise_std=0.0):
    """Helper to create an EHS with default parameters."""
    noise = None
    if noise_std > 0:
        noise = Noise(
            noise=np.zeros(3),
            std_noise=np.array([noise_std] * 3),
        )
    return EarthHorizonSensor(
        boresight=np.array([0.0, 0.0, -1.0]),
        fov=np.deg2rad(fov_deg),
        noise=noise,
    )


def _make_os(R=None, V=None):
    """Helper to create a standard orbital state."""
    if R is None:
        R = np.array([7000.0, 0.0, 0.0])
    if V is None:
        V = np.array([0.0, 7.5, 0.0])
    ephem = Ephemeris()
    return Orbital_State(ephem=ephem, J2000=0.22, R=R, V=V)


# ---- Clean reading ----

def test_clean_reading_returns_nadir():
    sensor = _make_sensor()
    os = _make_os()

    # Identity quaternion: body = ECI
    # Nadir in ECI: -R/|R| = [-1, 0, 0]
    # Body boresight is [0,0,-1], FOV=90 deg
    # nadir_body = R(q)^T @ nadir_eci = nadir_eci = [-1, 0, 0]
    # Angle between boresight [0,0,-1] and nadir [-1,0,0] = 90 deg = FOV limit
    # Use a quaternion that rotates nadir into boresight direction
    # Rotate so that -x_eci maps near -z_body
    # q that rotates [1,0,0] -> [0,0,1] is 90 deg about y
    angle = np.pi / 4  # 45 deg about y
    q = np.array([np.cos(angle / 2), 0, np.sin(angle / 2), 0])
    x = np.concatenate([np.zeros(3), q])

    n = sensor.clean_reading(x, os)

    if not np.any(np.isnan(n)):
        # Nadir in ECI is [-1, 0, 0]
        A = rot_mat(q)
        expected = A.T @ np.array([-1.0, 0.0, 0.0])
        np.testing.assert_allclose(n, expected, atol=1e-10)


def test_output_is_unit_vector():
    sensor = _make_sensor(fov_deg=180.0)
    os = _make_os()

    for _ in range(10):
        q = random_n_unit_vec(4)
        x = np.concatenate([np.zeros(3), q])

        n = sensor.clean_reading(x, os)

        if not np.any(np.isnan(n)):
            assert abs(np.linalg.norm(n) - 1.0) < 1e-10


def test_nadir_outside_fov_returns_nan():
    """If boresight points away from Earth, reading should be NaN."""
    # Boresight points in +z, nadir is along -x for R=[7000,0,0]
    sensor = EarthHorizonSensor(
        boresight=np.array([0.0, 0.0, 1.0]),
        fov=np.deg2rad(10.0),
    )
    os = _make_os()

    # Identity quaternion: nadir_body = [-1,0,0], boresight = [0,0,1]
    # Angle = 90 deg > 10 deg FOV → NaN
    x = np.zeros(7)
    x[3:7] = [1.0, 0.0, 0.0, 0.0]

    n = sensor.clean_reading(x, os)
    assert np.all(np.isnan(n))


def test_full_hemisphere_fov():
    """With 180-deg half-cone FOV, nadir should always be visible."""
    sensor = _make_sensor(fov_deg=180.0)
    os = _make_os()

    for _ in range(20):
        q = random_n_unit_vec(4)
        x = np.concatenate([np.zeros(3), q])
        n = sensor.clean_reading(x, os)
        assert not np.any(np.isnan(n)), "Should always be visible with 180-deg FOV"


# ---- Altitude independence ----

def test_altitude_independence():
    """Nadir direction should be correct at various altitudes."""
    sensor = _make_sensor(fov_deg=180.0)

    altitudes_km = [400, 800, 2000, 10000, 36000]
    R_earth = 6378.137

    for alt in altitudes_km:
        r_mag = R_earth + alt
        R = np.array([r_mag, 0.0, 0.0])
        os = _make_os(R=R)

        x = np.zeros(7)
        x[3:7] = [1.0, 0.0, 0.0, 0.0]

        n = sensor.clean_reading(x, os)

        if not np.any(np.isnan(n)):
            expected_nadir = np.array([-1.0, 0.0, 0.0])
            np.testing.assert_allclose(n, expected_nadir, atol=1e-10)


def test_earth_angular_radius():
    """Check that Earth angular radius is computed correctly."""
    sensor = _make_sensor(fov_deg=180.0)
    r_mag = 7000.0
    os = _make_os(R=np.array([r_mag, 0.0, 0.0]))

    x = np.zeros(7)
    x[3:7] = [1.0, 0.0, 0.0, 0.0]

    sensor.clean_reading(x, os)

    # Use the same R_earth the sensor uses internally
    expected_rho = np.arcsin(sensor._R_earth / r_mag)
    assert sensor.earth_angular_radius == pytest.approx(expected_rho, rel=1e-10)


# ---- Jacobians ----

def test_jacobian_finite_difference():
    sensor = _make_sensor(fov_deg=180.0)
    os = _make_os()

    q = np.array([0.9, 0.2, 0.3, 0.1])
    q = q / np.linalg.norm(q)
    x = np.concatenate([np.zeros(3), q])

    n0 = sensor.clean_reading(x, os)
    if np.any(np.isnan(n0)):
        return

    J_ana = sensor.basestate_jac(x, os)

    # Nadir ECI is fixed for a given orbit state
    nadir_eci = -os.R / np.linalg.norm(os.R)

    def measurement_func(state):
        q_test = state[3:7]
        A = rot_mat(q_test)
        return A.T @ nadir_eci

    J_fd = nd.Jacobian(measurement_func)(x)

    # Compare quaternion block
    np.testing.assert_allclose(
        J_ana[3:7, :], J_fd[:, 3:7].T, rtol=1e-4, atol=1e-8
    )


def test_omega_jacobian_is_zero():
    sensor = _make_sensor(fov_deg=180.0)
    os = _make_os()

    x = np.zeros(7)
    x[3:7] = [1.0, 0.0, 0.0, 0.0]

    sensor.clean_reading(x, os)
    J = sensor.basestate_jac(x, os)
    np.testing.assert_allclose(J[0:3, :], np.zeros((3, 3)), atol=1e-15)


def test_jacobian_zero_when_nan():
    """Jacobian should be zero when measurement is NaN (outside FOV)."""
    sensor = EarthHorizonSensor(
        boresight=np.array([0.0, 0.0, 1.0]),
        fov=np.deg2rad(10.0),
    )
    os = _make_os()

    x = np.zeros(7)
    x[3:7] = [1.0, 0.0, 0.0, 0.0]

    sensor.clean_reading(x, os)
    J = sensor.basestate_jac(x, os)
    np.testing.assert_allclose(J, np.zeros((7, 3)), atol=1e-15)


def test_bias_jacobian_no_estimate():
    sensor = _make_sensor()
    os = _make_os()

    x = np.zeros(7)
    x[3:7] = [1.0, 0.0, 0.0, 0.0]

    J_bias = sensor.bias_jac(x, os)
    assert J_bias.shape == (0, 3)


def test_bias_jacobian_with_estimate():
    sensor = EarthHorizonSensor(estimate_bias=True)
    os = _make_os()

    x = np.zeros(7)
    x[3:7] = [1.0, 0.0, 0.0, 0.0]

    J_bias = sensor.bias_jac(x, os)
    np.testing.assert_allclose(J_bias, np.eye(3))


# ---- Noisy reading ----

def test_noisy_reading_unit_vector():
    sensor = _make_sensor(fov_deg=180.0, noise_std=1e-3)
    os = _make_os()

    for _ in range(10):
        q = random_n_unit_vec(4)
        x = np.concatenate([np.zeros(3), q])

        n = sensor.reading(x, os)

        if not np.any(np.isnan(n)):
            assert abs(np.linalg.norm(n) - 1.0) < 1e-10


def test_noise_angular_spread():
    """Verify noise produces angular errors consistent with noise std."""
    noise_std = np.deg2rad(0.5)
    noise = Noise(
        noise=np.zeros(3),
        std_noise=np.array([noise_std] * 3),
    )
    sensor = EarthHorizonSensor(
        boresight=np.array([0.0, 0.0, -1.0]),
        fov=np.deg2rad(180.0),
        noise=noise,
    )
    os = _make_os()
    dmode = ErrorMode(add_bias=False, add_noise=True, update_bias=False, update_noise=True)

    N = 500
    angles = np.zeros(N)

    x = np.zeros(7)
    x[3:7] = [1.0, 0.0, 0.0, 0.0]

    # Get clean reading once (deterministic)
    clean = sensor.clean_reading(x, os)

    for i in range(N):
        noisy = sensor.reading(x, os, dmode=dmode)
        cos_angle = np.clip(np.dot(noisy, clean), -1.0, 1.0)
        angles[i] = np.arccos(cos_angle)

    # RMS angular error should be approximately sqrt(2) * noise_std
    # (two orthogonal noise components contribute to the angular error)
    rms_angle = np.sqrt(np.mean(angles**2))
    expected_rms = noise_std * np.sqrt(2)
    assert rms_angle < expected_rms * 3, f"RMS angle {np.rad2deg(rms_angle):.3f} deg too large"
    assert rms_angle > expected_rms * 0.3, f"RMS angle {np.rad2deg(rms_angle):.3f} deg too small"


# ---- Factory functions ----

def test_create_generic_earth_horizon():
    sensor = create_generic_earth_horizon(fov_deg=60.0, noise_deg=1.0)
    assert sensor.output_length == 3
    assert sensor.fov == pytest.approx(np.deg2rad(60.0))


def test_create_irst_horizon_sensor():
    sensor = create_irst_horizon_sensor()
    assert sensor.output_length == 3
    assert sensor.fov == pytest.approx(np.deg2rad(60.0))


def test_factory_custom_boresight():
    boresight = np.array([1.0, 0.0, 0.0])
    sensor = create_generic_earth_horizon(boresight=boresight)
    np.testing.assert_allclose(
        sensor.boresight,
        boresight / np.linalg.norm(boresight),
        rtol=1e-10,
    )


if __name__ == "__main__":
    test_clean_reading_returns_nadir()
    test_output_is_unit_vector()
    test_nadir_outside_fov_returns_nan()
    test_full_hemisphere_fov()
    test_altitude_independence()
    test_earth_angular_radius()
    test_jacobian_finite_difference()
    test_omega_jacobian_is_zero()
    test_jacobian_zero_when_nan()
    test_bias_jacobian_no_estimate()
    test_bias_jacobian_with_estimate()
    test_noisy_reading_unit_vector()
    test_noise_distribution_KS()
    test_create_generic_earth_horizon()
    test_create_irst_horizon_sensor()
    test_factory_custom_boresight()
    print("All tests passed.")
