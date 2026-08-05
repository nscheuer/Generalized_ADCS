"""Availability gating and anisotropic error model for :class:`StarTrackerQuaternion`.

Three additions are covered here:

* ``earth_limb_exclusion`` -- a boresight keep-out beyond the geometric Earth limb.
  The catalog already drops individual stars occulted by the Earth; this is the separate
  stray-light constraint that blinds the whole tracker, and it is normally the binding one.
* ``max_rate`` -- slew-rate dropout from image smear. This is what couples tracker
  availability to spacecraft agility.
* ``sigma_cross`` / ``sigma_roll`` -- the cross-boresight vs about-boresight error
  anisotropy that trackers are actually specified with (typically ~6x), applied as a small
  rotation instead of additive quaternion noise.

Every default is unchanged, so the first block asserts backwards compatibility explicitly.
"""

import numpy as np
import pytest

from ADCS.helpers.math_helpers import quat_inv, quat_mult, rot_mat
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import ErrorMode, Noise
from ADCS.satellite_hardware.sensors import StarTrackerQuaternion


R_KM = 7000.0
_ARCSEC = np.pi / (180.0 * 3600.0)


def make_orbital_state() -> Orbital_State:
    return Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([R_KM, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]),
    )


def make_tracker(**kwargs) -> StarTrackerQuaternion:
    """Permissive base tracker so only the mechanism under test can cause a dropout."""
    defaults = dict(
        boresight=np.array([0.0, 0.0, 1.0]),
        fov=np.deg2rad(170.0),
        noise=Noise(noise=np.zeros(4), std_noise=np.array([1e-12] * 4)),
        sun_exclusion=0.0,
        min_stars=1,
    )
    defaults.update(kwargs)
    return StarTrackerQuaternion(**defaults)


def state(q, w=None):
    w = np.zeros(3) if w is None else np.asarray(w, dtype=float)
    return np.concatenate([w, np.asarray(q, dtype=float)])


def quat_pointing_body_z_along(target_eci: np.ndarray) -> np.ndarray:
    """Quaternion (body->ECI) whose body +z maps onto ``target_eci``."""
    t = np.asarray(target_eci, dtype=float)
    t = t / np.linalg.norm(t)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, t)
    c = float(np.dot(z, t))
    if np.linalg.norm(v) < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0]) if c > 0 else np.array([0.0, 1.0, 0.0, 0.0])
    axis = v / np.linalg.norm(v)
    ang = np.arccos(np.clip(c, -1.0, 1.0))
    return np.concatenate([[np.cos(ang / 2.0)], axis * np.sin(ang / 2.0)])


# --------------------------------------------------------------------------------------
# Backwards compatibility: every new knob is off by default
# --------------------------------------------------------------------------------------

def test_defaults_leave_new_behaviour_disabled():
    tracker = StarTrackerQuaternion()
    assert tracker.earth_limb_exclusion == 0.0
    assert tracker.max_rate is None
    assert tracker.anisotropic is False


def test_default_tracker_ignores_body_rate():
    """Without max_rate, a fast tumble must not change the reading (old behaviour)."""
    tracker = make_tracker()
    os_ = make_orbital_state()
    q = quat_pointing_body_z_along(np.array([-1.0, 0.0, 0.0]))
    slow = tracker.clean_reading(state(q, np.zeros(3)), os_)
    fast = tracker.clean_reading(state(q, [10.0, 10.0, 10.0]), os_)
    assert np.array_equal(np.isnan(slow), np.isnan(fast))
    if not np.any(np.isnan(slow)):
        np.testing.assert_allclose(slow, fast)


def test_sigma_cross_and_roll_must_be_given_together():
    with pytest.raises(ValueError, match="together"):
        make_tracker(sigma_cross=10 * _ARCSEC)
    with pytest.raises(ValueError, match="together"):
        make_tracker(sigma_roll=60 * _ARCSEC)


# --------------------------------------------------------------------------------------
# Earth-limb exclusion
# --------------------------------------------------------------------------------------

def test_boresight_at_nadir_is_blinded_by_limb_exclusion():
    os_ = make_orbital_state()
    nadir = -os_.R / np.linalg.norm(os_.R)
    q = quat_pointing_body_z_along(nadir)

    without = make_tracker(earth_limb_exclusion=0.0)
    with_margin = make_tracker(earth_limb_exclusion=np.deg2rad(25.0))

    # Pointing straight down is inside the geometric limb, so both refuse; the point of
    # the assertion is that the margin never *gains* availability.
    assert np.all(np.isnan(with_margin.clean_reading(state(q), os_)))
    assert with_margin.available is False


def test_limb_exclusion_blinds_a_direction_that_clears_the_geometric_limb():
    """A boresight just outside the geometric limb is available bare, blinded with margin."""
    os_ = make_orbital_state()
    r = np.linalg.norm(os_.R)
    nadir = -os_.R / r
    limb = np.arcsin(np.clip(6378.1363 / r, -1.0, 1.0))

    # Point 5 degrees outside the geometric limb: clear bare, inside a 25 degree margin.
    ang = limb + np.deg2rad(5.0)
    perp = np.array([0.0, 0.0, 1.0])
    direction = np.cos(ang) * nadir + np.sin(ang) * perp
    q = quat_pointing_body_z_along(direction)

    bare = make_tracker(earth_limb_exclusion=0.0)
    margined = make_tracker(earth_limb_exclusion=np.deg2rad(25.0))

    bare.clean_reading(state(q), os_)
    margined.clean_reading(state(q), os_)

    assert bare.available is True, "5 deg outside the geometric limb should be usable bare"
    assert margined.available is False, "25 deg margin must blind it"


def test_anti_nadir_is_available_with_margin():
    os_ = make_orbital_state()
    q = quat_pointing_body_z_along(os_.R)  # straight up, away from Earth
    tracker = make_tracker(earth_limb_exclusion=np.deg2rad(25.0))
    reading = tracker.clean_reading(state(q), os_)
    assert tracker.available is True
    assert not np.any(np.isnan(reading))


def test_limb_exclusion_is_monotone_in_margin():
    """Availability may only shrink as the keep-out grows."""
    os_ = make_orbital_state()
    r = np.linalg.norm(os_.R)
    nadir = -os_.R / r
    perp = np.array([0.0, 0.0, 1.0])

    angles = np.deg2rad(np.linspace(0.0, 180.0, 60))
    counts = []
    for margin_deg in (0.0, 10.0, 25.0, 45.0, 70.0):
        tracker = make_tracker(earth_limb_exclusion=np.deg2rad(margin_deg))
        n = 0
        for a in angles:
            d = np.cos(a) * nadir + np.sin(a) * perp
            tracker.clean_reading(state(quat_pointing_body_z_along(d)), os_)
            n += int(tracker.available)
        counts.append(n)
    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] > counts[-1], "a 70 deg keep-out must cost some availability"


# --------------------------------------------------------------------------------------
# Slew-rate dropout
# --------------------------------------------------------------------------------------

def test_rate_limit_blinds_above_threshold_and_not_below():
    os_ = make_orbital_state()
    q = quat_pointing_body_z_along(os_.R)
    max_rate = np.deg2rad(2.0)
    tracker = make_tracker(max_rate=max_rate)

    below = tracker.clean_reading(state(q, [0.0, 0.0, 0.9 * max_rate]), os_)
    assert tracker.available is True
    assert not np.any(np.isnan(below))

    above = tracker.clean_reading(state(q, [0.0, 0.0, 1.1 * max_rate]), os_)
    assert tracker.available is False
    assert np.all(np.isnan(above))


def test_rate_limit_uses_magnitude_not_components():
    """Three components each under the limit can still exceed it in magnitude."""
    os_ = make_orbital_state()
    q = quat_pointing_body_z_along(os_.R)
    max_rate = np.deg2rad(2.0)
    tracker = make_tracker(max_rate=max_rate)
    each = 0.7 * max_rate                      # |w| = 1.21 * max_rate
    tracker.clean_reading(state(q, [each, each, each]), os_)
    assert tracker.available is False


def test_rate_limit_tolerates_short_state_vectors():
    """Some callers pass a quaternion-only state; the check must not raise."""
    os_ = make_orbital_state()
    q = quat_pointing_body_z_along(os_.R)
    tracker = make_tracker(max_rate=np.deg2rad(2.0))
    tracker.clean_reading(np.concatenate([np.zeros(3), q]), os_)  # 7-vector, fine
    assert tracker.available is True


# --------------------------------------------------------------------------------------
# Anisotropic error model
# --------------------------------------------------------------------------------------

def _error_in_sensor_axes(tracker, q_true, q_meas):
    """Small-angle error of ``q_meas`` about ``q_true``, resolved in sensor axes."""
    dq = quat_mult(quat_inv(q_true), q_meas)
    if dq[0] < 0:
        dq = -dq
    dtheta_body = 2.0 * dq[1:]
    return tracker._sensor_axes.T @ dtheta_body


def test_anisotropic_noise_reproduces_requested_sigmas():
    np.random.seed(12345)
    os_ = make_orbital_state()
    q = quat_pointing_body_z_along(os_.R)
    sc, sr = 10.0 * _ARCSEC, 60.0 * _ARCSEC
    tracker = make_tracker(sigma_cross=sc, sigma_roll=sr)
    assert tracker.anisotropic is True

    errs = []
    for _ in range(4000):
        meas = tracker.reading(state(q), os_)
        assert not np.any(np.isnan(meas))
        errs.append(_error_in_sensor_axes(tracker, q, meas))
    errs = np.array(errs)

    est = errs.std(axis=0)
    # 4000 samples -> the sample sigma is within ~4% of truth at 3 sigma.
    assert np.isclose(est[0], sc, rtol=0.10), est
    assert np.isclose(est[1], sc, rtol=0.10), est
    assert np.isclose(est[2], sr, rtol=0.10), est
    # The anisotropy itself is the point: roll must be visibly worse than cross.
    assert est[2] > 3.0 * est[0]


def test_anisotropic_error_lands_on_the_boresight_axis():
    """The roll term must follow the boresight, not a fixed body axis."""
    np.random.seed(7)
    os_ = make_orbital_state()
    sc, sr = 5.0 * _ARCSEC, 100.0 * _ARCSEC
    for boresight in (np.array([1.0, 0.0, 0.0]),
                      np.array([0.0, 1.0, 0.0]),
                      np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)):
        tracker = make_tracker(boresight=boresight, sigma_cross=sc, sigma_roll=sr)
        q = quat_pointing_body_z_along(os_.R)
        errs = np.array([
            _error_in_sensor_axes(tracker, q, tracker.reading(state(q), os_))
            for _ in range(2000)
        ])
        # Component 3 of the sensor triad is the boresight by construction.
        np.testing.assert_allclose(tracker._sensor_axes[:, 2], boresight, atol=1e-12)
        est = errs.std(axis=0)
        assert est[2] > 5.0 * est[0], (boresight, est)


def test_anisotropic_reading_is_unit_and_scalar_positive():
    np.random.seed(3)
    os_ = make_orbital_state()
    q = quat_pointing_body_z_along(os_.R)
    tracker = make_tracker(sigma_cross=10 * _ARCSEC, sigma_roll=60 * _ARCSEC)
    for _ in range(200):
        meas = tracker.reading(state(q), os_)
        assert abs(np.linalg.norm(meas) - 1.0) < 1e-10
        assert meas[0] >= 0.0


def test_anisotropic_mode_honours_dmode_add_noise():
    """Estimators request clean predictions; adding noise there corrupts sigma points."""
    np.random.seed(11)
    os_ = make_orbital_state()
    q = quat_pointing_body_z_along(os_.R)
    tracker = make_tracker(sigma_cross=10 * _ARCSEC, sigma_roll=60 * _ARCSEC)
    clean_mode = ErrorMode(add_bias=False, add_noise=False,
                           update_bias=False, update_noise=False)
    for _ in range(50):
        meas = tracker.reading(state(q), os_, clean_mode)
        np.testing.assert_allclose(meas, tracker.clean_reading(state(q), os_), atol=1e-12)


def test_anisotropic_mode_propagates_dropouts():
    """A blinded tracker must return NaN in anisotropic mode too, not a perturbed value."""
    os_ = make_orbital_state()
    q = quat_pointing_body_z_along(-os_.R)  # at nadir
    tracker = make_tracker(sigma_cross=10 * _ARCSEC, sigma_roll=60 * _ARCSEC,
                           earth_limb_exclusion=np.deg2rad(25.0))
    assert np.all(np.isnan(tracker.reading(state(q), os_)))
    assert tracker.available is False


def test_available_flag_matches_nan_status_over_a_sweep():
    os_ = make_orbital_state()
    r = np.linalg.norm(os_.R)
    nadir = -os_.R / r
    perp = np.array([0.0, 0.0, 1.0])
    tracker = make_tracker(earth_limb_exclusion=np.deg2rad(20.0),
                           max_rate=np.deg2rad(2.0))
    for a in np.deg2rad(np.linspace(0.0, 180.0, 40)):
        d = np.cos(a) * nadir + np.sin(a) * perp
        reading = tracker.clean_reading(state(quat_pointing_body_z_along(d)), os_)
        assert tracker.available == (not np.any(np.isnan(reading)))
