"""
Independent / analytic validation of orbit-dependent ``Goal.to_ref()``.

WHY THIS EXISTS (test-hardening backlog #6, tautology-audit sub-thread):
the magnetic-pointing controller tests (test_controller_mtq_lovera.py,
test_controller_mtq_wisniewski.py, test_controller_mtq_w_rw_lp.py) measure
"did it point correctly?" by comparing the achieved attitude against the
SAME ``goal.to_ref(os)`` the controller was driven toward. That is circular
for the bug-prone orbit-dependent goals: a frame / sign / degree-vs-radian
error in ``to_ref()`` (exactly the PR #36 class of bug) would be invisible --
both the controller and the test would use the identical wrong reference and
the test would still pass. The orbit-dependent goals had only *format*
checks anywhere in the suite (shape / NaN sentinel), never an independent
check of the actual reference DIRECTION.

This suite supplies that missing independent reference. Every expected value
here is derived from first-principles vector geometry / closed-form WGS84,
computed with plain numpy -- never from the goal code under test, never from
``ADCS.helpers.normalize``. It is test-only (no source change) and passes on
``origin/main`` (the geometry is currently sound); it goes RED on any future
sign / axis / frame / unit regression in ``*_Goal.to_ref()``.
"""

import numpy as np
import pytest

from ADCS.CONOPS.goals import (
    Nadir_Goal, Zenith_Goal, Velocity_Goal, AntiVelocity_Goal,
    Sun_Goal, AntiSun_Goal, BField_Goal, AntiBField_Goal, Coordinate_Goal,
)
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State

# Independent unit-normalize (numpy only -- NOT the codebase's normalize()).
def _u(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


@pytest.fixture(scope="module")
def os0():
    # Deliberately non-axis-aligned R and V so an axis permutation or a
    # single-component sign error cannot pass by coincidence.
    ep = Ephemeris()
    return Orbital_State(
        ephem=ep, J2000=0.137,
        R=np.array([6500.0, 1200.0, -900.0]),
        V=np.array([1.10, 7.00, 2.30]),
        S=np.array([1.2e8, -4.0e7, 3.0e7]),
        B=np.array([1.5e-5, -2.2e-5, 0.8e-5]),
        rho=0.0,
    )


def _check_vector_contract(r_ref, expected_dir):
    """Common contract: r_ref is (4,), r_ref[0] is the NaN 'vector-mode'
    sentinel, r_ref[1:4] is a finite unit vector equal to expected_dir."""
    r_ref = np.asarray(r_ref, float)
    assert r_ref.shape == (4,)
    assert np.isnan(r_ref[0]), "r_ref[0] must be the NaN vector-mode sentinel"
    assert np.all(np.isfinite(r_ref[1:4]))
    assert np.linalg.norm(r_ref[1:4]) == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(r_ref[1:4], expected_dir, atol=1e-9, rtol=0)


# ---- Spacecraft-geometry vector goals: exact closed-form direction ----

def test_nadir_is_negative_position_unit_vector(os0):
    r_ref, w_ref = Nadir_Goal().to_ref(os0)
    _check_vector_contract(r_ref, -_u(os0.R))
    # Feed-forward reference rate is the orbital angular velocity R x V/|R|^2.
    np.testing.assert_allclose(
        w_ref, np.cross(os0.R, os0.V) / np.dot(os0.R, os0.R), atol=1e-12, rtol=0)


def test_zenith_is_positive_position_unit_vector(os0):
    r_ref, _ = Zenith_Goal().to_ref(os0)
    _check_vector_contract(r_ref, _u(os0.R))


def test_velocity_is_velocity_unit_vector(os0):
    r_ref, w_ref = Velocity_Goal().to_ref(os0)
    _check_vector_contract(r_ref, _u(os0.V))
    np.testing.assert_allclose(
        w_ref, np.cross(os0.R, os0.V) / np.dot(os0.R, os0.R), atol=1e-12, rtol=0)


def test_antivelocity_is_negative_velocity_unit_vector(os0):
    r_ref, _ = AntiVelocity_Goal().to_ref(os0)
    _check_vector_contract(r_ref, -_u(os0.V))


def test_sun_goal_points_at_sun(os0):
    r_ref, w_ref = Sun_Goal().to_ref(os0)
    _check_vector_contract(r_ref, _u(os0.get_sun_eci()))
    np.testing.assert_allclose(w_ref, np.zeros(3), atol=1e-12, rtol=0)


def test_bfield_goal_points_along_B(os0):
    r_ref, _ = BField_Goal().to_ref(os0)
    _check_vector_contract(r_ref, _u(os0.get_b_eci()))


# ---- Cross-goal relational invariants (independent of any single goal) ----
# An "anti" goal must be exactly antiparallel to its base goal. A sign error
# in only one of the pair breaks this even if each individually "looked" unit.

@pytest.mark.parametrize("base,anti", [
    (Nadir_Goal, Zenith_Goal),
    (Velocity_Goal, AntiVelocity_Goal),
    (Sun_Goal, AntiSun_Goal),
    (BField_Goal, AntiBField_Goal),
])
def test_anti_goal_is_exactly_antiparallel(os0, base, anti):
    b = base().to_ref(os0)[0][1:4]
    a = anti().to_ref(os0)[0][1:4]
    np.testing.assert_allclose(a, -b, atol=1e-12, rtol=0)
    assert float(np.dot(_u(a), _u(b))) == pytest.approx(-1.0, abs=1e-9)


# ---- Coordinate_Goal: closed-form WGS84 geometry, independent of the
#      goal's own forward conversion. ----

WGS84_A = 6378.137          # km, equatorial radius
WGS84_B = 6356.7523142      # km, polar radius (a*sqrt(1-e^2))


def test_coordinate_equator_prime_meridian_is_plus_x():
    # lat=0, lon=0, alt=0 -> exactly [a, 0, 0] for any ellipsoid model.
    g = Coordinate_Goal(lat=0.0, lon=0.0, alt=0.0)
    np.testing.assert_allclose(g.target_ecef, [WGS84_A, 0.0, 0.0], atol=1e-3, rtol=0)


def test_coordinate_equator_90E_is_plus_y():
    g = Coordinate_Goal(lat=0.0, lon=90.0, alt=0.0)
    np.testing.assert_allclose(g.target_ecef, [0.0, WGS84_A, 0.0], atol=1e-3, rtol=0)


def test_coordinate_north_pole_is_plus_z_polar_radius():
    g = Coordinate_Goal(lat=90.0, lon=0.0, alt=0.0)
    p = np.asarray(g.target_ecef, float)
    assert np.hypot(p[0], p[1]) < 1e-6, "pole must be on the spin axis"
    assert p[2] == pytest.approx(WGS84_B, abs=1e-2)


@pytest.mark.parametrize("lat,lon,alt", [
    (37.4, -122.1, 0.0), (-33.9, 151.2, 0.5), (51.5, 0.0, 0.0), (0.0, 179.0, 2.0),
])
def test_coordinate_target_recovers_lat_lon_and_radius(lat, lon, alt):
    """Invert target_ecef with an INDEPENDENT algorithm (spherical atan2 for
    longitude; geocentric latitude bound; radius bracket) -- never the goal's
    own forward formula -- and check it round-trips."""
    p = np.asarray(Coordinate_Goal(lat=lat, lon=lon, alt=alt).target_ecef, float)
    # Longitude is exact: it is just the equatorial-plane azimuth.
    lon_rec = np.degrees(np.arctan2(p[1], p[0]))
    assert lon_rec == pytest.approx(lon, abs=1e-6)
    # Geocentric latitude differs from geodetic by <0.2 deg; sign/magnitude
    # must still match closely (catches a deg/rad or lat/lon swap instantly).
    lat_geoc = np.degrees(np.arctan2(p[2], np.hypot(p[0], p[1])))
    assert lat_geoc == pytest.approx(lat, abs=0.2)
    # On/above the ellipsoid: polar+alt <= |p| <= equatorial+alt.
    assert WGS84_B + alt - 1e-3 <= np.linalg.norm(p) <= WGS84_A + alt + 1e-3


def test_coordinate_to_ref_is_independent_line_of_sight(os0):
    """to_ref direction must be the unit LOS from the spacecraft to the
    ground target. ecef_to_eci is independently validated by PR #36; the
    goal-specific LOS/normalize/sentinel logic is what is pinned here."""
    g = Coordinate_Goal(lat=10.0, lon=20.0, alt=0.0)
    r_ref, _ = g.to_ref(os0)
    los = _u(os0.ecef_to_eci(g.target_ecef) - os0.R)
    _check_vector_contract(r_ref, los)
