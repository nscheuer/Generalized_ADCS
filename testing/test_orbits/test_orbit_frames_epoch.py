"""
Regression tests for orbit frames, epoch handling, and interpolation.

These tests check that the ECEF and geocentric frame transforms are exact
inverses, the J2000 epoch is interpreted consistently with Skyfield's TT time
scale, atmospheric density interpolation follows the intended exponential
behaviour, and ``Orbit.get_os`` returns accurately propagated states and
orthonormal frame matrices between stored orbit nodes.
"""

import numpy as np
import pytest

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.density_model import DensityModel
from ADCS.orbits.universal_constants import TimeConstants

EPHEM = Ephemeris()
J2000_TT_JD = 2451545.0  # J2000.0 epoch expressed in the TT time scale


def _os(j2000=0.22, R=None, V=None):
    return Orbital_State(
        ephem=EPHEM, J2000=j2000,
        R=np.array([7000.0, 0.0, 0.0]) if R is None else R,
        V=np.array([0.0, 7.5, 0.0]) if V is None else V,
        B=np.array([2e-5, -1e-5, 3e-5]),
    )


# --------------------------------------------------------------------------
# Bug 1: ECEF <-> geocentric must be exact inverses; basis orthonormal.
# --------------------------------------------------------------------------
def test_ecef_geocentric_roundtrip_is_identity():
    os_ = _os()
    rng = np.random.default_rng(0)
    for _ in range(25):
        v = rng.normal(size=3)
        assert np.allclose(os_.ecef_to_geocentric(os_.geocentric_to_ecef(v)), v, atol=1e-12)
        assert np.allclose(os_.geocentric_to_ecef(os_.ecef_to_geocentric(v)), v, atol=1e-12)
    M = os_._ecef_to_geo
    assert np.allclose(M @ M.T, np.eye(3), atol=1e-12)      # orthonormal
    assert np.isclose(np.linalg.det(M), 1.0, atol=1e-9)     # right-handed


# --------------------------------------------------------------------------
# Bug 2: J2000=0 must be the TT epoch, verified against Skyfield directly.
# --------------------------------------------------------------------------
def test_epoch_is_TT_not_TAI_vs_skyfield():
    ts = EPHEM.ts
    sun_at = lambda t: np.asarray(
        EPHEM.earth.at(t).observe(EPHEM.sun).apparent().position.km, float
    )
    ref_tt = sun_at(ts.tt_jd(J2000_TT_JD))    # correct interpretation
    ref_tai = sun_at(ts.tai_jd(J2000_TT_JD))  # the old (buggy) interpretation

    os0 = _os(j2000=0.0)
    sun = os0.get_sun_eci()

    # Skyfield is the external reference: the code must now agree with the TT
    # interpretation to machine precision...
    assert np.allclose(sun, ref_tt, rtol=0, atol=1e-3), \
        f"sun {sun} != TT ref {ref_tt}"
    # ...and the TT vs TAI difference must be physically significant (~32 s of
    # Earth heliocentric motion, hundreds of km) so this is a real test, and
    # the code must NOT match the old TAI mislabeling.
    assert np.linalg.norm(ref_tt - ref_tai) > 100.0
    assert np.linalg.norm(sun - ref_tai) > 100.0


# --------------------------------------------------------------------------
# Bug 4: density must be log-linear interior + exponential tail + guard.
# --------------------------------------------------------------------------
def test_density_log_interpolation_and_extrapolation():
    dm = DensityModel()
    alt, rho = dm.altitude_range, dm.rho_range

    # Exact at table nodes.
    for h, r in zip(alt, rho):
        assert np.isclose(dm.interpolate(float(h)), float(r), rtol=1e-9)

    # Interior: between two nodes the value is the GEOMETRIC (log-linear)
    # interp, which for an exponential atmosphere is within a small factor of
    # truth and FAR from the old linear-in-altitude blend.
    i = len(alt) // 2
    h_mid = 0.5 * (alt[i] + alt[i + 1])
    geom = np.sqrt(rho[i] * rho[i + 1])
    lin = 0.5 * (rho[i] + rho[i + 1])
    assert np.isclose(dm.interpolate(h_mid), geom, rtol=1e-9)
    assert abs(np.log(dm.interpolate(h_mid)) - np.log(geom)) \
        < abs(np.log(lin) - np.log(geom))  # closer to exponential than linear

    # Monotonic decreasing across a fine altitude sweep.
    sweep = np.linspace(alt[0] + 1.0, alt[-1] - 1.0, 200)
    d = np.array([dm.interpolate(h) for h in sweep])
    assert np.all(np.diff(d) <= 1e-30)

    # Above the table: exponential decay toward ~0 (not constant LEO tail).
    top = float(alt[-1])
    assert dm.interpolate(top + 5000.0) < dm.interpolate(top) * 1e-3
    assert dm.interpolate(1.0e6) < dm.interpolate(top) * 1e-12

    # Below the table / sub-surface: clamped, no crash.
    assert dm.interpolate(float(alt[0]) - 50.0) == pytest.approx(float(rho[0]))
    assert np.isfinite(dm.interpolate(-100.0))


# --------------------------------------------------------------------------
# Bug 3: get_os must propagate (not linearly blend) between nodes.
# --------------------------------------------------------------------------
def _build_orbit(dt_s=300.0, n=12):
    os0 = _os(j2000=0.0, R=np.array([6878.0, 0.0, 0.0]),
              V=np.array([0.0, 7.613, 0.0]))  # ~500 km circular
    states = [os0]
    for _ in range(n):
        states.append(states[-1].propagate_orbit_rk4(dt_s))
    return Orbit(states), os0, dt_s


def test_get_os_propagates_between_nodes_accurately():
    orbit, os0, dt_s = _build_orbit()
    t0 = orbit.times[0]
    t1 = orbit.times[1]

    # Exact-node query is unchanged.
    node = orbit.get_os(float(t1))
    assert np.allclose(node.R, orbit.states[orbit.times[1]].R, atol=1e-9)

    # Mid-node query: independent fine-RK4 truth (many small steps from node0).
    tm = 0.5 * (t0 + t1)
    truth = orbit.states[t0]
    fine = 400
    for _ in range(fine):
        truth = truth.propagate_orbit_rk4(0.5 * dt_s / fine)

    got = orbit.get_os(float(tm))
    pos_err = np.linalg.norm(got.R - truth.R)
    # Old linear blend was ~95 km here; propagation is RK4-truncation small.
    assert pos_err < 1.0, f"get_os mid-node position error {pos_err:.3f} km"

    # Frame matrices recomputed orthonormal (linear blend gave ~7% error).
    for M in (got._R_eci2ecef, got.ECI2ENUmat):
        assert np.allclose(M @ M.T, np.eye(3), atol=1e-6)
        assert np.isclose(abs(np.linalg.det(M)), 1.0, atol=1e-6)
