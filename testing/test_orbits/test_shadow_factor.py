"""Conical shadow factor (Montenbruck & Gill 3.4.2).

The 3-D viewer dims sunlight by this value, so it has to agree with the
binary eclipse test the simulation already uses wherever the two models
should agree, and behave sensibly across the penumbra.
"""

import numpy as np
import pytest

from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.shadow import shadow_factor
from ADCS.orbits.universal_constants import EarthConstants

AU = 149_597_870.7
SUN = np.array([AU, 0.0, 0.0])


def test_full_sun_on_the_day_side():
    assert shadow_factor(np.array([7000.0, 0.0, 0.0]), SUN) == 1.0


def test_umbra_directly_behind_earth():
    assert shadow_factor(np.array([-7000.0, 0.0, 0.0]), SUN) == 0.0


def test_scalar_and_vector_shapes():
    r = np.array([[7000.0, 0.0, 0.0], [-7000.0, 0.0, 0.0]])
    nu = shadow_factor(r, np.tile(SUN, (2, 1)))
    assert nu.shape == (2,)
    np.testing.assert_array_equal(nu, [1.0, 0.0])
    assert isinstance(shadow_factor(r[0], SUN), float)


def test_annular_case_matches_disk_area_ratio():
    # A 1 km occulter seen from 1e4 km is much smaller than the solar disk.
    r_sat = np.array([-1.0e4, 0.0, 0.0])
    nu = shadow_factor(r_sat, SUN, R_occ=1.0)
    a = np.arcsin(EarthConstants.R_sun / np.linalg.norm(SUN - r_sat))
    b = np.arcsin(1.0 / 1.0e4)
    assert nu == pytest.approx(1.0 - (b / a) ** 2, rel=1e-12)


def test_penumbra_is_monotonic_across_shadow_entry():
    # Circular equatorial orbit, sweeping the anomaly through shadow entry.
    radius = 7000.0
    theta = np.linspace(np.deg2rad(100.0), np.deg2rad(170.0), 20001)
    r = radius * np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
    nu = shadow_factor(r, np.tile(SUN, (theta.size, 1)))
    assert nu[0] == 1.0 and nu[-1] == 0.0
    assert np.all(np.diff(nu) <= 1e-12)
    partial = (nu > 0.0) & (nu < 1.0)
    assert partial.any(), "the sweep should resolve the penumbra"
    # In this geometry the Earth-centre/Sun-centre separation seen from the
    # spacecraft is pi - theta, so the penumbra spans the solar diameter 2a
    # of anomaly (~0.53 deg, ~8 s in LEO).
    width = theta[partial][-1] - theta[partial][0]
    two_a = 2.0 * np.arcsin(EarthConstants.R_sun / AU)
    assert width == pytest.approx(two_a, rel=0.02)


def test_inside_the_body_is_rejected():
    with pytest.raises(ValueError):
        shadow_factor(np.array([100.0, 0.0, 0.0]), SUN)


def test_agrees_with_binary_eclipse_outside_the_penumbra():
    ephem = Ephemeris()
    radius = 7000.0
    theta = np.linspace(0.0, 2.0 * np.pi, 73, endpoint=False)
    checked = 0
    for th in theta:
        r = radius * np.array([np.cos(th), 0.3 * np.sin(th), np.sin(th)])
        r *= radius / np.linalg.norm(r)
        os = Orbital_State(ephem=ephem, J2000=0.22, R=r, V=np.array([0.0, 7.5, 0.0]), fast=True)
        nu = shadow_factor(os.R, os.S)
        if 0.0 < nu < 1.0:
            continue
        assert os.is_sunlit() == (nu == 1.0), f"theta={th:.3f}, nu={nu}"
        checked += 1
    assert checked > 60


def test_scalar_path_matches_the_vectorized_path():
    # Single positions take a math-module fast path; it must agree everywhere,
    # including through the penumbra.
    theta = np.linspace(np.deg2rad(100.0), np.deg2rad(170.0), 4001)
    r = 7000.0 * np.column_stack([np.cos(theta), np.sin(theta), 0.01 * np.ones_like(theta)])
    vec = shadow_factor(r, np.tile(SUN, (theta.size, 1)))
    scal = np.array([shadow_factor(rk, SUN) for rk in r])
    np.testing.assert_allclose(scal, vec, rtol=0, atol=1e-12)
    annular = np.array([-1.0e4, 0.0, 0.0])
    assert shadow_factor(annular, SUN, R_occ=1.0) == pytest.approx(
        float(shadow_factor(annular[None, :], SUN[None, :], R_occ=1.0)[0]), abs=1e-15)


def test_one_position_broadcasts_against_many():
    # A fixed spacecraft against a Sun history, and the reverse, give the same values as matching rows.
    r = np.array([7000.0, 0.0, 0.0])
    suns = np.array([[-1.5e8, 0.0, 0.0], [1.5e8, 0.0, 0.0], [0.0, 1.5e8, 0.0]])
    nu = shadow_factor(r, suns)
    assert nu.shape == (3,)
    np.testing.assert_allclose(nu, [shadow_factor(r, s) for s in suns])
    assert nu[0] == 0.0 and nu[1] == 1.0  # Sun behind the Earth / behind the spacecraft
    sats = np.array([[7000.0, 0.0, 0.0], [-7000.0, 0.0, 0.0]])
    np.testing.assert_allclose(shadow_factor(sats, suns[1]), [1.0, 0.0])
    with pytest.raises(ValueError, match="different numbers of rows"):
        shadow_factor(sats, suns)
    with pytest.raises(ValueError, match=r"\(3,\) or \(N, 3\)"):
        shadow_factor(np.zeros((2, 2)), suns[0])
