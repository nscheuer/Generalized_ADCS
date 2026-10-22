"""Recomputed environment vs. JPL Horizons (committed fixtures, no network).

These check the quantities the viewer draws but never computes itself: the
Sun and Moon directions, and Earth's orientation through the sub-solar
longitude. Refresh the fixtures with ``tools/fetch_horizons_fixtures.py``.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from ADCS.orbits.ephemeris import Ephemeris
from ADCS.viz.derive import derive_environment

FIXTURE = Path(__file__).parent / "fixtures" / "horizons_sun_moon.json"
AU_KM = 149_597_870.7
ARCSEC = np.deg2rad(1.0 / 3600.0)


@pytest.fixture(scope="module")
def data():
    records = json.loads(FIXTURE.read_text())["records"]
    ephem = Ephemeris()
    jd = np.array([r["jd_tt"] for r in records])
    return records, ephem, jd, derive_environment(ephem, jd)


def _unit_radec(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra, dec = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _true_of_date(ephem: Ephemeris, jd: np.ndarray, v_gcrs: np.ndarray) -> np.ndarray:
    # Skyfield's t.M: frame bias + precession + nutation, GCRS -> true equator/equinox of date.
    M = np.asarray(ephem.ts.tt_jd(jd).M)
    M = np.transpose(M, (2, 0, 1)) if M.shape == (3, 3, len(jd)) else M
    return np.einsum("nij,nj->ni", M, v_gcrs)


@pytest.mark.parametrize("body, tol_arcsec", [("sun", 1.0), ("moon", 2.0)])
def test_direction_matches_horizons(data, body, tol_arcsec):
    records, ephem, jd, env = data
    v = _true_of_date(ephem, jd, env[f"r_{body}"])
    for rec, vk in zip(records, v):
        ref = _unit_radec(rec[f"{body}_ra_app_deg"], rec[f"{body}_dec_app_deg"])
        sep = np.arccos(np.clip(vk @ ref / np.linalg.norm(vk), -1.0, 1.0)) / ARCSEC
        assert sep < tol_arcsec, f"{body} at {rec['date_tt']}: {sep:.3f} arcsec"


@pytest.mark.parametrize("body", ["sun", "moon"])
def test_distance_matches_horizons(data, body):
    records, _, _, env = data
    for rec, r in zip(records, env[f"r_{body}"]):
        assert np.linalg.norm(r) / AU_KM == pytest.approx(rec[f"{body}_delta_au"], rel=1e-6)


def test_subsolar_longitude_matches_horizons_hour_angle(data):
    # Sub-solar longitude = -(apparent hour angle of the Sun at Greenwich). The
    # Horizons site is on the surface, so allow for solar parallax (8.8 arcsec).
    records, _, _, env = data
    for rec, R, s in zip(records, env["R_itrs_gcrs"], env["r_sun"]):
        s_itrs = R @ s
        lon = np.degrees(np.arctan2(s_itrs[1], s_itrs[0]))
        ref = -15.0 * rec["sun_local_hour_angle_greenwich_h"]
        diff = (lon - ref + 180.0) % 360.0 - 180.0
        assert abs(diff) * 3600.0 < 15.0, f"{rec['date_tt']}: {diff * 3600.0:.2f} arcsec"


def test_earth_orientation_quaternion_matches_matrix(data):
    from ADCS.helpers.math_helpers import rot_mat

    _, _, _, env = data
    for q, R in zip(env["q_itrs_gcrs"], env["R_itrs_gcrs"]):
        np.testing.assert_allclose(rot_mat(q), R, atol=1e-12)
