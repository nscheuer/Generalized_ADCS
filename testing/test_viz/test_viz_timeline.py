"""Timeline content (ADCS.viz.timeline): bookmarks, ground sites and passes."""

import numpy as np
import pytest

import ADCS
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.state import State
from ADCS.viz import VizLog, results_to_vizlog
from ADCS.viz import timeline as tl
from ADCS.viz.derive import derive_environment

EPOCH_JD = 2451545.0 + 0.22 * 36525.0  # J2000 = 0.22 centuries


class ArrayLog:
    """Stand-in for VizLog.read over in-memory columns."""

    def __init__(self, **cols):
        self.cols = {k: np.asarray(v, float).reshape(len(v), -1) for k, v in cols.items()}

    def read(self, run_id, name):
        return self.cols[name]


@pytest.fixture(scope="module")
def orbit():
    """Two hours of a 500 km, 51.6 deg circular orbit with the real Earth orientation."""
    ephem = Ephemeris()
    t = np.arange(0.0, 7200.0, 10.0)
    mu, r = 398600.4418, 6878.137
    n = np.sqrt(mu / r ** 3)
    inc = np.radians(51.6)
    u = n * t
    r_sc = r * np.c_[np.cos(u), np.sin(u) * np.cos(inc), np.sin(u) * np.sin(inc)]
    env = derive_environment(ephem, EPOCH_JD + t / 86400.0)
    return ephem, t, r_sc, env["q_itrs_gcrs"]


def test_site_position_on_the_ellipsoid():
    p, up = tl.site_itrs(tl.GroundSite("equator", 0.0, 0.0))
    np.testing.assert_allclose(p, [6378137.0, 0, 0], atol=1e-6)
    p, up = tl.site_itrs(tl.GroundSite("pole", 90.0, 0.0, alt_m=100.0))
    np.testing.assert_allclose(p, [0, 0, 6356752.314245 + 100.0], atol=1e-3)
    np.testing.assert_allclose(up, [0, 0, 1], atol=1e-12)


def test_elevation_matches_skyfield_topocentric(orbit):
    from skyfield.api import wgs84

    ephem, t, r_sc, q = orbit
    site = tl.GroundSite("Boston", 42.36, -71.06, alt_m=40.0)
    ours = tl.elevation_deg(r_sc, q, site)
    ts = ephem.ts.tt_jd(EPOCH_JD + t / 86400.0)
    topos = wgs84.latlon(site.lat_deg, site.lon_deg, elevation_m=site.alt_m)
    obs = topos.at(ts).position.m.T                       # site in GCRS, independent geodesy and Earth rotation
    zen = topos.rotation_at(ts)[2].T                      # local zenith (rows: north, west, zenith)
    los = r_sc * 1000.0 - obs
    ref = np.degrees(np.arcsin(np.einsum("ni,ni->n", los, zen) / np.linalg.norm(los, axis=1)))
    np.testing.assert_allclose(ours, ref, atol=0.01)


def test_passes_of_an_analytic_elevation_curve():
    t = np.arange(0.0, 101.0)
    e = 30.0 * np.sin(2 * np.pi * t / 50.0) - 10.0  # elevation 30 sin(.) minus a 10 deg threshold
    passes = tl._spans(t, e, 10.0)
    rise = np.degrees(np.arcsin(1 / 3)) / 360.0 * 50.0
    assert len(passes) == 2
    for k, p in enumerate(passes):
        assert p.t_rise == pytest.approx(50 * k + rise, abs=0.05)
        assert p.t_set == pytest.approx(50 * k + 25 - rise, abs=0.05)
        assert p.t_max == pytest.approx(50 * k + 12.5, abs=0.05)
        assert p.max_elev_deg == pytest.approx(30.0, abs=0.05)
        assert not p.partial
    # Starting inside a pass: partial, beginning at the first sample.
    first = tl._spans(t[10:], e[10:], 10.0)[0]
    assert first.partial and first.t_rise == 10.0


def test_ground_passes_of_an_orbit_agree_with_the_elevation(orbit):
    ephem, t, r_sc, q = orbit
    log = ArrayLog(t_s=t, r_sc=r_sc, q_itrs_gcrs=q)
    # A site under the ground track at t = 1200 s sees a high pass around then.
    sub = np.einsum("ij,j->i", tl._rot(q[120:121])[0], r_sc[120])
    lat, lon = np.degrees(np.arcsin(sub[2] / np.linalg.norm(sub))), np.degrees(np.arctan2(sub[1], sub[0]))
    site = tl.GroundSite("under", float(lat), float(lon), min_elev_deg=10.0)
    passes = tl.ground_passes(log, site)
    assert any(p.t_rise < 1200.0 < p.t_set and p.max_elev_deg > 80.0 for p in passes)
    e = tl.elevation_deg(r_sc, q, site)
    for p in passes:
        inside = (t > p.t_rise + 1e-9) & (t < p.t_set - 1e-9)
        assert np.all(e[inside] >= 10.0) and p.max_elev_deg >= e[inside].max() - 1e-9
        assert 120.0 < p.t_set - p.t_rise < 900.0  # a LEO pass above 10 deg lasts minutes


def test_validation_resolves_and_checks():
    tlj = tl.validate_timeline([tl.Bookmark(t_s=50.0, label="b"), {"t_s": 10.0, "label": "a"}],
                               [tl.GroundSite("A", 10.0, 20.0)], t_range=(0.0, 100.0))
    assert [b["label"] for b in tlj["bookmarks"]] == ["a", "b"]  # sorted by time
    assert len({b["id"] for b in tlj["bookmarks"]}) == 2
    s = tlj["sites"][0]
    assert s["min_elev_deg"] == 10.0 and len(s["itrs_m"]) == 3 and len(s["up_itrs"]) == 3
    with pytest.raises(ValueError, match="outside the log"):
        tl.validate_timeline([tl.Bookmark(t_s=101.0)], [], t_range=(0.0, 100.0))
    with pytest.raises(ValueError, match="needs t_s or utc"):
        tl.validate_timeline([tl.Bookmark(label="x")], [])
    with pytest.raises(ValueError, match="epoch"):
        tl.validate_timeline([tl.Bookmark(utc="2026-01-01T00:00:00Z")], [])
    with pytest.raises(ValueError, match="duplicate site"):
        tl.validate_timeline([], [tl.GroundSite("A", 0, 0), tl.GroundSite("A", 1, 1)])
    with pytest.raises(ValueError, match="out of range"):
        tl.validate_timeline([], [tl.GroundSite("A", 95.0, 0)])


def test_sites_from_goals():
    from ADCS.CONOPS.goallist import GoalList

    g1 = ADCS.goals.Coordinate_Goal(lat=42.36, lon=-71.09, alt=0.1)
    g2 = ADCS.goals.Coordinate_Goal(lat=33.75, lon=-84.39, alt=0)
    assert tl.sites_from_goals(None) == []
    one = tl.sites_from_goals(g1)
    assert len(one) == 1 and one[0].name == "target" and one[0].alt_m == pytest.approx(100.0)
    many = tl.sites_from_goals(GoalList({0.22: g1, 0.2201: g2, 0.2202: g1}))
    assert [s.name for s in many] == ["target 1", "target 2"]


def _short_run(goal=None, tf=30.0):
    sat = ADCS.Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]),
                         actuators=[ADCS.MTQ(axis=a, max_torque=0.1) for a in np.eye(3)],
                         sensors=[ADCS.MTM(axis=a) for a in np.eye(3)], boresight=np.array([0, 0, 1]))
    os0 = Orbital_State(ephem=Ephemeris(), J2000=0.22, R=np.array([5000.0, 0, 5000.0]), V=np.array([0, 7.5, 0]))
    x0 = State.from_array(np.array([0.01, 0.05, 0, 1, 0, 0, 0]))
    return dict(x=x0, satellite=sat, os0=os0, dt=5.0, tf=tf, goal=goal)


def test_vizlog_timeline_from_results(tmp_path):
    res = ADCS.simulate(**_short_run())
    log = VizLog.open(results_to_vizlog(res, tmp_path / "a.vizlog", bookmarks=[
        tl.Bookmark(t_s=10.0, label="ten"),
        tl.Bookmark(utc=VizLog.open(results_to_vizlog(res, tmp_path / "e.vizlog")).manifest["epoch"]["iso_utc"],
                    label="start")], sites=[tl.GroundSite("Boston", 42.36, -71.06)]))
    b = {x["label"]: x["t_s"] for x in log.scene["timeline"]["bookmarks"]}
    assert b["ten"] == 10.0 and b["start"] == pytest.approx(0.0, abs=2e-3)  # ISO epoch is rounded to 1 ms
    assert log.scene["timeline"]["sites"][0]["name"] == "Boston"
    assert "timeline" not in VizLog.open(tmp_path / "e.vizlog").scene


def test_simulate_adds_the_coordinate_goal_as_a_site(tmp_path):
    goal = ADCS.goals.Coordinate_Goal(lat=42.36, lon=-71.09, alt=0)
    ADCS.simulate(**_short_run(goal=goal), viz_log=tmp_path / "g.vizlog")
    sites = VizLog.open(tmp_path / "g.vizlog").scene["timeline"]["sites"]
    assert [(s["name"], s["lat_deg"], s["lon_deg"]) for s in sites] == [("target", 42.36, -71.09)]
