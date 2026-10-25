"""Overlay descriptors (ADCS.viz.overlays) and how they reach the vizlog."""

import numpy as np
import pytest

import ADCS
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.state import State
from ADCS.viz import VizLog, results_to_vizlog
from ADCS.viz import overlays as ov

DT = 5.0
TF = 40.0

CHANNELS = {"r_sc": {"width": 3, "frame": "GCRS"}, "tau_act_body": {"width": 3, "frame": "BODY"},
            "h_rw_body": {"width": 3, "frame": "BODY"}, "b_field": {"width": 3, "frame": "GCRS"},
            "h_wheel": {"width": 3, "frame": "-"}, "target": {"width": 3}, "target_kind": {"width": 1}}
SCENE = {"sensors": [{"id": "st0", "fov_half_rad": 0.2}, {"id": "sun0", "fov_half_rad": 1.57, "fov_assumed": True},
                     {"id": "mtm0"}],
         "boresights": {"main": [0, 0, 1]}}


def test_builders_serialise_with_their_type_and_without_unset_fields():
    d = ov.Arrow("tau_act_body", label="control torque").to_json()
    assert d["type"] == "arrow" and d["source"] == "tau_act_body" and d["scale"] == "log"
    assert "ref" not in d and "units_per_m" not in d and "id" not in d
    assert ov.OrbitTrack().to_json()["track"] == "orbit" and ov.GroundTrack().to_json()["type"] == "polyline"
    assert ov.Label("hello").to_json()["offset_px"] == (0.0, -40.0)


def test_validation_assigns_ids_and_accepts_dicts():
    out = ov.validate_overlays([ov.Triad("lvlh"), {"type": "arrow", "source": "@sun", "scale": "fixed_px", "id": "sun"},
                                ov.Arrow("@boresight:main"), ov.SensorCone("st0"), ov.TargetLine()], CHANNELS, SCENE)
    assert [d["id"] for d in out] == ["triad0", "sun", "arrow2", "cone3", "target_line4"]


@pytest.mark.parametrize("item, match", [
    (ov.Arrow("w_body"), "not a 3-wide channel"),
    (ov.Arrow("h_wheel_x"), "not a 3-wide channel"),
    (ov.Arrow("@moon"), "unknown derived vector"),
    (ov.Arrow("@boresight:aux"), "no boresight"),
    (ov.Arrow("@sun", scale="linear"), "scale must be"),
    (ov.Arrow("tau_act_body", scale="physical"), "units_per_m"),
    (ov.Triad("ecef"), "frame must be"),
    (ov.SensorCone("mtm0"), "no field of view"),
    ({"type": "polyline", "track": "moon"}, "track must be"),
    ({"type": "sparkle"}, "unknown overlay type"),
])
def test_validation_rejects_bad_descriptors(item, match):
    with pytest.raises(ValueError, match=match):
        ov.validate_overlays([item], CHANNELS, SCENE)


def test_target_line_needs_target_channels():
    with pytest.raises(ValueError, match="no target channels"):
        ov.validate_overlays([ov.TargetLine()], {"r_sc": {"width": 3}}, SCENE)


def test_default_set_follows_the_log_contents():
    items = ov.default_overlays(CHANNELS, SCENE, ref={"h_rw_body": 0.01})
    arrows = {o.source: o for o in items if isinstance(o, ov.Arrow)}
    assert set(arrows) == {"tau_act_body", "h_rw_body", "b_field", "@sun"}  # no disturbance channel in this log
    assert arrows["h_rw_body"].ref == 0.01 and arrows["tau_act_body"].ref is None
    assert arrows["b_field"].scale == "fixed_px"  # direction only: field magnitude says little on screen
    assert [o.sensor for o in items if isinstance(o, ov.SensorCone)] == ["st0"]  # assumed FOVs are not drawn
    assert sum(isinstance(o, ov.TargetLine) for o in items) == 1
    assert {o.track for o in items if isinstance(o, (ov.OrbitTrack, ov.GroundTrack))} == {"orbit", "ground"}
    ov.validate_overlays(items, CHANNELS, SCENE)  # the default set is always valid


# ---------------------------------------------------------------- through results_to_vizlog
@pytest.fixture(scope="module")
def run():
    sat = ADCS.Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]),
                         actuators=[ADCS.MTQ(axis=a, max_torque=0.1) for a in np.eye(3)]
                         + [ADCS.RW(axis=a, max_torque=1.0, J=0.22, h=0.1, h_max=3.8)
                            for a in (np.eye(3) + 0.2) / np.linalg.norm(np.eye(3) + 0.2, axis=1, keepdims=True)],
                         sensors=[ADCS.MTM(axis=a) for a in np.eye(3)], boresight=np.array([0, 0, 1]))
    ctrl = ADCS.controller.MTQ_w_RW(est_sat=sat, p_gain=0.1, d_gain=0.7, c_gain=0.1, h_target=np.zeros(3))
    os0 = Orbital_State(ephem=Ephemeris(), J2000=0.22, R=np.array([5000.0, 0, 5000.0]), V=np.array([0, 7.5, 0]))
    x0 = State.from_array(np.array([0.01, 0.05, 0, 1, 0, 0, 0, 0.1, -0.2, 0.0]))
    return ADCS.simulate(x=x0, satellite=sat, controller=ctrl, goal=ADCS.goals.Coordinate_Goal(lat=33.7, lon=-84.4, alt=0),
                         os0=os0, dt=DT, tf=TF)


def test_wheel_momentum_in_body_axes(run, tmp_path):
    log = VizLog.open(results_to_vizlog(run, tmp_path / "a.vizlog"))
    axes = np.vstack([rw.axis for rw in run.runs[0].satellite.rw_actuators])
    np.testing.assert_allclose(log.read("0", "h_rw_body"), log.read("0", "h_wheel") @ axes, rtol=1e-12)
    assert not np.allclose(axes, np.eye(3))  # skewed wheels: the projection matters


def test_default_overlays_are_written_with_reference_magnitudes(run, tmp_path):
    log = VizLog.open(results_to_vizlog(run, tmp_path / "b.vizlog"))
    items = log.scene["overlays"]
    assert all("id" in d for d in items) and len({d["id"] for d in items}) == len(items)
    h = next(d for d in items if d.get("source") == "h_rw_body")
    mags = np.linalg.norm(log.read("0", "h_rw_body"), axis=1)
    assert h["ref"] == pytest.approx(np.percentile(mags[mags > 0], 10))
    assert {d["type"] for d in items} >= {"triad", "arrow", "polyline"}


def test_overlays_can_be_chosen_or_left_out(run, tmp_path):
    log = VizLog.open(results_to_vizlog(run, tmp_path / "c.vizlog", overlays=None))
    assert "overlays" not in log.scene
    log = VizLog.open(results_to_vizlog(run, tmp_path / "d.vizlog", overlays=[ov.Arrow("@velocity", scale="fixed_px")]))
    assert [d["source"] for d in log.scene["overlays"]] == ["@velocity"]
    with pytest.raises(ValueError, match="not a 3-wide channel"):
        results_to_vizlog(run, tmp_path / "e.vizlog", overlays=[ov.Arrow("user_missing")])


def test_extra_channels_joint_and_user(run, tmp_path):
    n = len(run.runs[0].state_hist)
    angle = np.linspace(-1.0, 0.0, n)
    vec = np.tile([0.0, 0.0, 2.0], (n, 1))
    path = results_to_vizlog(run, tmp_path / "f.vizlog", extra_channels={"joint_array": angle, "user_thrust": vec},
                             overlays=[ov.Arrow("user_thrust", scale="physical", units_per_m=1.0)])
    log = VizLog.open(path)
    np.testing.assert_array_equal(log.read("0", "joint_array")[:, 0], angle)
    np.testing.assert_array_equal(log.read("0", "user_thrust"), vec)
    chans = {c["name"]: c for c in log.manifest["channels"]}
    assert chans["joint_array"]["unit"] == "rad" and chans["user_thrust"]["width"] == 3


@pytest.mark.parametrize("extra, err, match", [
    ({"joint_array": np.zeros(3)}, ValueError, "rows"),
    ({"joint_array": np.zeros((9, 2))}, ValueError, "width 1"),
    ({"thrust": np.zeros(9)}, KeyError, "unknown vizlog channel"),
])
def test_extra_channels_are_checked(run, tmp_path, extra, err, match):
    n = len(run.runs[0].state_hist)
    extra = {k: (np.zeros((n,) + v.shape[1:]) if v.shape[0] == 9 else v) for k, v in extra.items()}
    with pytest.raises(err, match=match):
        results_to_vizlog(run, tmp_path / "g.vizlog", extra_channels=extra)
