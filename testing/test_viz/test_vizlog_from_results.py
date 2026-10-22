"""Simulation results -> adcs-vizlog: hooks, converter and channel contents."""

import numpy as np
import pytest

import ADCS
from ADCS.helpers.math_helpers import normalize, rot_mat
from ADCS.mc.simulate_mc import simulate_mc
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.shadow import shadow_factor
from ADCS.state import State
from ADCS.viz import VizLog
from ADCS.viz.__main__ import main as viz_cli
from ADCS.viz.schema import REQUIRED_CHANNELS, TARGET_KIND

DT = 5.0
TF = 60.0


def _satellite(star_tracker=False):
    mtqs = [ADCS.MTQ(axis=a, max_torque=0.1) for a in np.eye(3)]
    rws = [ADCS.RW(axis=a, max_torque=4.51, J=0.22, h=1, h_max=3.8) for a in np.eye(3)]
    sensors = [ADCS.MTM(axis=a) for a in np.eye(3)]
    if star_tracker:
        # Multi-output sensor, to exercise the sensor_hist slicing.
        sensors.append(ADCS.StarTracker(boresight=np.array([1.0, 0, 0])))
    return ADCS.Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]), actuators=mtqs + rws,
                          sensors=sensors, boresight=np.array([0, 0, 1]))


def _x0():
    return State.from_array(np.array([0.01, 0.05, 0, 1, 0, 0, 0, 0.1, -0.2, 0.0]))


def _os0():
    return Orbital_State(ephem=Ephemeris(), J2000=0.22, R=np.array([5000.0, 0, 5000.0]), V=np.array([0, 7.5, 0]))


@pytest.fixture(scope="module")
def controlled_run(tmp_path_factory):
    path = tmp_path_factory.mktemp("viz") / "run.vizlog"
    sat = _satellite()
    ctrl = ADCS.controller.MTQ_w_RW(est_sat=sat, p_gain=0.1, d_gain=0.7, c_gain=0.1, h_target=np.zeros(3))
    goal = ADCS.goals.Coordinate_Goal(lat=33.75, lon=-84.3885, alt=0)
    res = ADCS.simulate(x=_x0(), satellite=sat, controller=ctrl, goal=goal, os0=_os0(),
                        dt=DT, tf=TF, viz_log=path)
    return res, VizLog.open(path)


@pytest.fixture(scope="module")
def tracker_run(tmp_path_factory):
    # No controller: MTQ_w_RW only handles magnetometer readings.
    path = tmp_path_factory.mktemp("viz") / "tracker.vizlog"
    res = ADCS.simulate(x=_x0(), satellite=_satellite(star_tracker=True), os0=_os0(),
                        dt=DT, tf=TF, viz_log=path)
    return res, VizLog.open(path)


def test_required_and_expected_channels_present(controlled_run):
    _, log = controlled_run
    for name in REQUIRED_CHANNELS:
        assert name in log
    for name in ["r_moon", "sunlit_binary", "h_wheel", "wheel_speed", "u_cmd", "tau_act_body",
                 "tau_dist_body", "target_kind", "pointing_err", "sensor_mtm0", "sensor_rw0_h"]:
        assert name in log, name
    assert log.channels["tau_act_body"].provenance.startswith("nominal_recompute")
    assert log.channels["u_cmd"].labels == ["mtq0", "mtq1", "mtq2", "rw0", "rw1", "rw2"]


def test_state_channels_match_the_simulation(controlled_run):
    res, log = controlled_run
    run = res.runs[0]
    n = len(run.state_hist)
    assert log.n_rows("0") == n
    np.testing.assert_allclose(log.read("0", "t_s")[:, 0], DT * np.arange(n), atol=1e-6)
    np.testing.assert_array_equal(log.read("0", "r_sc"), np.array([o.R for o in run.os_hist]))
    q = log.read("0", "q_gcrs_body")
    for qk, s in zip(q, run.state_hist):
        assert min(np.abs(qk - s.q).max(), np.abs(qk + s.q).max()) == 0.0
    assert np.all(np.einsum("ij,ij->i", q[1:], q[:-1]) > 0)
    np.testing.assert_allclose(log.read("0", "r_sun"), np.array([o.S for o in run.os_hist]), rtol=1e-12)


def test_derived_channels_are_consistent(controlled_run):
    res, log = controlled_run
    run = res.runs[0]
    nu = log.read("0", "shadow_factor")[:, 0]
    np.testing.assert_allclose(nu, shadow_factor(log.read("0", "r_sc"), log.read("0", "r_sun")))
    sunlit = log.read("0", "sunlit_binary")[:, 0]
    assert np.all(sunlit[nu == 1.0] == 1.0) and np.all(sunlit[nu == 0.0] == 0.0)
    h = log.read("0", "h_wheel")
    np.testing.assert_allclose(h, np.array([s.h for s in run.state_hist]))
    np.testing.assert_allclose(log.read("0", "wheel_speed"), h / 0.22)
    # Earth orientation: the stored quaternion reproduces the simulation's ECI->ECEF matrix.
    for qe, o in zip(log.read("0", "q_itrs_gcrs"), run.os_hist):
        np.testing.assert_allclose(rot_mat(qe), o._R_eci2ecef, atol=1e-12)


def test_sensor_channels_partition_the_reading_vector(tracker_run):
    res, log = tracker_run
    y = np.array(res.runs[0].sensor_hist)
    parts = [log.read("0", f"sensor_{sid}") for sid in
             ["mtm0", "mtm1", "mtm2", "startracker0", "rw0_h", "rw1_h", "rw2_h"]]
    np.testing.assert_array_equal(np.hstack(parts), y)


def test_pointing_error_matches_the_goal(controlled_run):
    res, log = controlled_run
    run = res.runs[0]
    kind = log.read("0", "target_kind")[:, 0]
    assert np.all(kind == TARGET_KIND["direction"])
    err = log.read("0", "pointing_err")[:, 0]
    for k in (0, len(err) - 1):
        b_eci = rot_mat(run.state_hist[k].q) @ run.boresight_hist[k]
        target = np.asarray(run.target_hist[k])[1:]
        expected = np.arccos(np.clip(normalize(b_eci) @ normalize(target), -1, 1))
        assert err[k] == pytest.approx(expected, abs=1e-12)


def test_scene_describes_the_spacecraft(tracker_run):
    _, log = tracker_run
    scene = log.scene
    assert scene["spacecraft"]["mass_kg"] == 4.0
    assert [a["type"] for a in scene["actuators"]] == ["mtq"] * 3 + ["rw"] * 3
    tracker = next(s for s in scene["sensors"] if s["type"] == "StarTracker")
    assert tracker["fov_half_rad"] == pytest.approx(np.deg2rad(2.0))
    assert tracker["slice"] == [3, 6]  # after three 1-value magnetometers
    assert log.manifest["ephemeris"]["name"] == "de421"


def test_saved_results_convert_identically(controlled_run, tmp_path):
    res, log = controlled_run
    sim_file = res.save("viz", out_dir=tmp_path)
    out = tmp_path / "converted.vizlog"
    assert viz_cli(["convert", str(sim_file), str(out)]) == 0
    converted = VizLog.open(out)
    for name in ["r_sc", "q_gcrs_body", "r_sun", "q_itrs_gcrs", "shadow_factor", "u_cmd", "tau_act_body"]:
        np.testing.assert_allclose(converted.read("0", name), log.read("0", name), rtol=1e-12, atol=1e-15)


def test_simulate_mc_writes_one_vizlog_run_per_run(tmp_path):
    sat = ADCS.Satellite(J_0=np.eye(3), sensors=[ADCS.MTM(a) for a in np.eye(3)])
    x0 = State(w=np.array([0.02, -0.01, 0.015]), q=normalize(np.array([0.9, 0.2, -0.3, 0.1])))
    res = simulate_mc(x=x0, satellite=sat, os0=_os0(), dt=10.0, tf=30.0, num_runs=2, max_workers=1,
                      base_seed=0, viz_log=tmp_path / "mc.vizlog")
    log = VizLog.open(tmp_path / "mc.vizlog")
    assert sorted(log.run_ids) == sorted(str(r) for r in res.run_ids)
    for rid in log.run_ids:
        assert log.n_rows(rid) == 3 and log.is_complete(rid)
