"""Visualization stream: framing, playback, live broadcast/lockstep, launcher."""

import hashlib
import socket
import stat
import sys
import threading
import warnings
import zipfile

import numpy as np
import pytest

import ADCS
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.state import State
from ADCS.viz import VizClient, VizLog, VizLogWriter, VizServer, binary_manager
from ADCS.viz.launcher import view, viewer_command
from ADCS.viz.protocol import (
    HEADER,
    MsgType,
    ProtocolError,
    decode_block,
    decode_row,
    encode_block,
    encode_row,
    pack_frame,
    read_frame,
)
from ADCS.viz.schema import REQUIRED_CHANNELS, channel_spec

DT = 5.0
TF = 60.0
N = int(TF / DT)


# ---------------------------------------------------------------- framing
def test_frame_round_trip():
    a, b = socket.socketpair()
    with a, b:
        a.sendall(pack_frame(MsgType.ROW, b"payload", seq=42))
        assert read_frame(b) == (MsgType.ROW, 42, b"payload")


@pytest.mark.parametrize("mutate, match", [
    (lambda h: b"XXXX" + h[4:], "magic"),
    (lambda h: h[:4] + (2).to_bytes(2, "little") + h[6:], "version"),
    (lambda h: h[:6] + (99).to_bytes(2, "little") + h[8:], "unknown message type"),
])
def test_bad_frames_are_rejected(mutate, match):
    a, b = socket.socketpair()
    with a, b:
        frame = pack_frame(MsgType.ROW, b"")
        a.sendall(mutate(frame[:HEADER.size]))
        with pytest.raises(ProtocolError, match=match):
            read_frame(b)


def test_row_and_block_payloads_round_trip():
    row = np.arange(10.0)
    assert decode_row(encode_row("run-1", row))[0] == "run-1"
    np.testing.assert_array_equal(decode_row(encode_row("run-1", row))[1], row)
    cols = [np.arange(6.0).reshape(3, 2), np.arange(12.0).reshape(3, 4)]
    rid, start, out = decode_block(encode_block("7", 100, cols), [2, 4])
    assert (rid, start) == ("7", 100)
    for c, o in zip(cols, out):
        np.testing.assert_array_equal(c, o)


# ---------------------------------------------------------------- playback
def _small_log(path, runs=("0", "1"), n=9):
    rng = np.random.default_rng(3)
    channels = [channel_spec(c) for c in REQUIRED_CHANNELS]
    data = {}
    with VizLogWriter(path, channels, epoch_jd_tt=2459580.5, scene={"spacecraft": {"mass_kg": 2.0}}) as w:
        for rid in runs:
            data[rid] = {c: rng.standard_normal((n, s.width)) for c, s in REQUIRED_CHANNELS.items()}
            w.write_run(rid, data[rid])
    return data


def test_playback_delivers_the_log_to_every_viewer(tmp_path):
    data = _small_log(tmp_path / "a.vizlog")
    with VizServer() as server:
        server.serve_log(VizLog.open(tmp_path / "a.vizlog"))
        for _ in range(2):  # a second viewer gets the full log too
            with VizClient(*server.address, server.token) as client:
                hello = client.connect()
                assert hello["mode"] == "playback" and hello["scene"]["spacecraft"]["mass_kg"] == 2.0
                got = client.read_all(timeout=10)
            assert set(got) == {"0", "1"}
            for rid in got:
                for name, arr in data[rid].items():
                    np.testing.assert_array_equal(got[rid][name], arr)


def test_wrong_token_is_rejected(tmp_path):
    _small_log(tmp_path / "a.vizlog")
    with VizServer() as server:
        server.serve_log(VizLog.open(tmp_path / "a.vizlog"))
        with VizClient(*server.address, "not-the-token") as client, pytest.raises(ConnectionRefusedError):
            client.connect()


# ---------------------------------------------------------------- live
def _sim_args():
    sat = ADCS.Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]),
                         actuators=[ADCS.RW(axis=a, max_torque=1.0, J=0.22, h=0, h_max=3.8) for a in np.eye(3)],
                         sensors=[ADCS.MTM(axis=a) for a in np.eye(3)])
    os0 = Orbital_State(ephem=Ephemeris(), J2000=0.22, R=np.array([5000.0, 0, 5000.0]), V=np.array([0, 7.5, 0]))
    x0 = State.from_array(np.array([0.01, 0.05, 0, 1, 0, 0, 0, 0.1, -0.2, 0.0]))
    return dict(x=x0, satellite=sat, os0=os0, dt=DT, tf=TF)


def _collect_in_thread(server, out, ack_limit=None):
    def run():
        with VizClient(*server.address, server.token) as client:
            out["hello"] = client.connect()
            rows = []
            for kind, data in client.messages(timeout=30):
                if kind == MsgType.ROW:
                    rows.append(data[1])
                    if ack_limit is not None and len(rows) >= ack_limit:
                        return out.setdefault("rows", rows)  # disconnect mid-run
            out["rows"] = rows
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def test_live_broadcast_matches_the_post_run_log(tmp_path):
    out = {}
    with VizServer(mode="broadcast") as server:
        t = _collect_in_thread(server, out)
        res = ADCS.simulate(**_sim_args(), viz_stream=server, viz_log=tmp_path / "post.vizlog")
        t.join(timeout=30)
    assert out["hello"]["mode"] == "live-broadcast"
    rows = out["rows"]
    assert len(rows) == N == len(res.runs[0].state_hist)
    log = VizLog.open(tmp_path / "post.vizlog")
    for name in ["t_s", "r_sc", "v_sc", "q_gcrs_body", "w_body", "r_sun", "q_itrs_gcrs",
                 "shadow_factor", "r_moon", "sunlit_binary", "h_wheel", "wheel_speed", "u_cmd"]:
        streamed = np.array([r[name] for r in rows])
        np.testing.assert_allclose(streamed, log.read("0", name), rtol=1e-12, atol=1e-15, err_msg=name)


def test_live_lockstep_delivers_every_row_in_order(tmp_path):
    out = {}
    with VizServer(mode="lockstep") as server:
        t = _collect_in_thread(server, out)
        ADCS.simulate(**_sim_args(), viz_stream=server)
        t.join(timeout=30)
    assert out["hello"]["mode"] == "live-lockstep"
    np.testing.assert_allclose([r["t_s"][0] for r in out["rows"]], DT * np.arange(N))


def test_lockstep_survives_the_viewer_leaving():
    out = {}
    with VizServer(mode="lockstep", ack_timeout=2.0) as server:
        t = _collect_in_thread(server, out, ack_limit=3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = ADCS.simulate(**_sim_args(), viz_stream=server)
        t.join(timeout=30)
    assert len(res.runs[0].state_hist) == N
    assert any("streaming stopped" in str(w.message) for w in caught)


def test_broadcast_buffers_until_a_viewer_connects_and_drops_the_oldest():
    with VizServer(mode="broadcast", queue_size=5) as server:
        server.begin_live({"channels": [{"name": "t_s", "width": 1}], "epoch": {}}, {})
        for k in range(8):
            server.publish_row("0", np.array([float(k)]))
        server.end_run("0")
        server.end_session()
        assert server.dropped_rows == 5  # 8 rows + 2 END frames into a queue of 5
        with VizClient(*server.address, server.token) as client:
            client.connect()
            got = client.read_all(timeout=10)
    np.testing.assert_array_equal(got["0"]["t_s"][:, 0], [5.0, 6.0, 7.0])


def test_live_tee_writes_the_streamed_rows(tmp_path):
    from ADCS.viz.live import LiveRecorder

    args = _sim_args()
    with VizServer() as server:
        rec = LiveRecorder(server, args["satellite"], args["os0"].ephem, args["os0"].J2000, DT, 3,
                           tee_path=tmp_path / "tee.vizlog")
        os_k = args["os0"]
        for k in range(3):
            rec.step(k, args["x"], os_k, np.zeros(3))
        rec.close()
    log = VizLog.open(tmp_path / "tee.vizlog")
    assert log.n_rows("0") == 3 and log.is_complete("0")
    np.testing.assert_allclose(log.read("0", "t_s")[:, 0], [0.0, DT, 2 * DT])


# ---------------------------------------------------------------- launcher
FAKE_VIEWER = """#!{python}
# Stand-in viewer: connect as instructed, read the whole session, report rows.
import sys
from ADCS.viz import VizClient
args = sys.argv[sys.argv.index("--") + 1:]
host, port = args[args.index("--connect") + 1].rsplit(":", 1)
token = args[args.index("--token") + 1]
with VizClient(host, int(port), token) as c:
    c.connect()
    data = c.read_all(timeout=20)
open({out!r}, "w").write(str(sum(len(cols["t_s"]) for cols in data.values())))
"""


def test_view_launches_the_viewer_against_the_served_log(tmp_path):
    _small_log(tmp_path / "a.vizlog", runs=("0",), n=11)
    report = tmp_path / "rows.txt"
    fake = tmp_path / "fake_viewer"
    fake.write_text(FAKE_VIEWER.format(python=sys.executable, out=str(report)))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    assert view(tmp_path / "a.vizlog", binary=fake) == 0
    assert report.read_text() == "11"


def test_viewer_command_passes_user_args_after_double_dash():
    cmd = viewer_command("viewer.exe", "127.0.0.1", 5000, "tok", ["--fullscreen"])
    assert cmd == ["viewer.exe", "--", "--connect", "127.0.0.1:5000", "--token", "tok", "--fullscreen"]


def test_find_viewer_order(tmp_path, monkeypatch):
    monkeypatch.delenv("ADCS_VIEWER_BIN", raising=False)
    monkeypatch.setenv("ADCS_VIEWER_CACHE", str(tmp_path / "cache"))
    with pytest.raises(binary_manager.ViewerNotFoundError, match="no viewer release"):
        binary_manager.find_viewer()
    exe = tmp_path / "viewer"
    exe.write_text("")
    monkeypatch.setenv("ADCS_VIEWER_BIN", str(exe))
    assert binary_manager.find_viewer() == exe
    other = tmp_path / "other"
    other.write_text("")
    assert binary_manager.find_viewer(other) == other


def _release(tmp_path, members):
    archive = tmp_path / "viewer.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return {"version": "9.9.9", "assets": {"test-x86_64": {
        "url": archive.as_uri(), "sha256": digest, "executable": "adcs-viewer"}}}


def test_install_viewer_verifies_and_unpacks(tmp_path, monkeypatch):
    monkeypatch.setenv("ADCS_VIEWER_CACHE", str(tmp_path / "cache"))
    manifest = _release(tmp_path, {"adcs-viewer": "#!/bin/sh\n", "data.pck": "x"})
    exe = binary_manager.install_viewer("test-x86_64", manifest)
    assert exe == tmp_path / "cache" / "9.9.9" / "test-x86_64" / "adcs-viewer"
    assert exe.stat().st_mode & stat.S_IEXEC


def test_install_viewer_rejects_a_bad_checksum(tmp_path, monkeypatch):
    monkeypatch.setenv("ADCS_VIEWER_CACHE", str(tmp_path / "cache"))
    manifest = _release(tmp_path, {"adcs-viewer": "x"})
    manifest["assets"]["test-x86_64"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        binary_manager.install_viewer("test-x86_64", manifest)
    assert not (tmp_path / "cache" / "9.9.9").exists()


def test_install_viewer_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setenv("ADCS_VIEWER_CACHE", str(tmp_path / "cache"))
    manifest = _release(tmp_path, {"../escape": "x", "adcs-viewer": "x"})
    with pytest.raises(ValueError, match="escapes"):
        binary_manager.install_viewer("test-x86_64", manifest)
    assert not (tmp_path / "escape").exists()
