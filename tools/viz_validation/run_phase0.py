"""Phase 0-V: run the Godot prototype's end-to-end tests and write PHASE0_results.md.

Writes the measured table to PHASE0_results.md (PHASE0.md holds the
hand-written findings and decision). Each test serves data through the real stream protocol (ADCS.viz.VizServer),
launches the prototype viewer (viewer_proto/) with Godot, collects the
viewer's raw measurements and judges them against values computed here in
Python, independently of the viewer.

Usage::

    python tools/viz_validation/run_phase0.py                 # all tests
    python tools/viz_validation/run_phase0.py jitter perf     # a subset
    python tools/viz_validation/run_phase0.py --godot PATH    # explicit Godot binary

Godot is found via --godot, $ADCS_GODOT_BIN, or the Windows build installed
under %LOCALAPPDATA%/generalized_adcs/tools/godot-4.7.2 (from WSL as well).
Under WSL the project is copied to the Windows side and Godot runs natively
on the Windows GPU, talking to this script over 127.0.0.1.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ADCS.helpers.math_helpers import rot_mat  # noqa: E402
from ADCS.orbits.ephemeris import Ephemeris  # noqa: E402
from ADCS.orbits.shadow import shadow_factor  # noqa: E402
from ADCS.orbits.universal_constants import EarthConstants  # noqa: E402
from ADCS.viz import VizLog, VizLogWriter, VizServer  # noqa: E402
from ADCS.viz.binary_manager import is_wsl  # noqa: E402
from ADCS.viz.derive import derive_environment  # noqa: E402
from ADCS.viz.interp import quat_continuity  # noqa: E402
from ADCS.viz.launcher import viewer_command  # noqa: E402
from ADCS.viz.schema import REQUIRED_CHANNELS, channel_spec  # noqa: E402
from ADCS.viz.writer import make_manifest  # noqa: E402

PROTO = REPO / "viewer_proto"
HERE = Path(__file__).resolve().parent
EPOCH_JD_TT = 2459580.5  # 2022-01-01 00:00 TT, also a Horizons fixture epoch
ALL_TESTS = ["jitter", "jitter_control", "zoom", "terminator", "landmarks", "hdr",
             "capture", "transport_broadcast", "transport_lockstep", "perf_1080p", "perf_1440p"]
LANDMARKS = {"null_island": (0.0, 0.0), "greenwich": (51.4779, -0.0015),
             "cape_horn": (-55.98, -67.27), "honolulu": (21.31, -157.86)}


# ------------------------------------------------------------------ platform
def _local_appdata() -> Path:
    if is_wsl():
        win = subprocess.run(["cmd.exe", "/c", "echo %LOCALAPPDATA%"], capture_output=True,
                             text=True, check=True).stdout.strip()
        return Path(subprocess.run(["wslpath", "-u", win], capture_output=True, text=True,
                                   check=True).stdout.strip())
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share"))


def find_godot(explicit: str | None) -> Path:
    for c in (explicit, os.environ.get("ADCS_GODOT_BIN")):
        if c:
            return Path(c)
    default = _local_appdata() / "generalized_adcs" / "tools" / "godot-4.7.2" / "Godot_v4.7.2-stable_win64_console.exe"
    if default.exists():
        return default
    sys.exit("Godot not found: pass --godot or set ADCS_GODOT_BIN")


def host_path(p: Path, windows_binary: bool) -> str:
    """Path as the Godot process sees it."""
    if windows_binary and is_wsl():
        return subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True, check=True).stdout.strip()
    return str(p)


# ------------------------------------------------------------------ test data
def build_log(path: Path, seconds: int = 240) -> None:
    """Synthetic 7000 km, 51.6 deg circular orbit with real Sun/Moon/Earth orientation."""
    ephem = Ephemeris()
    t = np.arange(seconds + 1, dtype=float)
    jd = EPOCH_JD_TT + t / 86400.0
    env = derive_environment(ephem, jd)
    a = 7000.0
    n = np.sqrt(EarthConstants.mu_e / a**3)
    inc = np.deg2rad(51.6)
    # Start the orbit on the sunlit side.
    u0 = np.arctan2(env["r_sun"][0, 1], env["r_sun"][0, 0])
    u = u0 + n * t
    r = a * np.column_stack([np.cos(u), np.sin(u) * np.cos(inc), np.sin(u) * np.sin(inc)])
    v = a * n * np.column_stack([-np.sin(u), np.cos(u) * np.cos(inc), np.cos(u) * np.sin(inc)])
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    ang = 0.02 * t
    q = quat_continuity(np.column_stack([np.cos(ang / 2), np.outer(np.sin(ang / 2), axis)]))
    cols = {"t_s": t, "r_sc": r, "v_sc": v, "q_gcrs_body": q, "w_body": np.tile(0.02 * axis, (len(t), 1)),
            "r_sun": env["r_sun"], "q_itrs_gcrs": env["q_itrs_gcrs"], "shadow_factor": shadow_factor(r, env["r_sun"]),
            "r_moon": env["r_moon"]}
    channels = [channel_spec(c) for c in REQUIRED_CHANNELS] + [channel_spec("r_moon")]
    with VizLogWriter(path, channels, epoch_jd_tt=EPOCH_JD_TT, scene={"spacecraft": {"mass_kg": 4.0}},
                      overwrite=True) as w:
        w.write_run(0, cols, dt_s=1.0)


def build_landmark_texture(path: Path, w: int = 4096, h: int = 2048) -> None:
    from PIL import Image

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[...] = (20, 40, 90)
    yy, xx = np.mgrid[0:h, 0:w]
    for lat, lon in LANDMARKS.values():
        px = (lon / 360.0 + 0.5) * w
        py = (0.5 - lat / 180.0) * h
        dx = np.minimum(np.abs(xx + 0.5 - px), w - np.abs(xx + 0.5 - px))
        img[(dx**2 + (yy + 0.5 - py) ** 2) <= 4.0**2] = (255, 0, 0)
    Image.fromarray(img).save(path)


def subsolar_expected(log: VizLog) -> dict:
    q = log.read("0", "q_itrs_gcrs")[0]
    s = rot_mat(q) @ log.read("0", "r_sun")[0]
    s /= np.linalg.norm(s)
    return {"lon_deg": float(np.degrees(np.arctan2(s[1], s[0]))), "lat_deg": float(np.degrees(np.arcsin(s[2])))}


def landmark_dirs(log: VizLog) -> list:
    R = rot_mat(log.read("0", "q_itrs_gcrs")[0])  # v_itrs = R v_gcrs
    out = []
    for name, (lat, lon) in LANDMARKS.items():
        la, lo = np.radians(lat), np.radians(lon)
        p_itrs = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
        out.append({"name": name, "lat": lat, "lon": lon, "dir_gcrs": (R.T @ p_itrs).tolist()})
    return out


# ------------------------------------------------------------------ running
def viewer_test(name: str) -> str:
    """Test name the viewer understands (variants share one implementation)."""
    for base in ("jitter", "perf", "capture", "transport"):
        if name.startswith(base):
            return base
    return name


class Runner:
    def __init__(self, godot: Path, out_root: Path, timeout: float) -> None:
        self.godot = godot
        self.windows = godot.suffix.lower() == ".exe"
        self.timeout = timeout
        self.out_root = out_root
        if self.windows and is_wsl():
            # Godot gets its own copy on the Windows filesystem (fast, no UNC paths).
            self.run_root = _local_appdata() / "generalized_adcs" / "proto_run"
            if (self.run_root / "viewer_proto").exists():
                shutil.rmtree(self.run_root / "viewer_proto")
            shutil.copytree(PROTO, self.run_root / "viewer_proto", ignore=shutil.ignore_patterns(".godot"))
            self.project = self.run_root / "viewer_proto"
            self.work = self.run_root / "work"
        else:
            self.project = PROTO
            self.work = out_root / "work"
        if self.work.exists():
            shutil.rmtree(self.work)
        self.work.mkdir(parents=True)
        self.log_path = self.work / "phase0.vizlog"
        build_log(self.log_path)
        self.texture = self.work / "landmarks.png"
        build_landmark_texture(self.texture)
        self.log = VizLog.open(self.log_path)
        self.config = {
            "jitter": {"frames": 600},
            "zoom": {},
            "landmarks": {"texture": host_path(self.texture, self.windows), "points": landmark_dirs(self.log)},
            "perf": {"frames": 600, "model_triangles": 100000},
            "capture": {"frames": 120},
            "transport": {"dt": 0.001},
        }

    def _launch(self, server: VizServer, test: str, cfg: dict, res: str = "1280x720",
                engine_args: tuple = ()) -> tuple:
        out = self.work / test
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        cfg_path = out / "config.json"
        cfg_path.write_text(json.dumps(cfg))
        h, p = server.address
        cmd = viewer_command(self.godot, h, p, server.token,
                             ["--test", viewer_test(test),
                              "--config", host_path(cfg_path, self.windows),
                              "--out", host_path(out, self.windows), "--res", res])
        cmd = [cmd[0], "--path", host_path(self.project, self.windows), *engine_args, *cmd[1:]]
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            output, _ = proc.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            # A script that fails to compile leaves Godot idling in an empty window.
            proc.kill()
            output, _ = proc.communicate()
            output = f"[timed out after {self.timeout:.0f} s]\n" + output
        elapsed = time.perf_counter() - t0
        (out / "godot_output.txt").write_text(output)
        result_file = out / "result.json"
        result = json.loads(result_file.read_text()) if result_file.exists() else {"error": "no result.json"}
        result["_returncode"] = proc.returncode
        result["_elapsed_s"] = elapsed
        if proc.returncode != 0 or "error" in result:
            result.setdefault("error", f"godot exited {proc.returncode}")
            errors = [ln for ln in output.splitlines() if "ERROR" in ln or "Error" in ln]
            result["_godot_errors"] = errors[:20]
        return result, out

    def playback(self, test: str, res: str = "1280x720", engine_args: tuple = (), cfg: dict | None = None) -> tuple:
        with VizServer() as server:
            server.serve_log(self.log)
            return self._launch(server, test, cfg if cfg is not None else self.config, res, engine_args)


# ------------------------------------------------------------------ judging
def judge_jitter(r: dict, control: bool) -> dict:
    cx, cy = np.array(r["cx"]), np.array(r["cy"])
    lost = int(r["lost"])
    ok = (cx >= 0) & (cy >= 0)
    if ok.sum() < 2:
        rms = mx = float("inf")
    else:
        d = np.hypot(cx[ok] - cx[ok].mean(), cy[ok] - cy[ok].mean())
        rms, mx = float(np.sqrt(np.mean(d**2))), float(d.max())
    if control:
        passed = lost > 0 or mx > 2.0
        crit = "control must show > 2 px (or lose the fiducial)"
    else:
        passed = lost == 0 and rms < 0.1 and mx < 0.25
        crit = "RMS < 0.1 px, max < 0.25 px, never lost"
    return {"passed": passed, "criterion": crit, "rms_px": rms, "max_px": mx, "lost_frames": lost,
            "frames": r["frames"]}


def judge_zoom(r: dict) -> dict:
    bad_prec = [p for p in r["precision"] if p["front"] != p["seen"]]
    bad_occ = [o for o in r["occlusion"] if o["expected"] != o["seen"]]
    limit = max((p["far_over_near"] for p in r["ratio_probe"] if p["visible"]), default=None)
    return {"passed": not bad_prec and not bad_occ,
            "criterion": "depth order correct at 1 m..4e8 m within a 1e6 frustum (sep 1e-5 x D) "
                         "and across the scaled-space split",
            "single_frustum_max_ratio": limit, "ratio_probe": r["ratio_probe"],
            "precision": r["precision"], "occlusion": r["occlusion"], "snapshots": r["snapshots"]}


def judge_terminator(r: dict, expected: dict) -> dict:
    dlon = (r["subsolar"]["lon_deg"] - expected["lon_deg"] + 180.0) % 360.0 - 180.0
    dlat = r["subsolar"]["lat_deg"] - expected["lat_deg"]
    num_err = float(np.hypot(dlon * np.cos(np.radians(expected["lat_deg"])), dlat))
    off = np.array(r["offsets_px"])
    img_err_deg = float(np.degrees(np.abs(off.mean()) / r["disk_radius_px"])) if len(off) else float("inf")
    passed = num_err < 0.005 and img_err_deg < 0.1 and r["basis_err_deg"] < 0.005
    return {"passed": passed, "criterion": "numeric < 0.005 deg; image < 0.1 deg; float32 basis < 0.005 deg",
            "numeric_err_deg": num_err, "image_err_deg": img_err_deg, "image_offset_px": float(off.mean()) if len(off) else None,
            "basis_err_deg": r["basis_err_deg"], "viewer_subsolar": r["subsolar"], "python_subsolar": expected}


def judge_landmarks(r: dict) -> dict:
    rows, worst = [], 0.0
    km_per_px = r["altitude_m"] / r["focal_px"] / 1000.0
    for p in r["points"]:
        err = float(np.hypot(p["cx"] - p["center"][0], p["cy"] - p["center"][1])) if p["cx"] >= 0 else float("inf")
        worst = max(worst, err)
        rows.append({"name": p["name"], "err_px": err, "err_km": err * km_per_px})
    return {"passed": worst < 2.0, "criterion": "landmark at screen centre within 2 px", "points": rows}


def judge_hdr(r: dict) -> dict:
    cases = []
    for c in r["cases"]:
        rel = abs(c["seen_linear"] - c["expected_linear"]) / c["expected_linear"]
        cases.append({**c, "rel_err": rel, "ok": rel < 0.1})
    return {"passed": all(c["ok"] for c in cases), "criterion": "values > 1 survive to tonemapping (within 10%)",
            "cases": cases}


def judge_perf(r: dict) -> dict:
    gpu, frame = np.array(r["gpu_ms"]), np.array(r["frame_ms"])
    stats = {"gpu_p50_ms": float(np.percentile(gpu, 50)), "gpu_p99_ms": float(np.percentile(gpu, 99)),
             "frame_p50_ms": float(np.percentile(frame, 50)), "frame_p99_ms": float(np.percentile(frame, 99))}
    return {"passed": None, "criterion": "RTX 3060 target: GPU p50 < 4 ms, p99 < 8.3 ms (recorded only on other GPUs)",
            "res": r["res"], **stats, "within_reference_budget": stats["gpu_p50_ms"] < 4 and stats["gpu_p99_ms"] < 8.3}


def run_capture(runner: Runner) -> dict:
    hashes = []
    for k in range(2):
        movie_dir = runner.work / f"capture_movie_{k}"
        if movie_dir.exists():
            shutil.rmtree(movie_dir)
        movie_dir.mkdir(parents=True)
        res, _ = runner.playback(f"capture_{k}", engine_args=(
            "--write-movie", host_path(movie_dir / "frame.png", runner.windows), "--fixed-fps", "60"),
            cfg=runner.config)
        if "error" in res:
            return {"passed": False, "error": res["error"], "godot_errors": res.get("_godot_errors", [])}
        frames = sorted(movie_dir.glob("frame*.png"))
        hashes.append([hashlib.sha256(f.read_bytes()).hexdigest() for f in frames])
    same = hashes[0] == hashes[1]
    return {"passed": same and len(hashes[0]) >= runner.config["capture"]["frames"],
            "criterion": "two --write-movie runs produce bit-identical frames",
            "frames_run0": len(hashes[0]), "frames_run1": len(hashes[1]),
            "identical": same, "first_mismatch": next((i for i, (a, b) in enumerate(zip(*hashes)) if a != b), None)}


def transport_manifest() -> dict:
    from ADCS.viz.schema import Channel
    chans = [Channel("t_s", 1, "s", "-"), Channel("pad", 63, "-", "-")]  # 64 float64 = 512 B rows
    return make_manifest(chans, EPOCH_JD_TT)


def run_transport_broadcast(runner: Runner, rate_hz: float = 1000.0, seconds: float = 10.0) -> dict:
    n = int(rate_hz * seconds)
    with VizServer(mode="broadcast", queue_size=n + 10) as server:
        manifest = transport_manifest()
        manifest["runs"] = [{"id": "0", "n_rows": n, "dt_s": 1.0 / rate_hz}]
        server.begin_live(manifest, {})
        result_box = {}
        th = threading.Thread(target=lambda: result_box.update(r=runner._launch(server, "transport_broadcast",
                                                                                runner.config)[0]), daemon=True)
        th.start()
        server.wait_for_client(30)
        row = np.zeros(64)
        t0 = time.perf_counter()
        for k in range(n):
            target = t0 + k / rate_hz
            while time.perf_counter() < target:
                pass
            row[0] = k / rate_hz
            server.publish_row("0", row)
        send_s = time.perf_counter() - t0
        server.end_run("0")
        server.end_session()
        th.join(runner.timeout)
        dropped = server.dropped_rows
    r = result_box.get("r", {"error": "viewer did not finish"})
    if "error" in r:
        return {"passed": False, "error": r["error"], "godot_errors": r.get("_godot_errors", [])}
    return {"passed": r["rows"] == n and r["gaps"] == 0 and dropped == 0,
            "criterion": f"{n} rows of 512 B at {rate_hz:.0f} Hz: all received, no gaps, none dropped",
            "sent": n, "received": r["rows"], "gaps": r["gaps"], "dropped": dropped, "send_duration_s": send_s}


def run_transport_lockstep(runner: Runner, rows: int = 2000) -> dict:
    with VizServer(mode="lockstep") as server:
        manifest = transport_manifest()
        manifest["runs"] = [{"id": "0", "n_rows": rows, "dt_s": 0.001}]
        server.begin_live(manifest, {})
        result_box = {}
        th = threading.Thread(target=lambda: result_box.update(r=runner._launch(server, "transport_lockstep",
                                                                                runner.config)[0]), daemon=True)
        th.start()
        if not server.wait_for_client(60):
            return {"passed": False, "error": "viewer never connected"}
        row = np.zeros(64)
        rtt = []
        for k in range(rows):
            row[0] = k * 0.001
            t0 = time.perf_counter()
            server.publish_row("0", row)
            rtt.append((time.perf_counter() - t0) * 1000.0)
        server.end_run("0")
        server.end_session()
        th.join(runner.timeout)
    rtt = np.array(rtt[10:])
    r = result_box.get("r", {})
    p99 = float(np.percentile(rtt, 99))
    # The plan's criterion is one-way latency; Windows and WSL clocks are not
    # shared, so it is bounded by half the measured round trip.
    return {"passed": p99 / 2.0 < 5.0 and r.get("rows") == rows,
            "criterion": "all rows delivered; one-way latency p99 < 5 ms, estimated as round trip / 2",
            "one_way_p99_ms_est": p99 / 2.0,
            "rtt_p50_ms": float(np.percentile(rtt, 50)), "rtt_p99_ms": p99, "rtt_max_ms": float(rtt.max()),
            "received": r.get("rows")}


# ------------------------------------------------------------------ report
def write_report(results: dict, meta: dict, path: Path) -> None:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    lines = ["# Phase 0-V prototype results", "",
             f"Generated {meta['date']} by `tools/viz_validation/run_phase0.py`.", "",
             f"- GPU: **{meta.get('adapter')}** ({meta.get('api')}), Godot {meta.get('godot')}",
             f"- Host: {meta['host']}", f"- Viewer process: {meta['viewer_os']}", "",
             "| Test | Result | Key numbers | Criterion |", "|---|---|---|---|"]
    keys = {
        "jitter": ["rms_px", "max_px", "lost_frames"], "jitter_control": ["rms_px", "max_px", "lost_frames"],
        "zoom": [], "terminator": ["numeric_err_deg", "image_err_deg", "image_offset_px", "basis_err_deg"],
        "landmarks": [], "hdr": [], "capture": ["frames_run0", "identical"],
        "transport_broadcast": ["received", "gaps", "dropped"],
        "transport_lockstep": ["received", "rtt_p50_ms", "rtt_p99_ms", "one_way_p99_ms_est", "rtt_max_ms"],
        "perf_1080p": ["gpu_p50_ms", "gpu_p99_ms", "frame_p50_ms"], "perf_1440p": ["gpu_p50_ms", "gpu_p99_ms", "frame_p50_ms"],
    }
    for name, r in results.items():
        status = "ERROR" if "error" in r else ("recorded" if r.get("passed") is None else ("PASS" if r["passed"] else "FAIL"))
        nums = ", ".join(f"{k}={fmt(r.get(k))}" for k in keys.get(name, []) if k in r)
        if name == "landmarks" and "points" in r:
            nums = ", ".join(f"{p['name']} {p['err_px']:.2f} px" for p in r["points"])
        if name == "zoom" and "precision" in r:
            nums = (f"precision {sum(p['front'] == p['seen'] for p in r['precision'])}/{len(r['precision'])}, "
                    f"occlusion {sum(o['expected'] == o['seen'] for o in r['occlusion'])}/{len(r['occlusion'])}, "
                    f"single frustum works up to far/near = {r['single_frustum_max_ratio']:.0e}")
        if name == "hdr" and "cases" in r:
            nums = ", ".join(f"{c['name']} {c['seen_linear']:.3f}/{c['expected_linear']:.3f}" for c in r["cases"])
        if "error" in r:
            nums = str(r["error"])[:120]
        lines.append(f"| {name} | {status} | {nums} | {r.get('criterion', '')} |")
    lines += ["", "Raw measurements: `tools/viz_validation/results/latest.json`.", ""]
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tests", nargs="*", default=ALL_TESTS)
    ap.add_argument("--godot")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--near", type=float, help="override camera near plane [m]")
    ap.add_argument("--far", type=float, help="override camera far plane [m]")
    args = ap.parse_args()

    godot = find_godot(args.godot)
    out_root = HERE / "results"
    out_root.mkdir(exist_ok=True)
    runner = Runner(godot, out_root, args.timeout)
    expected_sub = subsolar_expected(runner.log)
    if args.near or args.far:
        runner.config["camera"] = {k: v for k, v in (("near", args.near), ("far", args.far)) if v}
    results, meta = {}, {}
    for test in args.tests:
        print(f"[phase0] {test} ...", flush=True)
        if test in ("jitter", "jitter_control"):
            cfg = json.loads(json.dumps(runner.config))
            cfg["jitter"]["absolute"] = test == "jitter_control"
            r, _ = runner.playback(test, cfg=cfg)
            out = r if "error" in r else judge_jitter(r, test == "jitter_control")
        elif test == "zoom":
            r, _ = runner.playback("zoom")
            out = r if "error" in r else judge_zoom(r)
        elif test == "terminator":
            r, _ = runner.playback("terminator")
            out = r if "error" in r else judge_terminator(r, expected_sub)
        elif test == "landmarks":
            r, _ = runner.playback("landmarks")
            out = r if "error" in r else judge_landmarks(r)
        elif test == "hdr":
            r, _ = runner.playback("hdr")
            out = r if "error" in r else judge_hdr(r)
        elif test.startswith("perf"):
            res = {"perf_1080p": "1920x1080", "perf_1440p": "2560x1440"}[test]
            r, _ = runner.playback(test, res=res)
            out = r if "error" in r else judge_perf(r)
        elif test == "capture":
            r, out = {}, run_capture(runner)
        elif test == "transport_broadcast":
            r, out = {}, run_transport_broadcast(runner)
        elif test == "transport_lockstep":
            r, out = {}, run_transport_lockstep(runner)
        else:
            print(f"unknown test {test}")
            continue
        for k in ("adapter", "api", "godot"):
            if k in r:
                meta[k] = r[k]
        results[test] = out
        print(f"[phase0] {test}: {'ERROR ' + str(out.get('error'))[:200] if 'error' in out else ('PASS' if out.get('passed') else ('recorded' if out.get('passed') is None else 'FAIL'))}", flush=True)

    meta.update({"date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                 "host": f"{platform.system()} {platform.release()} ({'WSL2' if is_wsl() else 'native'})",
                 "viewer_os": "Windows (native, launched from WSL)" if runner.windows and is_wsl() else platform.system()})
    (out_root / "latest.json").write_text(json.dumps({"meta": meta, "results": results}, indent=1, default=str))
    write_report(results, meta, HERE / "PHASE0_results.md")
    print(f"[phase0] report: {HERE / 'PHASE0_results.md'}")
    return 0 if all(r.get("passed") is not False and "error" not in r for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
