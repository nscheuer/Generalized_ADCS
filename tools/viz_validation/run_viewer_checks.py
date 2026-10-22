"""Phase 5: end-to-end checks of the real viewer (viewer/), written to VIEWER_CHECKS.md.

Runs the viewer exactly as ``python -m ADCS.viz view`` does (VizServer playback
over TCP, the same launcher command and project sync), on data from a real
``ADCS.simulate`` run with an imported spacecraft model, and judges the
viewer's measurements against values computed here in Python.

Usage::

    python tools/viz_validation/run_viewer_checks.py              # all checks
    python tools/viz_validation/run_viewer_checks.py jitter shots # a subset

Needs Godot (see run_phase0.py), the asset cache (python -m ADCS.viz.assets build)
and the viz-mesh extra.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase0 import (  # noqa: E402
    _local_appdata,
    build_landmark_texture,
    build_log,
    find_godot,
    judge_landmarks,
    landmark_dirs,
    subsolar_expected,
)

import ADCS  # noqa: E402
from ADCS.viz import VizLog, VizServer, results_to_vizlog  # noqa: E402
from ADCS.viz.binary_manager import is_wsl  # noqa: E402
from ADCS.viz.launcher import (  # noqa: E402
    _default_assets,
    viewer_command,
    viewer_project,
)

HERE = Path(__file__).resolve().parent
ALL = ["assets", "file_source", "shots", "jitter", "jitter_control", "terminator", "landmarks", "scrub", "capture",
       "live"]


# ------------------------------------------------------------------ data
def build_sim_log(work: Path) -> Path:
    """A real closed-loop run (3 RW + 3 MTQ, coordinate goal) with an imported model."""
    import trimesh

    from ADCS.satellite_hardware.disturbances import (
        Drag_Disturbance,
        GeometryConfig,
        GeometryFace,
    )
    from ADCS.viz.mesh_import import import_spacecraft

    bus = (0.10, 0.10, 0.20)
    faces = []
    for axis, (area, half) in enumerate([(bus[1] * bus[2], bus[0] / 2), (bus[0] * bus[2], bus[1] / 2),
                                         (bus[0] * bus[1], bus[2] / 2)]):
        for sign in (1.0, -1.0):
            n = np.zeros(3)
            n[axis] = sign
            faces.append(GeometryFace(area=area, centroid=n * half, normal=n, CD=2.2))
    mtqs = [ADCS.MTQ(axis=a, max_torque=0.1) for a in np.eye(3)]
    rws = [ADCS.RW(axis=a, max_torque=0.01, J=2e-5, h=0.0, h_max=0.02) for a in np.eye(3)]
    sat = ADCS.Satellite(mass=2.6, J_0=np.diag([0.011, 0.011, 0.0043]), actuators=mtqs + rws,
                         sensors=[ADCS.MTM(axis=a) for a in np.eye(3)], disturbances=[Drag_Disturbance(GeometryConfig(faces))],
                         boresight=np.array([0, 0, 1]))

    # Model: bus + three wheels + a deployable array, exported as glTF.
    sc = trimesh.Scene()
    box = trimesh.creation.box(extents=bus)
    box.visual.face_colors = [180, 180, 190, 255]
    sc.add_geometry(box, node_name="bus", geom_name="bus")
    rot = trimesh.transformations.rotation_matrix
    for name, T, pos in (("rw_x", rot(np.pi / 2, [0, 1, 0]), [0, 0, -0.05]), ("rw_y", rot(-np.pi / 2, [1, 0, 0]), [0, 0, 0]),
                         ("rw_z", np.eye(4), [0, 0, 0.05])):
        T = T.copy()
        T[:3, 3] = pos
        wheel = trimesh.creation.cylinder(radius=0.03, height=0.012, sections=48)
        wheel.visual.face_colors = [200, 120, 40, 255]
        sc.add_geometry(wheel, node_name=name, geom_name=name, transform=T)
    array = trimesh.creation.box(extents=(0.002, 0.30, 0.18))
    array.visual.face_colors = [30, 40, 110, 255]
    sc.add_geometry(array, node_name="array", geom_name="array",
                    transform=trimesh.transformations.translation_matrix([0.051, 0.0, 0.0]))
    (work / "model").mkdir(parents=True, exist_ok=True)
    sc.export(work / "model" / "demo.glb")
    res = import_spacecraft({"name": "demo", "mesh": str(work / "model" / "demo.glb"), "nodes": {
        "rw_x": {"role": "rw", "actuator": "rw0"}, "rw_y": {"role": "rw", "actuator": "rw1"},
        "rw_z": {"role": "rw", "actuator": "rw2"}, "array": {"role": "panel"}}}, work / "model", satellite=sat)
    if not res.ok:
        raise RuntimeError(f"model import failed: {res.errors}")

    ctrl = ADCS.controller.MTQ_w_RW(est_sat=sat, p_gain=0.002, d_gain=0.02, c_gain=0.001, h_target=np.zeros(3))
    os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(), J2000=0.22, R=np.array([5000.0, 0, 5000.0]), V=np.array([0, 7.5, 0]))
    x0 = ADCS.State.from_array(np.array([0.01, 0.02, -0.01, 1, 0, 0, 0, 0, 0, 0]))
    results = ADCS.simulate(x=x0, satellite=sat, controller=ctrl, goal=ADCS.goals.Coordinate_Goal(lat=42.36, lon=-71.09, alt=0),
                            os0=os0, dt=2.0, tf=1200.0)
    path = work / "sim.vizlog"
    results_to_vizlog(results, path, model=res.sidecar, overwrite=True)
    return path


def build_landmark_cache(work: Path) -> Path:
    """Asset cache whose earth_day is built, by the real pipeline, from a texture with red dots."""
    from ADCS.viz.assets import products

    tex = work / "landmarks.png"
    build_landmark_texture(tex, 4096, 2048)
    data = tex.read_bytes()
    sources = {"bmng_topo_bathy_200407": {"url": tex.as_uri(), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                                          "license": "test", "attribution": "test", "redistributable": True,
                                          "requires_acceptance": False}}
    cache = work / "landmark_cache"
    old = os.environ.get("ADCS_VIZ_CACHE")
    os.environ["ADCS_VIZ_CACHE"] = str(cache)
    try:
        products.build(["earth_day"], overrides={"earth_day": {"face_size": 1024}}, sources=sources, log=lambda *_: None)
    finally:
        if old is None:
            os.environ.pop("ADCS_VIZ_CACHE")
        else:
            os.environ["ADCS_VIZ_CACHE"] = old
    return cache


# ------------------------------------------------------------------ running
class Runner:
    def __init__(self, godot: Path, timeout: float) -> None:
        self.godot = godot
        self.timeout = timeout
        self.windows = godot.suffix.lower() == ".exe" and is_wsl()
        self.project = viewer_project(godot)
        self.work = (_local_appdata() / "generalized_adcs" / "viewer_checks") if self.windows else (HERE / "results" / "viewer")
        if self.work.exists():
            shutil.rmtree(self.work)
        self.work.mkdir(parents=True)
        self.assets = _default_assets()

    def host(self, p: Path) -> str:
        if self.windows:
            return subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True, check=True).stdout.strip()
        return str(p)

    def run(self, name: str, check: str, log: Path, cfg: dict | None = None, assets: Path | None = None,
            extra: tuple = (), engine: tuple = (), via_file: bool = False) -> dict:
        out = self.work / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(cfg or {}))
        args = ["--check", check, "--config", self.host(out / "config.json"), "--out", self.host(out), *extra]
        asset_dir = assets if assets is not None else self.assets
        if via_file:
            cmd = [str(self.godot), "--path", self.host(self.project), *engine, "--", "--log", self.host(log), *args]
            if asset_dir:
                cmd += ["--assets", self.host(asset_dir)]
            return self._exec(cmd, out)
        with VizServer() as server:
            server.serve_log(VizLog.open(log))
            h, p = server.address
            cmd = viewer_command(self.godot, h, p, server.token, args, self.project, asset_dir)
            if engine:
                cmd = cmd[:cmd.index("--")] + list(engine) + cmd[cmd.index("--"):]
            return self._exec(cmd, out)

    def _exec(self, cmd: list, out: Path) -> dict:
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            output, _ = proc.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            output = "[timed out]\n" + proc.communicate()[0]
        (out / "godot_output.txt").write_text(output)
        f = out / "result.json"
        r = json.loads(f.read_text()) if f.exists() else {"error": "no result.json"}
        r["_elapsed_s"] = time.perf_counter() - t0
        if proc.returncode != 0 or "error" in r:
            r.setdefault("error", f"viewer exited {proc.returncode}")
            r["_errors"] = [ln for ln in output.splitlines() if "ERROR" in ln or "rror:" in ln][:20]
        return r


def run_live(runner: "Runner") -> dict:
    """Viewer connected while ADCS.simulate streams to it (broadcast)."""
    import threading

    out = runner.work / "live"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"timeout_s": 180}))
    sat = ADCS.Satellite(mass=4.0, J_0=np.diag([0.03, 0.03, 0.01]),
                         actuators=[ADCS.RW(axis=a, max_torque=0.01, J=2e-5, h=0.0, h_max=0.02) for a in np.eye(3)],
                         sensors=[ADCS.MTM(axis=a) for a in np.eye(3)])
    os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(), J2000=0.22, R=np.array([5000.0, 0, 5000.0]), V=np.array([0, 7.5, 0]))
    x0 = ADCS.State.from_array(np.array([0.02, 0.01, -0.01, 1, 0, 0, 0, 0, 0, 0]))
    box = {}
    with VizServer(mode="broadcast", queue_size=100000) as server:
        h, p = server.address
        cmd = viewer_command(runner.godot, h, p, server.token,
                             ["--check", "live", "--config", runner.host(out / "config.json"), "--out", runner.host(out),
                              "--res", "1280x720"], runner.project, runner.assets)
        th = threading.Thread(target=lambda: box.update(r=runner._exec(cmd, out)), daemon=True)
        th.start()
        t0 = time.perf_counter()
        # Long enough (~10 s of simulation) that the viewer is up while it is still running.
        results = ADCS.simulate(x=x0, satellite=sat, os0=os0, dt=1.0, tf=2400.0, viz_stream=server)
        sim_s = time.perf_counter() - t0
        th.join(runner.timeout)
    r = box.get("r", {"error": "viewer did not finish"})
    if "error" in r:
        return r
    lag = np.array(r["follow_lag_s"])
    n = len(results.runs[0].state_hist)
    return {"passed": r["live"] and r["finished"] and r["rows"] == n and r["frames"] > 10,
            "criterion": "viewer follows a running simulation (broadcast) and ends with every row",
            "rows": r["rows"], "expected_rows": n, "frames_while_live": r["frames"], "sim_seconds": round(sim_s, 1),
            "follow_lag_s": stats(lag) if len(lag) else None}


# ------------------------------------------------------------------ judging
def stats(a) -> dict:
    a = np.asarray(a, float)
    return {"p50": float(np.percentile(a, 50)), "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checks", nargs="*", default=ALL)
    ap.add_argument("--godot")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()
    runner = Runner(find_godot(args.godot), args.timeout)
    print(f"[viewer] work dir {runner.work}; assets {runner.assets}", flush=True)
    sim_log = build_sim_log(runner.work)
    results = {}

    def report(name, verdict):
        results[name] = verdict
        state = "ERROR " + str(verdict.get("error"))[:200] if "error" in verdict else \
            ("PASS" if verdict.get("passed") else ("recorded" if verdict.get("passed") is None else "FAIL"))
        print(f"[viewer] {name}: {state}", flush=True)

    for name in args.checks:
        if name == "assets":
            r = runner.run("assets", "assets", sim_log)
            want = {"earth_day", "earth_night", "earth_clouds", "earth_normals", "earth_ocean_mask", "moon_color", "moon_normals"}
            ok = "error" not in r and want <= set(r["loaded"]) and r["loaded"].get("stars", 0) > 9000 \
                and r["model"] == "model" and r["model_lods"] == 3 and r["wheels"] == 9 and r["source"] == "stream"
            report(name, r if "error" in r else {"passed": ok, "criterion": "all 8 asset products, model (3 LODs, 3 wheels x 3 LODs) over the stream",
                                                  "loaded": r["loaded"], "load_ms": r["load_ms"], "model": r["model"],
                                                  "model_lods": r["model_lods"], "wheels": r["wheels"], "n_rows": r["n_rows"]})
        elif name == "file_source":
            r = runner.run("file_source", "assets", sim_log, via_file=True)
            report(name, r if "error" in r else {"passed": r["source"] == "file" and r["model"] == "model" and r["n_rows"] == 600,
                                                  "criterion": "vizlog read from disk, model found via scene.json paths",
                                                  "n_rows": r["n_rows"], "model": r["model"]})
        elif name == "shots":
            shots = [{"name": "01_craft_sunlit", "dist": 0.7, "sunward": True, "yaw_deg": 35, "pitch_deg": 20},
                     {"name": "02_craft_over_earth", "dist": 2.0, "sunward": True, "yaw_deg": 70, "pitch_deg": -35},
                     {"name": "03_craft_100km", "dist": 1.0e5, "sunward": True, "yaw_deg": 40, "pitch_deg": 30},
                     {"name": "04_day_side", "mode": "earth", "dist": 2.2e7, "sunward": True, "yaw_deg": 25, "pitch_deg": 10},
                     {"name": "05_terminator_night", "mode": "earth", "dist": 1.6e7, "sunward": True, "yaw_deg": 120},
                     {"name": "06_earth_moon", "mode": "earth", "dist": 4.5e8, "sunward": True, "yaw_deg": 150, "pitch_deg": 5}]
            r = runner.run("shots", "shots", sim_log, {"shots": shots}, extra=("--res", "1600x900"))
            report(name, r if "error" in r else {"passed": None, "criterion": "screenshots for visual review",
                                                  "files": [s["name"] for s in r["shots"]], "dir": str(runner.work / "shots"),
                                                  "info": {s["name"]: s["info"] for s in r["shots"]}})
        elif name in ("jitter", "jitter_control"):
            control = name == "jitter_control"
            r = runner.run(name, "jitter", sim_log, {"frames": 300, "absolute": control}, extra=("--res", "1280x720"))
            if "error" in r:
                report(name, r)
                continue
            d = np.array(r["max_diff"])
            ok = (d.max() > 5.0) if control else (d.max() < 0.5)
            report(name, {"passed": bool(ok), "criterion": ("control (absolute float32) must visibly move: mean ROI diff > 5"
                                                           if control else "spacecraft static in a body-locked view: mean ROI diff < 0.5 / 255"),
                          "roi_mean_abs_diff": stats(d)})
        elif name == "terminator":
            r = runner.run("terminator", "terminator", sim_log, extra=("--res", "1280x720"))
            if "error" in r:
                report(name, r)
                continue
            exp = subsolar_expected(VizLog.open(sim_log))
            dlon = (r["subsolar"]["lon_deg"] - exp["lon_deg"] + 180) % 360 - 180
            num = float(np.hypot(dlon * np.cos(np.radians(exp["lat_deg"])), r["subsolar"]["lat_deg"] - exp["lat_deg"]))
            off = np.array(r["offsets_px"])
            img = float(np.degrees(abs(off.mean()) / r["disk_radius_px"])) if len(off) else float("inf")
            report(name, {"passed": num < 0.005 and img < 0.1, "criterion": "sub-solar point < 0.005 deg; image terminator < 0.1 deg",
                          "numeric_err_deg": num, "image_err_deg": img, "image_offset_px": float(off.mean()) if len(off) else None})
        elif name == "landmarks":
            log = runner.work / "phase0.vizlog"
            build_log(log)
            cache = build_landmark_cache(runner.work)
            r = runner.run("landmarks", "landmarks", log, {"points": landmark_dirs(VizLog.open(log))}, assets=cache,
                           extra=("--res", "1280x720"))
            report(name, r if "error" in r else {**judge_landmarks(r), "criterion": "red dot at screen centre within 2 px, "
                                                  "texture through the asset pipeline's cube maps"})
        elif name == "scrub":
            log = runner.work / "day.vizlog"
            build_log(log, seconds=86400)
            r = runner.run("scrub", "scrub", log, {"jumps": 200}, extra=("--res", "1600x900"))
            if "error" in r:
                report(name, r)
                continue
            s = stats(r["displayed_ms"])
            report(name, {"passed": s["p99"] < 50.0, "criterion": "jump anywhere in a 24 h log (86 401 rows) and show it: p99 < 50 ms",
                          "cpu_ms": stats(r["cpu_ms"]), "displayed_ms": s, "rows": r["n_rows"]})
        elif name == "capture":
            hashes = []
            for k in range(2):
                movie = runner.work / f"capture_{k}"
                if movie.exists():
                    shutil.rmtree(movie)
                movie.mkdir(parents=True)
                r = runner.run(f"capture_run{k}", "capture", sim_log, {"frames": 90},
                               engine=("--write-movie", runner.host(movie / "frame.png"), "--fixed-fps", "60"),
                               extra=("--res", "960x540"))
                if "error" in r:
                    break
                hashes.append([hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(movie.glob("frame*.png"))])
            if len(hashes) < 2:
                report(name, r)
                continue
            report(name, {"passed": hashes[0] == hashes[1] and len(hashes[0]) >= 90,
                          "criterion": "two --write-movie runs: bit-identical frames", "frames": len(hashes[0])})

    if "live" in args.checks:
        report("live", run_live(runner))

    meta = {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "godot": str(runner.godot)}
    out_root = HERE / "results"
    out_root.mkdir(exist_ok=True)
    (out_root / "viewer_latest.json").write_text(json.dumps({"meta": meta, "results": results}, indent=1, default=str))
    lines = ["# Viewer end-to-end checks", "", f"Generated {meta['date']} by `tools/viz_validation/run_viewer_checks.py`.", "",
             "| Check | Result | Details | Criterion |", "|---|---|---|---|"]
    for name, r in results.items():
        state = "ERROR" if "error" in r else ("recorded" if r.get("passed") is None else ("PASS" if r["passed"] else "FAIL"))
        detail = {k: v for k, v in r.items() if k not in ("passed", "criterion", "info", "files", "dir", "_errors")}
        text = json.dumps(detail, default=lambda v: round(v, 4) if isinstance(v, float) else str(v))
        lines.append(f"| {name} | {state} | {text[:300]} | {r.get('criterion', '')} |")
    (HERE / "VIEWER_CHECKS.md").write_text("\n".join(lines) + "\n")
    print(f"[viewer] report: {HERE / 'VIEWER_CHECKS.md'}")
    return 0 if all(r.get("passed") is not False and "error" not in r for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
