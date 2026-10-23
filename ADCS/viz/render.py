r"""
Offline video of a visualization log.

The viewer renders every frame at a fixed simulation time and camera (never
from the wall clock), so the same job gives the same frames on the same
machine. Supersampling and motion blur are done here, in linear light: the
viewer writes each sub-frame at ``supersample`` times the output resolution,
then they are averaged (box filter) and encoded with ffmpeg.

.. code-block:: python

   from ADCS.viz.render import CameraKey, RenderJob, render_video
   job = RenderJob(duration_s=12, fps=30, supersample=2, motion_blur=4, keys=[
       CameraKey(at=0, mode="orbit", frame="lvlh", dist_m=1.5, yaw_deg=-140, pitch_deg=18),
       CameraKey(at=8, mode="orbit", frame="lvlh", dist_m=2.0e6, yaw_deg=-60, pitch_deg=30),
   ])
   render_video("run.vizlog", job, "run.mp4")

Camera keys are placed in video time. Between two keys of the same mode and
frame the camera eases (smoothstep) with distance interpolated
logarithmically and yaw along the shorter way round; a key with a different
mode or frame is a cut. Simulation time runs linearly from ``t_start`` to
``t_end`` over the video.
"""

__all__ = ["CameraKey", "RenderJob", "RenderResult", "camera_at", "samples", "render_video", "find_ffmpeg",
           "average_linear", "srgb_to_linear", "linear_to_srgb", "load_job", "parse_key"]

import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

MODES = ("orbit", "earth")
FRAMES = ("lvlh", "inertial", "body")
CODECS = {"h264": (".mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow"]),
          "h265": (".mp4", ["-c:v", "libx265", "-pix_fmt", "yuv420p", "-tag:v", "hvc1"]),
          "prores": (".mov", ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]),
          "none": ("", [])}


@dataclass
class CameraKey:
    """A camera at video time ``at`` [s]. ``mode="earth"`` orbits the Earth at ``dist_m`` from its centre."""

    at: float
    mode: str = "orbit"
    frame: str = "lvlh"
    dist_m: float = 3.0
    yaw_deg: float = -140.0
    pitch_deg: float = 18.0
    fov_deg: float = 45.0
    ease: str = "smooth"  # "smooth" or "linear", for the segment that ends at this key


@dataclass
class RenderJob:
    """
    What to render.

    :param t_start: Simulation time of the first frame [s since the log epoch]; default: the start of the log.
    :param t_end: Simulation time of the last frame; default: the end of the log.
    :param duration_s: Video length [s].
    :param supersample: Rendered at this multiple of the output size per axis (1-4), then averaged.
    :param motion_blur: Sub-frames per frame, spread over ``shutter`` of the frame interval (1 = none).
    :param overlays: Overlay groups to show, e.g. ``{"vectors": False}``; others keep the log's setting.
    :param ui: Draw the timeline and HUD into the video (needs ``supersample=1`` and at least 960x540).
    """

    duration_s: float = 10.0
    fps: float = 30.0
    width: int = 1920
    height: int = 1080
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    supersample: int = 1
    motion_blur: int = 1
    shutter: float = 0.5
    quality: str = "high"
    exposure: str = "artistic"
    ev: float = 0.0
    overlays: Dict[str, bool] = field(default_factory=dict)
    ui: bool = False
    keys: List[CameraKey] = field(default_factory=lambda: [CameraKey(at=0.0)])

    def validate(self) -> None:
        if self.duration_s <= 0 or self.fps <= 0:
            raise ValueError("duration_s and fps must be positive")
        if self.width < 16 or self.height < 16 or self.width % 2 or self.height % 2:
            raise ValueError("width and height must be even and at least 16 (video encoders need even sizes)")
        if self.supersample not in (1, 2, 3, 4):
            raise ValueError("supersample must be 1, 2, 3 or 4")
        if max(self.width, self.height) * self.supersample > 16384:
            raise ValueError("width/height x supersample exceeds the 16384 px render limit")
        if self.motion_blur < 1 or not 0.0 <= self.shutter <= 1.0:
            raise ValueError("motion_blur must be >= 1 and shutter in [0, 1]")
        if self.ui and self.supersample != 1:
            raise ValueError("ui=True draws the window itself, so it needs supersample=1")
        if self.ui and (self.width < 960 or self.height < 540):
            raise ValueError("ui=True needs at least 960x540 (the timeline and HUD are laid out for window sizes)")
        if self.quality not in ("high", "balanced", "fast") or self.exposure not in ("artistic", "physical"):
            raise ValueError("quality: high/balanced/fast; exposure: artistic/physical")
        if not self.keys:
            raise ValueError("at least one camera key is needed")
        for k in self.keys:
            if k.mode not in MODES or k.frame not in FRAMES or k.dist_m <= 0 or not 0.1 <= k.fov_deg <= 150:
                raise ValueError(f"bad camera key {k}")
            if k.ease not in ("smooth", "linear"):
                raise ValueError(f"ease must be 'smooth' or 'linear' ({k})")
        ats = [k.at for k in self.keys]
        if ats != sorted(ats):
            raise ValueError("camera keys must be in time order")
        for g in self.overlays:
            if g not in ("vectors", "axes", "tracks"):
                raise ValueError(f"unknown overlay group {g!r}")

    @property
    def n_frames(self) -> int:
        return max(1, int(round(self.duration_s * self.fps)))

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RenderResult:
    video: Optional[Path]
    frames_dir: Path
    metadata: Dict[str, Any]


def camera_at(keys: Sequence[CameraKey], u: float) -> Dict[str, Any]:
    """Camera at video time u: eased between keys of the same mode and frame, held then cut otherwise."""
    if u <= keys[0].at or len(keys) == 1:
        k = keys[0]
        return {"mode": k.mode, "frame": k.frame, "dist_m": k.dist_m, "yaw_deg": k.yaw_deg,
                "pitch_deg": k.pitch_deg, "fov_deg": k.fov_deg}
    for a, b in zip(keys[:-1], keys[1:]):
        if u <= b.at:
            break
    else:
        a = b = keys[-1]
    if a is b or (a.mode, a.frame) != (b.mode, b.frame) or b.at <= a.at:
        k = a if u < b.at else b
        return {"mode": k.mode, "frame": k.frame, "dist_m": k.dist_m, "yaw_deg": k.yaw_deg,
                "pitch_deg": k.pitch_deg, "fov_deg": k.fov_deg}
    s = (u - a.at) / (b.at - a.at)
    if b.ease == "smooth":
        s = s * s * (3.0 - 2.0 * s)
    dyaw = (b.yaw_deg - a.yaw_deg + 180.0) % 360.0 - 180.0
    return {"mode": a.mode, "frame": a.frame,
            "dist_m": float(np.exp(np.log(a.dist_m) + s * (np.log(b.dist_m) - np.log(a.dist_m)))),
            "yaw_deg": a.yaw_deg + s * dyaw, "pitch_deg": a.pitch_deg + s * (b.pitch_deg - a.pitch_deg),
            "fov_deg": a.fov_deg + s * (b.fov_deg - a.fov_deg)}


def samples(job: RenderJob, t_first: float, t_last: float) -> List[Dict[str, Any]]:
    """
    Every image to render: ``{i, j, u, t, cam, layer}`` (frame i, image j, video time u, simulation time t).

    Frame i is at u = i / fps. Without motion blur it is one image of everything (``layer="all"``). With
    ``motion_blur = m``, images 0..m-1 are the scene without overlays at times centred on the frame and spread
    evenly over ``shutter`` / fps; image m is everything and m+1 the scene alone, both at the frame time (the
    overlays are taken from their difference, so they stay sharp).
    """
    t0 = t_first if job.t_start is None else job.t_start
    t1 = t_last if job.t_end is None else job.t_end
    if not t_first - 1e-9 <= t0 < t1 <= t_last + 1e-9:
        raise ValueError(f"t_start/t_end [{t0}, {t1}] must lie inside the log [{t_first}, {t_last}] and increase")
    out = []
    n, m = job.n_frames, job.motion_blur

    def at(i, j, u, layer):
        u = min(max(u, 0.0), job.duration_s)
        t = t0 + (t1 - t0) * u / job.duration_s
        return {"i": i, "j": j, "u": u, "t": float(min(max(t, t0), t1)), "cam": camera_at(job.keys, u), "layer": layer}

    for i in range(n):
        if m == 1:
            out.append(at(i, 0, i / job.fps, "all"))
            continue
        # Motion blur: the scene alone at each sub-frame time, then the frame time with and without the overlays
        # (and HUD) so they can be laid over the blurred scene sharp.
        for j in range(m):
            out.append(at(i, j, i / job.fps + (job.shutter / job.fps) * ((j + 0.5) / m - 0.5), "scene"))
        out.append(at(i, m, i / job.fps, "all"))
        out.append(at(i, m + 1, i / job.fps, "scene"))
    return out


# ------------------------------------------------------------------ image maths (linear light)
def srgb_to_linear(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64) / 255.0
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    s = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
    return np.clip(np.round(s * 255.0), 0, 255).astype(np.uint8)


def average_linear(images: Sequence[np.ndarray], supersample: int, sharp: Optional[tuple] = None) -> np.ndarray:
    """
    Mean of 8-bit sRGB images in linear light, then a box filter down by ``supersample``; back to 8-bit sRGB.

    :param sharp: ``(with_overlays, without_overlays)`` at the frame time: pixels where they differ (the
        overlays, labels and HUD) are taken from ``with_overlays`` instead of the mean, so they are not blurred.
    """
    acc = None
    for im in images:
        lin = srgb_to_linear(im[..., :3])
        acc = lin if acc is None else acc + lin
    acc /= len(images)
    if sharp is not None:
        on, off = (np.asarray(x)[..., :3] for x in sharp)
        mask = np.abs(on.astype(np.int16) - off.astype(np.int16)).max(axis=2) > 2
        acc[mask] = srgb_to_linear(on[mask])
    if supersample > 1:
        h, w = acc.shape[0] // supersample, acc.shape[1] // supersample
        acc = acc[: h * supersample, : w * supersample].reshape(h, supersample, w, supersample, 3).mean(axis=(1, 3))
    return linear_to_srgb(acc)


# ------------------------------------------------------------------ ffmpeg
def find_ffmpeg() -> Optional[str]:
    """ffmpeg on PATH, else the binary shipped with the ``imageio-ffmpeg`` package (extra ``viz-video``)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def ffmpeg_command(ffmpeg: str, frames_dir: Path, fps: float, out: Path, codec: str, crf: int) -> List[str]:
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", f"{fps:g}", "-i", str(frames_dir / "%06d.png")]
    cmd += CODECS[codec][1]
    if codec in ("h264", "h265"):
        cmd += ["-crf", str(crf), "-movflags", "+faststart"]
    return cmd + [str(out)]


# ------------------------------------------------------------------ running it
def _work_dir(binary: Path, name: str) -> Path:
    """Where the viewer writes frames: the Windows side for a Windows viewer started from WSL."""
    from ADCS.viz.binary_manager import _windows_local_appdata, is_wsl

    if is_wsl() and str(binary).lower().endswith(".exe"):
        base = _windows_local_appdata()
        if base is not None:
            return base / "generalized_adcs" / "render" / name
    import tempfile
    return Path(tempfile.gettempdir()) / "generalized_adcs_render" / name


def render_video(log_path: str | Path, job: RenderJob, out: str | Path, *, binary: Optional[str | Path] = None,
                 project: Optional[str | Path] = None, assets: Optional[str | Path] = None, codec: str = "h264",
                 crf: int = 16, keep_frames: bool = False, run_id: Optional[str] = None,
                 timeout_s: float = 7200.0, keep_work: bool = False, log=print) -> RenderResult:
    r"""
    Render a vizlog to a video.

    :param out: Output video (the extension follows ``codec``), or with ``codec="none"`` a directory for the PNGs.
    :param binary: Viewer executable (see :func:`~ADCS.viz.binary_manager.find_viewer`).
    :param codec: ``h264`` (default), ``h265``, ``prores`` (10-bit, for editing) or ``none`` (PNG frames only).
    :param crf: Quality for h264/h265 (lower is better; 16 is visually lossless).
    :param keep_frames: Keep the final PNG frames next to the video (``<out>_frames``).
    :param keep_work: Keep the viewer's raw images and the job (for debugging); their folder is in the metadata.
    :return: Paths and the metadata also written to ``<out>.json``.
    :raises RuntimeError: If the viewer fails, or ffmpeg is needed but not found.
    """
    from PIL import Image

    from ADCS.viz.binary_manager import find_viewer
    from ADCS.viz.launcher import (
        _default_assets,
        _host_path,
        _stop,
        launch_viewer,
        viewer_project,
    )
    from ADCS.viz.reader import VizLog
    from ADCS.viz.stream import VizServer

    job.validate()
    if codec not in CODECS:
        raise ValueError(f"codec must be one of {list(CODECS)}")
    ffmpeg = find_ffmpeg() if codec != "none" else None
    if codec != "none" and ffmpeg is None:
        raise RuntimeError("ffmpeg not found: install it, or `pip install imageio-ffmpeg` (extra viz-video), "
                           "or use codec='none' for PNG frames")
    out = Path(out)
    if codec != "none":
        out = out.with_suffix(CODECS[codec][0])
    vlog = VizLog.open(log_path)
    rid = str(run_id) if run_id is not None else str(vlog.run_ids[0])
    ts = vlog.read(rid, "t_s")[:, 0]
    plan = samples(job, float(ts[0]), float(ts[-1]))
    exe = find_viewer(binary)
    job_hash = hashlib.sha256(json.dumps(job.to_json(), sort_keys=True).encode()).hexdigest()[:16]
    work = _work_dir(exe, f"{out.stem}_{job_hash}")
    if work.exists():
        shutil.rmtree(work)
    raw = work / "raw"
    raw.mkdir(parents=True)
    rate = (plan[-1]["t"] - plan[0]["t"]) / max(plan[-1]["u"] - plan[0]["u"], 1e-9) if len(plan) > 1 else 1.0
    (work / "job.json").write_text(json.dumps({**job.to_json(), "samples": plan, "playback_rate": rate}))
    asset_dir = Path(assets) if assets is not None else _default_assets()
    res = f"{job.width * job.supersample}x{job.height * job.supersample}"
    args = ["--render", _host_path(work / "job.json", exe), "--out", _host_path(raw, exe), "--res", res,
            "--quality", job.quality, "--run", str(rid)]
    log(f"[render] {job.n_frames} frames x {job.motion_blur} sub-frames at {res} -> {out}")
    t_wall = time.perf_counter()
    with VizServer() as server:
        server.serve_log(vlog)
        h, p = server.address
        proc = launch_viewer(exe, h, p, server.token, args, viewer_project(exe, project), asset_dir)
        try:
            proc.wait(timeout=timeout_s)
        finally:
            _stop(proc)
    meta_path = raw / "render.json"
    if not meta_path.exists():
        raise RuntimeError(f"the viewer did not finish rendering (exit {proc.returncode}); see its console output")
    viewer_meta = json.loads(meta_path.read_text())
    if viewer_meta.get("error"):
        raise RuntimeError(f"viewer: {viewer_meta['error']}")
    t_render = time.perf_counter() - t_wall

    # Combine sub-frames and supersamples in linear light.
    frames_dir = out if codec == "none" else work / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    simple = job.motion_blur == 1 and job.supersample == 1
    for i in range(job.n_frames):
        img = lambda j: np.asarray(Image.open(raw / f"{i:06d}_{j:02d}.png").convert("RGB"))
        m = job.motion_blur
        if simple:
            shutil.copyfile(raw / f"{i:06d}_00.png", frames_dir / f"{i:06d}.png")
        elif m == 1:
            Image.fromarray(average_linear([img(0)], job.supersample)).save(frames_dir / f"{i:06d}.png")
        else:
            Image.fromarray(average_linear([img(j) for j in range(m)], job.supersample, sharp=(img(m), img(m + 1)))
                            ).save(frames_dir / f"{i:06d}.png")
    video = None
    ffmpeg_version = None
    if codec != "none":
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(ffmpeg_command(ffmpeg, frames_dir, job.fps, out, codec, crf), check=True)
        ffmpeg_version = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True,
                                        check=False).stdout.splitlines()[0]
        video = out
        if keep_frames:
            kept = out.with_name(out.stem + "_frames")
            if kept.exists():
                shutil.rmtree(kept)
            shutil.copytree(frames_dir, kept)
            frames_dir = kept
    meta = {"format": "adcs-viz-render", "version": 1, "log": str(Path(log_path).resolve()), "run": rid,
            "log_epoch": vlog.manifest.get("epoch"), "job": job.to_json(), "job_sha256_16": job_hash,
            "frames": job.n_frames, "sub_frames": len(plan), "render_size": res,
            "output_size": f"{job.width}x{job.height}", "codec": codec, "crf": crf if codec in ("h264", "h265") else None,
            "ffmpeg": ffmpeg_version, "viewer": viewer_meta, "render_wall_s": t_render,
            "work_dir": str(work) if keep_work else None,
            "total_wall_s": time.perf_counter() - t_wall}
    side = (out.with_suffix(".json") if codec != "none" else out / "render.json")
    side.write_text(json.dumps(meta, indent=1, default=str))
    if not keep_work:
        shutil.rmtree(work / "raw", ignore_errors=True)
        if codec != "none" and not keep_frames:
            shutil.rmtree(work, ignore_errors=True)
            frames_dir = Path()
    log(f"[render] done in {meta['total_wall_s']:.1f} s -> {video or out}")
    return RenderResult(video, frames_dir, meta)


def load_job(path: str | Path) -> RenderJob:
    """A job from JSON: the RenderJob fields, with ``keys`` as a list of CameraKey fields."""
    d = json.loads(Path(path).read_text())
    keys = [CameraKey(**k) for k in d.pop("keys", [])] or [CameraKey(at=0.0)]
    return RenderJob(**d, keys=keys)


def parse_key(text: str) -> CameraKey:
    """``"at=0,mode=orbit,frame=lvlh,dist=1.5,yaw=-140,pitch=18,fov=45"`` -> CameraKey (CLI shorthand)."""
    names = {"at": "at", "mode": "mode", "frame": "frame", "dist": "dist_m", "yaw": "yaw_deg", "pitch": "pitch_deg",
             "fov": "fov_deg", "ease": "ease"}
    kw: Dict[str, Any] = {}
    for part in text.split(","):
        k, _, v = part.partition("=")
        k = k.strip()
        if k not in names:
            raise ValueError(f"camera key field {k!r}: use one of {list(names)}")
        kw[names[k]] = v.strip() if k in ("mode", "frame", "ease") else float(v)
    if "at" not in kw:
        raise ValueError("a camera key needs at=<video time>")
    return CameraKey(**kw)
