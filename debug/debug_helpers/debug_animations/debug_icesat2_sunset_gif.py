"""Render a 10x ICESat-2 sunset GIF for the repository front page.

Run from the repository root with the project virtual environment active:
    python -m debug.debug_helpers.debug_animations.debug_icesat2_sunset_gif

The locally cached NASA ICESat-2 model and Godot viewer are required. The GIF
is saved beside this script; intermediate vizlog and PNG frames are temporary.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import ADCS
from ADCS.helpers.animations.godot.assets.fetch import cache_dir
from ADCS.helpers.animations.godot.assets.spacecraft import ensure_model
from ADCS.helpers.animations.godot.from_results import results_to_vizlog
from ADCS.helpers.animations.godot.render import CameraKey, RenderJob, render_video
from ADCS.helpers.simresults import SimulationResults
from ADCS.helpers.math_helpers import dcm_to_quat, rotation_between_vectors


HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parents[3] / "docs" / "source" / "_static" / "release_notes" / "0_2_2_animation.gif"
SIMULATION_SECONDS = 240.0
PLAYBACK_RATE = 10.0
FPS = 10
RENDER_SIZE = (1600, 900)
OUTPUT_SIZE = (1280, 720)
PHASES = ((0.0, "NADIR POINTING"), (80.0, "COORDINATE POINTING"),
          (160.0, "ALONG-TRACK POINTING"))
# The NASA GLB's ATLAS telescope barrel runs along model-local +Y.
ATLAS_MEASUREMENT_AXIS_BODY = np.array([0.0, 1.0, 0.0])


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    stem = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / stem
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def _initial_orbit() -> ADCS.Orbital_State:
    """Place the spacecraft near the evening terminator at the log epoch."""
    epoch = 0.26
    ephem = ADCS.Ephemeris()
    radius = 6378.137 + 500.0  # km
    speed = np.sqrt(398600.4418 / radius)  # km/s
    radial_axis = np.array([1.0, 0.0, 0.0])
    tangent_axis = np.array([0.0, np.cos(np.deg2rad(51.6)), np.sin(np.deg2rad(51.6))])
    sun = np.asarray(ephem.position_eci("sun", epoch), dtype=float).reshape(3)
    sun /= np.linalg.norm(sun)
    sun_phase = np.arctan2(sun @ tangent_axis, sun @ radial_axis)
    projection = np.hypot(sun @ radial_axis, sun @ tangent_axis)
    # The orbital phase increases with time. A solar angle near 78 degrees
    # now becomes about 93 degrees after four minutes: visible dusk on Earth.
    phase = sun_phase + np.arccos(np.cos(np.deg2rad(78.0)) / projection)
    radial = np.cos(phase) * radial_axis + np.sin(phase) * tangent_axis
    tangent = -np.sin(phase) * radial_axis + np.cos(phase) * tangent_axis
    return ADCS.Orbital_State(ephem=ephem, J2000=epoch, R=radius * radial, V=speed * tangent)


def _ground_target(orbit: ADCS.Orbital_State) -> tuple[float, float]:
    """Choose a fixed Earth coordinate slightly ahead of the ground track."""
    # Advance the circular reference to the middle of the coordinate phase.
    omega = np.linalg.norm(orbit.V) / np.linalg.norm(orbit.R)
    angle = omega * 125.0
    future = np.cos(angle) * orbit.R + np.sin(angle) * orbit.V / omega
    target_eci = future / np.linalg.norm(future)
    # The geodetic coordinate is fixed in ECEF, so rotate it at target time.
    from skyfield import framelib

    t = orbit.ephem.ts.tt_jd(orbit.sf_pos.t.tt + 125.0 / 86400.0)
    ecef = framelib.itrs.rotation_at(t) @ target_eci
    lat = np.rad2deg(np.arctan2(ecef[2], np.hypot(ecef[0], ecef[1])))
    lon = np.rad2deg(np.arctan2(ecef[1], ecef[0]))
    return float(lat), float(lon)


def _simulate():
    mtqs = [ADCS.MTQ(axis=axis, max_torque=0.1) for axis in np.eye(3)]
    rws = [ADCS.RW(axis=axis, max_torque=4.51, J=0.22, h=0.0, h_max=3.8)
           for axis in np.eye(3)]
    orbit = _initial_orbit()
    initial_measurement_direction = -orbit.V / np.linalg.norm(orbit.V)
    q0 = dcm_to_quat(rotation_between_vectors(ATLAS_MEASUREMENT_AXIS_BODY,
                                               initial_measurement_direction))
    satellite = ADCS.Satellite(
        mass=4.0, J_0=np.diag([3.4, 2.9, 1.3]), actuators=mtqs + rws,
        sensors=[ADCS.MTM(axis=axis) for axis in np.eye(3)],
        boresight={"ATLAS measurement": ATLAS_MEASUREMENT_AXIS_BODY},
    )
    state = ADCS.State(w=np.zeros(3), q=q0, h=np.zeros(3))
    controller = ADCS.controller.MTQ_w_RW(
        est_sat=satellite, p_gain=0.04, d_gain=0.7, c_gain=0.1, h_target=np.zeros(3),
    )
    lat, lon = _ground_target(orbit)
    goals = ADCS.GoalList(
        goal_timeline={
            0.0: ADCS.goals.Nadir_Goal(),
            80.0: ADCS.goals.Coordinate_Goal(lat=lat, lon=lon, alt=0.0),
            160.0: ADCS.goals.LVLH_Tangential_Goal(),
        },
        time_units="seconds", start_juliantime=orbit.J2000,
    )
    return ADCS.simulate(x=state, satellite=satellite, controller=controller,
                         goal=goals, os0=orbit, dt=2.0, tf=SIMULATION_SECONDS + 2.0).first()


def _model_sidecar(model: Path, work: Path) -> Path:
    sidecar = work / "icesat2.viz.json"
    sidecar.write_text(json.dumps({
        "format": "adcs-spacecraft-model", "version": 1, "name": "ICESat-2",
        "coords": "BODY", "units": "m", "lods": [{"file": str(model)}], "nodes": {},
    }), encoding="utf-8")
    return sidecar


def _caption(frame: Image.Image, index: int, count: int) -> Image.Image:
    image = frame.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    sim_time = SIMULATION_SECONDS * index / (count - 1)
    phase = max(i for i, (at, _) in enumerate(PHASES) if sim_time >= at)
    title = _font(16, bold=True)
    label = _font(19, bold=True)
    draw.rounded_rectangle((16, 106, 315, 170), radius=11, fill=(4, 15, 27, 191))
    draw.text((28, 114), "ICESat-2", font=title, fill=(245, 250, 255))
    draw.text((28, 138), PHASES[phase][1], font=label, fill=(105, 222, 244))
    return image.convert("RGB")


def _gif(frames_dir: Path, output: Path) -> None:
    paths = sorted(frames_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9].png"))
    if not paths:
        raise RuntimeError("Godot produced no frames")
    # One palette for the whole animation avoids color shifts between frames.
    picks = np.linspace(0, len(paths) - 1, min(12, len(paths)), dtype=int)
    sheet = Image.new("RGB", (OUTPUT_SIZE[0], OUTPUT_SIZE[1] // 4 * 3))
    for slot, idx in enumerate(picks):
        with Image.open(paths[idx]) as source:
            thumb = source.convert("RGB").resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)
        thumb = _caption(thumb, int(idx), len(paths)).resize((OUTPUT_SIZE[0] // 4, OUTPUT_SIZE[1] // 4))
        sheet.paste(thumb, ((slot % 4) * thumb.width, (slot // 4) * thumb.height))
    palette = sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = []
    for i, path in enumerate(paths):
        with Image.open(path) as source:
            image = source.convert("RGB").resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)
        image = _caption(image, i, len(paths))
        frames.append(image.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG))
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=round(1000 / FPS),
                   loop=0, disposal=1, optimize=True)


def main() -> None:
    model = ensure_model("icesat2")
    if model is None:
        raise FileNotFoundError("ICESat-2 model is needed; open the viewer once to approve its NASA download")
    print("Simulating ICESat-2 nadir, coordinate, and along-track maneuvers...", flush=True)
    run = _simulate()
    with TemporaryDirectory(prefix="adcs-icesat2-gif-") as temporary:
        work = Path(temporary)
        log = results_to_vizlog(SimulationResults([run]), work / "icesat2.vizlog",
                                model=_model_sidecar(model, work),
                                torques="none")
        duration = SIMULATION_SECONDS / PLAYBACK_RATE
        job = RenderJob(duration_s=duration, fps=FPS, width=RENDER_SIZE[0], height=RENDER_SIZE[1],
                        quality="high", exposure="artistic", ui=True,
                        overlays={"axes": False, "tracks": False}, keys=[
                            CameraKey(at=0.0, dist_m=17.0, yaw_deg=-135, pitch_deg=23, fov_deg=48),
                            CameraKey(at=8.0, dist_m=15.0, yaw_deg=-95, pitch_deg=20, fov_deg=48),
                            CameraKey(at=16.0, dist_m=18.0, yaw_deg=-48, pitch_deg=17, fov_deg=48),
                            CameraKey(at=duration, dist_m=16.0, yaw_deg=-18, pitch_deg=22, fov_deg=48),
                        ])
        render_video(log, job, work / "frames", codec="none", assets=cache_dir())
        _gif(work / "frames", OUTPUT)
    print(f"Saved {OUTPUT} ({OUTPUT.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
