import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from pathlib import Path

import numpy as np

import ADCS as ADCS
from ADCS.viz import VizLog, results_to_vizlog
from ADCS.viz import overlays as ov
from ADCS.viz import timeline as tl
from ADCS.viz.binary_manager import ViewerNotFoundError
from ADCS.viz.launcher import view
from ADCS.viz.mesh_import import import_spacecraft

parser = argparse.ArgumentParser(description="Tutorial 9: one orbit of a 2U CubeSat drawn from its STL model.")
parser.add_argument("--video", action="store_true", help="also render tutorial_9_output/space_raccoon.mp4")
args = parser.parse_args()

HERE = Path(__file__).resolve().parent
STL = HERE / "Custom_STLs" / "SPACE_RACCOON_Dummy_Chassis.stl"
OUT = Path("tutorial_9_output")
OUT.mkdir(exist_ok=True)

# --- The spacecraft -------------------------------------------------------
# A 2U CubeSat (10 x 10 x 22.7 cm, 2.6 kg) with three magnetometers. Its inertia
# is that of a uniform box of that size; the long axis is body Z.
mass = 2.6
a, b, c = 0.10, 0.10, 0.227
J = mass / 12.0 * np.diag([b**2 + c**2, a**2 + c**2, a**2 + b**2])
satellite = ADCS.Satellite(mass=mass, J_0=J, sensors=[ADCS.MTM(axis) for axis in np.eye(3)])

# --- Its 3-D model ----------------------------------------------------------
# STL files carry no units: this chassis was drawn in inches (8.94 in = 22.7 cm).
# Its CAD origin is off-centre, so the translation moves the centre of the part
# onto the body origin (the centre of mass). The import checks the model against
# the satellite and writes reduced levels of detail for the viewer.
model_config = {
    "name": "space_raccoon",
    "mesh": str(STL),
    "units": "in",
    "cad_to_body": {"rotation": "identity", "translation_m": [0.0149, -0.0045, -0.0662]},
    "lods": [0.5, 0.15, 0.04],
}
model = import_spacecraft(model_config, OUT / "model", satellite=satellite)
if not model.ok:
    raise SystemExit(f"model import failed: {model.errors}")
for w in model.warnings:
    print("model:", w)

# --- One orbit --------------------------------------------------------------
# A circular orbit 420 km up at 51.6 deg inclination (like the ISS), starting
# with a slow tumble and no controller: the attitude simply evolves.
mu = 398600.4418                       # km^3/s^2
r = 6378.137 + 420.0                   # km
v = np.sqrt(mu / r)
inc = np.radians(51.6)
os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(), J2000=0.22,
                         R=np.array([r, 0.0, 0.0]), V=np.array([0.0, v * np.cos(inc), v * np.sin(inc)]))
x_0 = ADCS.State.from_array(np.array([0.002, -0.003, 0.004] + [1, 0, 0, 0]))  # w [rad/s], q
period = 2 * np.pi * np.sqrt(r**3 / mu)

results = ADCS.simulate(x=x_0, satellite=satellite, os0=os0, dt=10.0, tf=round(period))

# --- Visualization log --------------------------------------------------------
# The model, the overlays and a ground station go into the log. Bookmarks mark
# the moments the spacecraft enters and leaves the Earth's shadow.
log_path = OUT / "space_raccoon.vizlog"
boston = tl.GroundSite("Boston", lat_deg=42.36, lon_deg=-71.06, min_elev_deg=10.0)
overlays = [
    ov.Triad("body"),
    ov.Arrow("@sun", label="Sun", color="#ffe066", scale="fixed_px", px=100),
    ov.Arrow("@nadir", label="nadir", color="#7cfc00", scale="fixed_px", px=80),
    ov.Arrow("@velocity", label="velocity", color="#9fd3ff", scale="fixed_px", px=80),
    ov.OrbitTrack(past_s=period, future_s=period),
    ov.GroundTrack(past_s=period, future_s=period),
]
results_to_vizlog(results, log_path, model=model.sidecar, overlays=overlays, sites=[boston], overwrite=True)

log = VizLog.open(log_path)
t = log.read("0", "t_s")[:, 0]
lit = log.read("0", "shadow_factor")[:, 0] > 0.5
bookmarks = [tl.Bookmark(t_s=float(t[0]), label="start")]
for k in np.flatnonzero(lit[1:] != lit[:-1]) + 1:
    bookmarks.append(tl.Bookmark(t_s=float(t[k]), label="enters eclipse" if lit[k - 1] else "back in sunlight"))
passes = tl.ground_passes(log, boston)
for p in passes:
    bookmarks.append(tl.Bookmark(t_s=p.t_max, label=f"over Boston ({p.max_elev_deg:.0f} deg)"))
results_to_vizlog(results, log_path, model=model.sidecar, overlays=overlays, sites=[boston], bookmarks=bookmarks,
                  overwrite=True)

print(f"One orbit: {period / 60:.1f} min, {np.mean(~lit) * 100:.0f}% of it in eclipse, "
      f"{len(passes)} pass(es) over Boston")

# --- Look at it -------------------------------------------------------------
if args.video:
    from ADCS.viz.render import CameraKey, RenderJob, render_video

    # 12 s of video for the whole orbit: start close to the spacecraft, pull back
    # to see the orbit, then cut to a view of the Earth.
    job = RenderJob(duration_s=12, fps=30, width=1920, height=1080, supersample=2, keys=[
        CameraKey(at=0, mode="orbit", frame="lvlh", dist_m=0.6, yaw_deg=-140, pitch_deg=18),
        CameraKey(at=7, mode="orbit", frame="lvlh", dist_m=3.0e6, yaw_deg=-70, pitch_deg=35),
        CameraKey(at=7.5, mode="earth", dist_m=2.2e7, yaw_deg=20, pitch_deg=25),
        CameraKey(at=12, mode="earth", dist_m=1.8e7, yaw_deg=80, pitch_deg=20),
    ])
    try:
        render_video(log_path, job, OUT / "space_raccoon.mp4")
    except (ViewerNotFoundError, RuntimeError) as e:
        print(f"video not rendered: {e}")

try:
    view(log_path)
except ViewerNotFoundError as e:
    print(f"3-D viewer not available ({e}).\nOpen the log later with:  python -m ADCS.viz view {log_path}")
