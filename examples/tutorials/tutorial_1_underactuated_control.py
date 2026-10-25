import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import ADCS as ADCS
import numpy as np
import matplotlib.pyplot as plt
from ADCS.viz import VizLog, results_to_vizlog
from ADCS.viz import overlays as ov
from ADCS.viz import timeline as tl
from ADCS.viz.binary_manager import ViewerNotFoundError
from ADCS.viz.launcher import view

acts: list[ADCS.Actuator] = [ADCS.RW(axis=np.array([1, 0, 0]), max_torque=0.0023, J=5.7e-6, h=0.0, h_max=0.0036)]
acts += [ADCS.MTQ(axis, max_torque=0.2) for axis in np.eye(3)]
sens = [ADCS.MTM(axis) for axis in np.eye(3)]

satellite = ADCS.Satellite(mass=4, J_0=np.diag([0.03, 0.03, 0.01]), actuators=acts, sensors=sens, boresight=np.array([0, 0, 1]))
x_0 = ADCS.State.from_array(np.array([0.01, -0.02, 0.01] + [1, 0, 0, 0] + [0.0]))  # w, q, h

controller = ADCS.controller.MTQ_w_RW_LP(est_sat=satellite, p_gain=0.00005, d_gain=0.002, c_gain=0.001, h_target=np.array([0, 0, 0]))

goal = ADCS.goals.Coordinate_Goal(lat=42.36, lon=-71.06, alt=0)
os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(),J2000=0.22, R=np.array([5000, 0, 5000]), V=np.array([0, 7.5, 0]))

results = ADCS.simulate(
    x=x_0,
    satellite=satellite,
    controller=controller,
    goal=goal,
    os0=os0,
    dt=1.0,
    tf=500.0
)

# --- 3-D view -------------------------------------------------------------
# Write the run as a visualization log. The overlays say what the viewer draws
# on top of the scene; the ground site adds a lane to the timeline showing when
# Boston can see the spacecraft (above 10 deg elevation).
log_path = "tutorial_1.vizlog"
boston = tl.GroundSite("Boston", lat_deg=42.36, lon_deg=-71.06, min_elev_deg=10.0)
overlays = [
    ov.Triad("body"),                                                         # body axes (X red, Y green, Z blue)
    ov.Arrow("@boresight:default", label="boresight", color="#ffffff", scale="fixed_px", px=110),
    ov.TargetLine(),                                                          # toward the ground target
    ov.Arrow("tau_act_body", label="control torque", color="#ff8c1a"),        # log-scaled length
    ov.Arrow("b_field", label="magnetic field", color="#d47cff", scale="fixed_px", px=80),
    ov.Arrow("h_rw_body", label="wheel momentum", color="#33e0c8"),
    ov.Arrow("@sun", label="Sun", color="#ffe066", scale="fixed_px", px=100),
    ov.OrbitTrack(past_s=500, future_s=500),
    ov.GroundTrack(past_s=500, future_s=500),
]
results_to_vizlog(results, log_path, overwrite=True, overlays=overlays, sites=[boston])

# The log can be analysed like any data set. Find the moments worth looking at
# and bookmark them, so they are one keypress away in the viewer's timeline.
log = VizLog.open(log_path)
t = log.read("0", "t_s")[:, 0]
err_deg = np.degrees(log.read("0", "pointing_err")[:, 0])
bookmarks = [tl.Bookmark(t_s=float(t[0]), label="start: tumbling")]
on_target = np.nonzero(err_deg < 10.0)[0]
if len(on_target):
    bookmarks.append(tl.Bookmark(t_s=float(t[on_target[0]]), label="pointing error < 10 deg"))
for p in tl.ground_passes(log, boston):
    print(f"Boston pass: {p.t_rise:6.1f} s to {p.t_set:6.1f} s, highest {p.max_elev_deg:.1f} deg at {p.t_max:.1f} s"
          + ("  (cut off by the start or end of the run)" if p.partial else ""))
    bookmarks.append(tl.Bookmark(t_s=p.t_max, label="highest over Boston"))
results_to_vizlog(results, log_path, overwrite=True, overlays=overlays, sites=[boston], bookmarks=bookmarks)

# Open the viewer (it blocks until you close its window). Without one set up,
# print how to open the log later.
try:
    view(log_path)
except ViewerNotFoundError as e:
    print(f"3-D viewer not available ({e}).\nOpen the log later with:  python -m ADCS.viz view {log_path}")

# --- 2-D analysis ---------------------------------------------------------
ADCS.plot(
    results,
    ADCS.plots.TargetPlot(),
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(),
    ADCS.plots.ControlPlotSingle(index=0, title="RW Control Torque"),
    layout=(2,2),
    title="Underactuated Control with MTQ and RW",
)
plt.show()
