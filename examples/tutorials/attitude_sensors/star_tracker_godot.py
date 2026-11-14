"""Measure attitude from stars and show when the solution disappears."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ADCS
from ADCS.helpers.animations import animate_orbit_godot
from ADCS.helpers.math_helpers import quaternion_angle

np.random.seed(7)
environment = ADCS.Environment()
radius = 6878.1363  # 500 km altitude
orbit = ADCS.Orbital_State(
    environment=environment, J2000=0.22,
    R=np.array([radius, 0.0, 0.0]),
    V=np.array([0.0, np.sqrt(398600.4415 / radius), 0.0]),
)
sun = orbit.S - orbit.R
sun /= np.linalg.norm(sun)
spin_axis = np.cross([0.0, 0.0, 1.0], sun)
spin_axis /= np.linalg.norm(spin_axis)
tracker = ADCS.StarTrackerQuaternion(
    boresight=np.array([0.0, 0.0, 1.0]), fov=np.deg2rad(60.0),
    sun_exclusion=np.deg2rad(35.0), min_stars=2, sample_time=20.0,
    noise=ADCS.Noise(std_noise=np.full(4, np.deg2rad(20.0 / 3600.0))),
)
satellite = ADCS.Satellite(mass=4.0, J_0=np.eye(3) * 0.02, sensors=[tracker])
results = ADCS.simulate(
    x=ADCS.State(w=spin_axis * (2.0 * np.pi / 1200.0), q=np.array([1.0, 0.0, 0.0, 0.0])),
    satellite=satellite, os0=orbit, environment=environment, dt=20.0, tf=1800.0,
)

run = results.first()
time_min = np.asarray(run.time_s) / 60.0
measured = np.asarray(run.sensor_hist)
valid = np.all(np.isfinite(measured), axis=1)
error_arcmin = np.array([
    np.rad2deg(quaternion_angle(state.q, q)) * 60.0 if ok else np.nan
    for state, q, ok in zip(run.state_hist, measured, valid)
])
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(time_min, error_arcmin, ".", color="#2563a6", label="Attitude error")
ax.plot(time_min[~valid], np.zeros(np.count_nonzero(~valid)), "x",
        color="#888888", label="No solution")
ax.set(xlabel="Time [min]", ylabel="Attitude error [arcmin]",
       title="Star tracker attitude measurement", ylim=(-0.1, 2.3))
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

plt.show()

animate_orbit_godot(results=results, graphics_quality="high", overlays=[])
