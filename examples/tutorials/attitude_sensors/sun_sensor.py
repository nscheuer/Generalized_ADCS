"""Show a coarse Sun sensor response through sunlight and eclipse."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ADCS
from ADCS.helpers.animations import animate_orbit_godot
from ADCS.helpers.animations.godot.overlays import Arrow, SensorCone


environment = ADCS.Environment()
radius = 6878.1363  # 500 km altitude
orbit = ADCS.Orbital_State(
    environment=environment, J2000=0.22,
    R=np.array([radius, 0.0, 0.0]),
    V=np.array([0.0, np.sqrt(398600.4415 / radius), 0.0]),
)

# Point the sensor toward the Sun at the start, then keep the body inertially fixed.
sun_axis = orbit.S - orbit.R
sun_axis /= np.linalg.norm(sun_axis)
sun_sensor = ADCS.SunSensor(axis=sun_axis, efficiency=1.0, sample_time=10.0)
satellite = ADCS.Satellite(
    mass=4.0, J_0=np.diag([0.0333, 0.0333, 0.00667]),
    sensors=[sun_sensor], disturbances=[],
)
results = ADCS.simulate(
    x=ADCS.State(w=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])),
    satellite=satellite, os0=orbit, environment=environment,
    dt=10.0, tf=1500.0,
)

animate_orbit_godot(
    results=results, model="mirata", graphics_quality="high",
    overlays=[
        Arrow("@sun", label="Sun", color="#ffe066", scale="fixed_px", px=100),
        SensorCone("sunsensor0", color="#66ccff", opacity=0.3),
    ],
)

run = results.first()
time_min = np.asarray(run.time_s) / 60.0
reading = np.asarray(run.clean_sensor_hist).reshape(-1)
sunlit = np.array([os.is_sunlit() for os in run.os_hist])

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(time_min, reading, color="#e6a700", label="Sun sensor reading")
ax.fill_between(
    time_min, 0.0, 1.0, where=~sunlit,
    color="lightgray", alpha=0.6, label="Eclipse",
)
ax.set(xlabel="Time [min]", ylabel="Normalized reading",
       ylim=(-0.05, 1.05), title="Coarse Sun sensor · eclipse blanks the measurement")
ax.grid(alpha=0.3)
ax.legend(loc="best")
fig.tight_layout()
plot_path = Path(__file__).resolve().parents[3] / "docs/source/_static/tutorials/attitude_sensors/sun_sensor_eclipse.png"
fig.savefig(plot_path, dpi=160, bbox_inches="tight")
plt.close(fig)
