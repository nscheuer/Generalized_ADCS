"""Show how a rotating coarse Sun sensor responds to the Sun angle."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ADCS
from ADCS.helpers.animations import animate_orbit_godot
from ADCS.helpers.animations.godot.overlays import Arrow
from ADCS.helpers.math_helpers import rot_mat

environment = ADCS.Environment()
radius = 6878.1363  # 500 km altitude
orbit = ADCS.Orbital_State(
    environment=environment, J2000=0.22,
    R=np.array([radius, 0.0, 0.0]),
    V=np.array([0.0, np.sqrt(398600.4415 / radius), 0.0]),
)

# Start with the sensor pointed at the Sun and rotate the spacecraft away from it.
sun_axis = orbit.S - orbit.R
sun_axis /= np.linalg.norm(sun_axis)
spin_axis = np.cross(sun_axis, [0.0, 0.0, 1.0])
spin_axis /= np.linalg.norm(spin_axis)
sun_sensor = ADCS.SunSensor(axis=sun_axis, efficiency=1.0, sample_time=10.0)
satellite = ADCS.Satellite(
    mass=4.0, J_0=np.eye(3) * 0.02,
    sensors=[sun_sensor], disturbances=[],
)
results = ADCS.simulate(
    x=ADCS.State(w=spin_axis * (2.0 * np.pi / 2400.0), q=np.array([1.0, 0.0, 0.0, 0.0])),
    satellite=satellite, os0=orbit, environment=environment,
    dt=10.0, tf=6000.0,
)

# View from the anti-Sun side, offset so the Sun appears beside the spacecraft.
camera_yaw = np.degrees(np.arctan2(-sun_axis[1], -sun_axis[0])) + 12.0
camera_pitch = np.degrees(np.arcsin(-sun_axis[2]))

run = results.first()
time_min = np.asarray(run.time_s) / 60.0
reading = np.asarray(run.sensor_hist).reshape(-1)
sunlit = np.array([os.is_sunlit() for os in run.os_hist])
sun_direction = np.array([(os.S - os.R) / np.linalg.norm(os.S - os.R) for os in run.os_hist])
sensor_direction = np.array([rot_mat(state.q) @ sun_sensor.axis for state in run.state_hist])
angle_deg = np.degrees(np.arccos(np.clip(np.sum(sensor_direction * sun_direction, axis=1), -1.0, 1.0)))

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axes[0].plot(time_min, reading, color="#e6a700", label="Sun sensor reading")
axes[0].set(ylabel="Normalized reading", ylim=(-0.05, 1.05),
            title="Coarse Sun sensor during spacecraft rotation")
axes[0].legend(loc="upper right")
axes[1].plot(time_min, angle_deg, color="#2563a6", label="Sensor angle to Sun")
axes[1].axhline(90.0, color="#777777", linestyle="--", linewidth=1,
                label="Edge of sensor view")
axes[1].set(xlabel="Time [min]", ylabel="Angle to Sun [deg]", ylim=(0, 185))
axes[1].legend(loc="upper right")
for ax in axes:
    ax.fill_between(time_min, 0, 1, where=~sunlit, transform=ax.get_xaxis_transform(),
                    color="lightgray", alpha=0.35)
    ax.grid(alpha=0.3)
fig.tight_layout()

plt.show()

animate_orbit_godot(
    results=results, model="mirata", graphics_quality="high",
    camera_frame="inertial", camera_distance_m=30.0,
    camera_yaw_deg=camera_yaw, camera_pitch_deg=camera_pitch,
    overlays=[Arrow("@sun", label="Sun", color="#ffe066", scale="fixed_px", px=100)],
)
