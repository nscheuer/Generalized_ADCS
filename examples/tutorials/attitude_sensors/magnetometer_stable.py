"""Show orbital magnetic-field variation with a fixed inertial attitude."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[3]))
import ADCS

environment = ADCS.Environment(magnetic_field=ADCS.IGRF14())
magnetometers = [ADCS.MTM(axis=axis, sample_time=20.0) for axis in np.eye(3)]
satellite = ADCS.Satellite(
    mass=4.0, J_0=np.diag([0.0333, 0.0333, 0.00667]),
    sensors=magnetometers, disturbances=[],
)
radius = 6878.1363  # 500 km altitude
orbit = ADCS.Orbital_State(
    environment=environment, J2000=0.22,
    R=np.array([0.0, -radius, 0.0]),
    V=np.array([np.sqrt(398600.4415 / radius), 0.0, 0.0]),
)
results = ADCS.simulate(
    x=ADCS.State(w=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])),
    satellite=satellite, os0=orbit, environment=environment,
    dt=20.0, tf=6000.0,
)

run = results.runs[0]
time_min = np.asarray(run.time_s) / 60.0
field_nt = np.vstack(run.clean_sensor_hist) * 1e9
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for i, axis in enumerate("xyz"):
    axes[0].plot(time_min, field_nt[:, i], label=f"$B_{axis}$")
axes[0].set_ylabel("Body-frame field [nT]")
axes[0].set_title("Magnetic field over one orbit · fixed inertial attitude")
axes[0].legend(ncol=3)
axes[0].grid(alpha=0.3)
axes[1].plot(time_min, np.linalg.norm(field_nt, axis=1), color="black")
axes[1].set_ylabel("Field magnitude [nT]")
axes[1].set_xlabel("Time [min]")
axes[1].grid(alpha=0.3)
fig.tight_layout()

plt.show()
