"""Show fast body-frame magnetic-field variation during a Dzhanibekov tumble."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[3]))
import ADCS


environment = ADCS.Environment(magnetic_field=ADCS.IGRF14())
inertia = np.array([
    [0.0333, 0.0060, 0.0],
    [0.0060, 0.0333, 0.0],
    [0.0, 0.0, 0.00667],
])
magnetometers = [ADCS.MTM(axis=axis, sample_time=1.0) for axis in np.eye(3)]
satellite = ADCS.Satellite(
    mass=4.0, J_0=inertia, sensors=magnetometers, disturbances=[],
)
orbit = ADCS.Orbital_State(
    environment=environment, J2000=0.22,
    R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]),
)
results = ADCS.simulate(
    x=ADCS.State(
        w=np.deg2rad([3.5, -3.5, 0.1]), q=np.array([1.0, 0.0, 0.0, 0.0]),
    ),
    satellite=satellite, os0=orbit, environment=environment,
    dt=1.0, tf=180.0,
)

run = results.runs[0]
field_nt = np.vstack(run.clean_sensor_hist) * 1e9
fig, ax = plt.subplots(figsize=(9, 4.5))
for i, axis in enumerate("xyz"):
    ax.plot(run.time_s, field_nt[:, i], label=f"$B_{axis}$")
ax.set_title("Body-frame magnetic field during the tumble")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Magnetic field [nT]")
ax.legend(ncol=3)
ax.grid(alpha=0.3)
fig.tight_layout()
plot_path = Path(__file__).resolve().parents[3] / "docs/source/_static/tutorials/attitude_sensors/magnetometer_tumbling.png"
fig.savefig(plot_path, dpi=160, bbox_inches="tight")
plt.close(fig)
