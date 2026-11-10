"""Use GPS position and velocity to recover the spacecraft orbit."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ADCS


np.random.seed(7)
gps = ADCS.GPS(
    sample_time=20.0,
    noise=ADCS.Noise(std_noise=np.array([0.5, 0.5, 0.5, 0.001, 0.001, 0.001])),
)
satellite = ADCS.Satellite(mass=4.0, J_0=np.eye(3) * 0.02, sensors=[gps])
estimated = ADCS.EstimatedSatellite(mass=4.0, J_0=np.eye(3) * 0.02, sensors=[gps])
orbit = ADCS.Orbital_State(
    ephem=ADCS.Ephemeris(), J2000=0.22,
    R=np.array([6878.1363, 0.0, 0.0]), V=np.array([0.0, 7.6, 0.0]),
)
results = ADCS.simulate(
    x=ADCS.State(w=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])),
    satellite=satellite, os0=orbit,
    orbit_estimator=ADCS.Orbit_GPS(est_sat=estimated, J2000=orbit.J2000, os_template=orbit),
    dt=20.0, tf=6000.0,
)

ADCS.plot(results, ADCS.plots.OrbitPositionPlot(sources=["real", "estimated"]),
          layout=(1, 1), figsize=(9, 7), title="Orbit position from GPS")
plot_path = Path(__file__).resolve().parents[3] / "docs/source/_static/tutorials/attitude_sensors/gps.png"
plt.savefig(plot_path, dpi=160, bbox_inches="tight")
plt.close()
