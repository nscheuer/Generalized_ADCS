"""Compare two-body, J2, and higher-zonal orbit propagation."""

import matplotlib.pyplot as plt
import numpy as np

import ADCS
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import TimeConstants

orbit_start = ADCS.Orbital_State(
    ephem=ADCS.Ephemeris(), J2000=0.22,
    R=np.array([6878.0, 0.0, 0.0]),
    V=np.array([0.0, 4.9, 5.8]),
)

for degree in (0, 2, 6):
    orbit = Orbit(
        os0=orbit_start,
        end_time=orbit_start.J2000 + 2 * 86400 * TimeConstants.sec2cent,
        dt=60.0, zonal_J=degree, verbose=False,
    )
    run = ADCS.RunResults(satellite=None, os_hist=list(orbit.states.values()))
    ADCS.plot(run, ADCS.plots.OrbitPlot(), layout=(1, 1),
              title=f"Orbit with zonal_J={degree}")
    plt.savefig(f"j{degree}_orbit.png", dpi=160)

plt.show()
