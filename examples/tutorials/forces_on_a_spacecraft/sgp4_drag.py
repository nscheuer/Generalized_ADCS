"""Compare SGP4 orbits with geometry-derived B* and zero B*."""

import matplotlib.pyplot as plt
import numpy as np

import ADCS
from ADCS.orbits.sgp4_orbit import SGP4Orbit
from ADCS.orbits.universal_constants import EarthConstants, TimeConstants


size = np.array([0.1, 0.3, 0.1])  # 3U spacecraft, metres
faces = [
    ADCS.disturbances.GeometryFace(
        area=np.prod(np.delete(size, axis)),
        centroid=0.5 * size[axis] * sign * np.eye(3)[axis],
        normal=sign * np.eye(3)[axis], CD=2.2,
    )
    for axis in range(3) for sign in (-1.0, 1.0)
]
drag = ADCS.disturbances.Drag_Disturbance(
    ADCS.disturbances.GeometryConfig(faces)
)
satellite = ADCS.Satellite(
    mass=4.0, J_0=np.diag([0.0333, 0.00667, 0.0333]),
    disturbances=[drag],
)

radius = EarthConstants.R_e + 300.0
speed = np.sqrt(EarthConstants.mu_e / radius)
orbit_start = ADCS.Orbital_State(
    ephem=ADCS.Ephemeris(), J2000=0.22,
    R=np.array([radius, 0.0, 0.0]),
    V=np.array([0.0, speed * np.cos(np.deg2rad(50)),
                speed * np.sin(np.deg2rad(50))]),
)
end_time = orbit_start.J2000 + 2 * 86400 * TimeConstants.sec2cent
with_drag = SGP4Orbit(orbit_start, end_time=end_time, dt=60.0,
                      satellite=satellite, verbose=False)
without_drag = SGP4Orbit(orbit_start, end_time=end_time, dt=60.0,
                         bstar=0.0, verbose=False)

print(f"Estimated B*: {with_drag.bstar:.4g} / Earth radius")
print(with_drag.tle[0])  # TLE line 1 contains the B* field

run = ADCS.RunResults(satellite=satellite, os_hist=list(with_drag.states.values()))
ADCS.plot(run, ADCS.plots.OrbitPlot(), layout=(1, 1), title="SGP4 orbit with drag")
plt.savefig("sgp4_drag_orbit.png", dpi=160)

days = (with_drag.times - with_drag.times[0]) * TimeConstants.cent2sec / 86400
a_drag = np.array([state.to_keplerian()[0] for state in with_drag.states.values()])
a_zero = np.array([state.to_keplerian()[0] for state in without_drag.states.values()])
plt.figure()
plt.plot(days, a_drag - a_zero)
plt.xlabel("Time [days]")
plt.ylabel("Semi-major axis difference [km]")
plt.title("Effect of SGP4 B* relative to B* = 0")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("sgp4_drag_effect.png", dpi=160)
plt.show()
