import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import ADCS as ADCS
import numpy as np
import matplotlib.pyplot as plt

gps_noise = ADCS.Noise(std_noise=np.array([5, 5, 5, 0.1, 0.1, 0.1]))  # km, km/s
gps = [ADCS.GPS(noise=gps_noise)]
gps_noise = ADCS.Noise(std_noise=np.array([3, 3, 3, 0.1, 0.1, 0.1]))  # km, km/s
est_gps = [ADCS.GPS(noise=gps_noise)]

satellite = ADCS.Satellite(mass=28.9, J_0=np.diag([0.34, 0.27, 0.30]), sensors=gps)
est_satellite = ADCS.EstimatedSatellite(mass=28.9, J_0=np.diag([0.34, 0.27, 0.30]), sensors=est_gps)
x_0 = ADCS.State.from_array(np.array([0.00023, 0.0, 0.000065, 1, 0, 0, 0]))

os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(),J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 7.5, 1]))
est_os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(),J2000=0.22, R=np.array([5000, 5, -5]), V=np.array([0.1, 4, 1.1]))

P0 = np.diag([500**2.0, 500**2.0, 500**2.0, 0.5**2.0, 0.5**2.0, 0.5**2.0])    # initial covariance
Q0 = np.diag([1, 1, 1, 10, 10, 10])

orbit_estimator = ADCS.Orbit_EKF(est_sat=est_satellite, J2000=0.22, os_hat=os0, P_hat=P0, Q_hat=Q0, dt=20.0)

results = ADCS.simulate(
    x=x_0,
    satellite=satellite,
    orbit_estimator=orbit_estimator,
    os0=os0,
    dt=20.0,
    tf=1000.0
)

from ADCS.helpers.plot import plot
from ADCS.helpers.plot import OrbitPositionPlot, OrbitPositionPlotSingle, OrbitPositionPlotCombined

plot(
    results,
    OrbitPositionPlot(sources=["real", "estimated"]),
    OrbitPositionPlotSingle(sources=["real", "estimated"], component="x"),
    OrbitPositionPlotCombined(sources=["real", "estimated"]),
    layout=(3,1),
    title="Orbit Position",
)

from ADCS.helpers.plot import OrbitVelocityPlot, OrbitVelocityPlotSingle, OrbitVelocityPlotCombined, OrbitPlot

plot(
    results,
    OrbitVelocityPlot(sources=["real", "estimated"]),
    OrbitVelocityPlotSingle(sources=["real", "estimated"], component="x"),
    OrbitVelocityPlotCombined(sources=["real", "estimated"]),
    layout=(3,1),
    title="Orbit Velocity",
)

plot(
    results,
    OrbitPlot(sources=["real", "estimated"]),
    layout=(1,1),
    title="3D Orbit Plot",
)

plt.show()