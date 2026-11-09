"""Detumble a 3U spacecraft with B-dot magnetorquer control."""

import numpy as np
import ADCS
from ADCS.helpers.animations import animate_orbit_godot


environment = ADCS.Environment(magnetic_field=ADCS.IGRF14())
satellite = ADCS.Satellite(
    mass=4.0, J_0=np.diag([0.0333, 0.00667, 0.0333]),
    actuators=[ADCS.MTQ(axis, max_torque=0.2) for axis in np.eye(3)],
    sensors=[ADCS.MTM(axis) for axis in np.eye(3)],
)
orbit = ADCS.Orbital_State(
    environment=environment, J2000=0.22,
    R=np.array([6878.1363, 0.0, 0.0]),
    V=np.array([0.0, 7.6126, 0.0]),
)
results = ADCS.simulate(
    x=ADCS.State(w=np.array([0.0, 0.08, 0.0]), q=np.array([1.0, 0.0, 0.0, 0.0])),
    satellite=satellite, controller=ADCS.controller.BDot(satellite, gain=5e4),
    os0=orbit, environment=environment, dt=5.0, tf=5400.0,
)

ADCS.plot(results, ADCS.plots.AngularVelocityPlotCombined(sources=["real"]), layout=(1, 1))
animate_orbit_godot(results=results, model="mirata", overlays=[], graphics_quality="high")
