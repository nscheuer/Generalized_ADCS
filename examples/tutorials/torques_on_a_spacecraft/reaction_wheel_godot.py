"""Accelerate one reaction wheel, then brake it; the 3U body reacts."""

import numpy as np
import ADCS
from ADCS.helpers.animations import animate_orbit_godot


class SpinWheel(ADCS.controller.Controller):
    def __init__(self, satellite):
        super().__init__(satellite)
        self.step = 0

    def find_u(self, x_hat, **_):
        self.step += 1
        return np.array([-3e-5 if self.step <= 30 else 3e-5 if self.step <= 60 else 0.0])


wheel = ADCS.RW(axis=np.array([0.0, 1.0, 0.0]), max_torque=3e-5,
                J=5e-6, h=0.0, h_max=0.003)
satellite = ADCS.Satellite(
    mass=4.0, J_0=np.diag([0.0333, 0.00667, 0.0333]),
    actuators=[wheel],
)
orbit = ADCS.Orbital_State(
    ephem=ADCS.Ephemeris(), J2000=0.22,
    R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]),
)
results = ADCS.simulate(
    x=ADCS.State(w=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]), h=np.array([0.0])),
    satellite=satellite, os0=orbit, controller=SpinWheel(satellite), dt=1.0, tf=61.0,
)

ADCS.plot(results, ADCS.plots.WheelSpeedPlot(),
          ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
          layout=(2, 1), figsize=(8, 6))
animate_orbit_godot(results=results, overlays=[], graphics_quality="high")
