"""Point a 1U CubeSat with magnetorquers alone over several orbits."""

import matplotlib.pyplot as plt
import numpy as np

import ADCS
from ADCS.helpers.animations import animate_orbit_godot
from ADCS.helpers.animations.godot.overlays import Arrow, TargetLine

magnetorquers = [ADCS.MTQ(axis=axis, max_torque=0.005)
                for axis in np.eye(3)]
magnetometers = [ADCS.MTM(axis=axis) for axis in np.eye(3)]
satellite = ADCS.Satellite(
    mass=4.0, J_0=np.diag([0.0333, 0.0333, 0.00667]),
    actuators=magnetorquers, sensors=magnetometers,
)
orbit = ADCS.Orbital_State(
    ephem=ADCS.Ephemeris(), J2000=0.22,
    R=np.array([5000.0, 0.0, 5000.0]), V=np.array([0.0, 7.5, 0.0]),
)
initial_state = ADCS.State(
    w=np.zeros(3),
    q=np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]),
)
controller = ADCS.controller.MTQ_Lovera(
    est_sat=satellite, p_gain=5e-7, d_gain=1e-3, eps=1.0,
)
goal = ADCS.goals.ECI_Goal(eci_vector=np.array([0.0, 0.0, 1.0]))
results = ADCS.simulate(
    x=initial_state, satellite=satellite, os0=orbit,
    controller=controller, goal=goal, dt=5.0, tf=18000.0,
)
ADCS.plot(results, ADCS.plots.TargetPlot(modes=["real_target"],
                                    title="Boresight error"),
          layout=(1, 1), figsize=(8, 4))

animate_orbit_godot(
    results=results, model="cubesat_1ru", graphics_quality="high",
    overlays=[
        Arrow("@boresight:default", label="Body boresight", color="#facc15",
              scale="fixed_px", px=120, width_px=5),
        TargetLine(color="#22c55e", length_px=150, width_px=5),
    ],
    camera_distance_m=12.0, camera_yaw_deg=0.0, camera_pitch_deg=45.0,
)
