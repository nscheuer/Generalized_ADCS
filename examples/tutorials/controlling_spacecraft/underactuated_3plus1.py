"""Track a ground target with three MTQs and one reaction wheel."""

import matplotlib.pyplot as plt
import numpy as np

import ADCS
from ADCS.helpers.animations import animate_orbit_godot
from ADCS.helpers.animations.godot.overlays import Arrow, TargetLine


wheel = ADCS.RW(axis=np.array([1.0, 0.0, 0.0]), max_torque=0.0023,
                J=5.7e-6, h=0.0, h_max=0.0036)
magnetorquers = [ADCS.MTQ(axis=axis, max_torque=0.2)
                for axis in np.eye(3)]
magnetometers = [ADCS.MTM(axis=axis) for axis in np.eye(3)]
satellite = ADCS.Satellite(
    mass=4.0, J_0=np.diag([0.03, 0.03, 0.01]),
    actuators=[wheel] + magnetorquers, sensors=magnetometers,
)
orbit = ADCS.Orbital_State(
    ephem=ADCS.Ephemeris(), J2000=0.22,
    R=np.array([5000.0, 0.0, 5000.0]), V=np.array([0.0, 7.5, 0.0]),
)
pitch = np.deg2rad(-75.0)
initial_state = ADCS.State(
    w=np.zeros(3),
    q=np.array([np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]),
    h=np.zeros(1),
)
controller = ADCS.controller.MTQ_w_RW_LP(
    est_sat=satellite, p_gain=5e-5, d_gain=0.002,
    c_gain=0.0, h_target=np.zeros(3),
)
goal = ADCS.goals.Coordinate_Goal(lat=42.0, lon=-80.0, alt=0.0)
results = ADCS.simulate(
    x=initial_state, satellite=satellite, os0=orbit,
    controller=controller, goal=goal, dt=1.0, tf=800.0,
)
ADCS.plot(results, ADCS.plots.TargetPlot(modes=["real_target"],
                                    title="Ground-target pointing error"),
          layout=(1, 1), figsize=(8, 4))
plt.savefig("underactuated_3plus1_angle_error.png", dpi=160)

animate_orbit_godot(
    results=results, model="mirata", graphics_quality="high",
    overlays=[
        Arrow("@boresight:default", label="Body boresight", color="#facc15",
              scale="fixed_px", px=120, width_px=5),
        TargetLine(color="#22c55e", length_px=150, width_px=5),
    ],
    camera_distance_m=20.0, camera_yaw_deg=0.0, camera_pitch_deg=45.0,
)
