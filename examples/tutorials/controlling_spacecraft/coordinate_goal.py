"""Track a fixed ground location with MiRaTa."""

import numpy as np

import ADCS
from ADCS.helpers.animations import animate_orbit_godot
from ADCS.helpers.animations.godot.overlays import Arrow, TargetLine
from ADCS.helpers.math_helpers import quat_mult


axes = np.array([[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1]], dtype=float)
axes /= np.linalg.norm(axes, axis=1)[:, None]
wheels = [ADCS.RW(axis=axis, max_torque=1e-3, J=5e-6,
                  h=0.0, h_max=0.02) for axis in axes]
magnetorquers = [ADCS.MTQ(axis=axis, max_torque=0.1) for axis in np.eye(3)]
magnetometers = [ADCS.MTM(axis=axis) for axis in np.eye(3)]
satellite = ADCS.Satellite(
    mass=4.0, J_0=np.diag([0.0333, 0.0333, 0.00667]),
    actuators=magnetorquers + wheels, sensors=magnetometers,
)
orbit = ADCS.Orbital_State(
    ephem=ADCS.Ephemeris(), J2000=0.22,
    R=np.array([0.0, -7000.0, 0.0]), V=np.array([7.5, 0.0, 0.0]),
)
yaw = np.array([np.cos(np.pi / 6), 0.0, 0.0, np.sin(np.pi / 6)])
pitch = np.array([np.cos(np.pi / 3), 0.0, np.sin(np.pi / 3), 0.0])
initial_state = ADCS.State(
    w=np.zeros(3), q=quat_mult(yaw, pitch), h=np.zeros(len(wheels)),
)
controller = ADCS.controller.MTQ_w_RW(
    est_sat=satellite, p_gain=1e-4, d_gain=5e-3,
    c_gain=0.0, h_target=np.zeros(3),
)
goal = ADCS.goals.Coordinate_Goal(lat=0.0, lon=-176.6, alt=0.0)
results = ADCS.simulate(
    x=initial_state, satellite=satellite, os0=orbit,
    controller=controller, goal=goal, dt=1.0, tf=480.0,
)
animate_orbit_godot(
    results=results, model="mirata", graphics_quality="high",
    overlays=[
        Arrow("@boresight:default", label="Body boresight", color="#facc15",
              scale="fixed_px", px=120, width_px=5),
        TargetLine(color="#22c55e", length_px=150, width_px=5),
    ],
    camera_distance_m=20.0, camera_pitch_deg=60.0,
)
