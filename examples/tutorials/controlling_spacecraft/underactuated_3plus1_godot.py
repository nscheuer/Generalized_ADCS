"""Track a ground target with three MTQs and one reaction wheel."""

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
epoch = 0.22
ephemeris = ADCS.Ephemeris()
# Start on Earth's daylight hemisphere, then place the fixed target a few
# degrees away from the subsatellite point so the line of sight is oblique.
earth_frame = ADCS.Orbital_State(
    ephem=ephemeris, J2000=epoch,
    R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]),
)
target = ADCS.goals.Coordinate_Goal(lat=-16.0, lon=-160.0, alt=0.0)
subpoint = ADCS.goals.Coordinate_Goal(lat=-22.0, lon=-167.0, alt=700.0)
position_eci = earth_frame.ecef_to_eci(subpoint.target_ecef)
_, lon_rad = np.deg2rad([-22.0, -167.0])
east_ecef = np.array([-np.sin(lon_rad), np.cos(lon_rad), 0.0])
velocity_eci = earth_frame.ecef_to_eci(east_ecef)
radial_hat = position_eci / np.linalg.norm(position_eci)
velocity_eci -= np.dot(velocity_eci, radial_hat) * radial_hat
velocity_eci = velocity_eci / np.linalg.norm(velocity_eci) * 7.5
orbit = ADCS.Orbital_State(
    ephem=ephemeris, J2000=epoch, R=position_eci, V=velocity_eci,
)
controller = ADCS.controller.MTQ_w_RW_LP(
    est_sat=satellite, p_gain=5e-5, d_gain=0.002,
    c_gain=0.0, h_target=np.zeros(3),
)
goal_direction = target.to_ref(orbit)[0][1:]
goal_direction /= np.linalg.norm(goal_direction)
orbit_normal = np.cross(orbit.R, orbit.V)
orbit_normal /= np.linalg.norm(orbit_normal)
initial_error = np.deg2rad(35.0)
boresight_direction = (
    np.cos(initial_error) * goal_direction
    + np.sin(initial_error) * np.cross(orbit_normal, goal_direction)
)
boresight_direction /= np.linalg.norm(boresight_direction)
q_scalar = np.sqrt((1.0 + boresight_direction[2]) / 2.0)
q_vector = np.cross([0.0, 0.0, 1.0], boresight_direction) / (2.0 * q_scalar)
initial_state = ADCS.State(
    w=np.zeros(3), q=np.r_[q_scalar, q_vector], h=np.zeros(1),
)
results = ADCS.simulate(
    x=initial_state, satellite=satellite, os0=orbit,
    controller=controller, goal=target, dt=1.0, tf=800.0,
)
animate_orbit_godot(
    results=results, graphics_quality="high",
    overlays=[
        Arrow("@boresight:default", label="Body boresight", color="#facc15",
              scale="fixed_px", px=120, width_px=5),
        TargetLine(color="#22c55e", length_px=150, width_px=5),
    ],
    camera_distance_m=20.0, camera_yaw_deg=0.0, camera_pitch_deg=45.0,
)
