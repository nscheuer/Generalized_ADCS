"""Show a four-minute reaction-wheel slew and ground-target tracking pass.

Run with: venv/bin/python debug/debug_helpers/debug_animations/debug_orbit_panda3d.py
"""

import numpy as np

import ADCS
from ADCS.CONOPS.goals import Coordinate_Goal
from ADCS.controller import MTQ_w_RW
from ADCS.helpers.animations import animate_orbit_panda3d
from ADCS.satellite_hardware.actuators import RW
from ADCS.satellite_hardware.sensors import MTM


def build_ground_tracking_run(duration_s: float = 240.0, dt: float = 1.0):
    """Simulate a CubeSat slewing onto an equatorial ground site and tracking it."""
    goal = Coordinate_Goal(lat=0.0, lon=0.0, alt=0.0)
    ephemeris = ADCS.Ephemeris()
    epoch = 0.22

    # Put the spacecraft over the site at the start of the pass. Its +Z
    # boresight initially points sideways, providing a visible 90-degree slew.
    reference = ADCS.Orbital_State(
        ephem=ephemeris, J2000=epoch,
        R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]),
    )
    radial = reference.ecef_to_eci(goal.target_ecef)
    radial /= np.linalg.norm(radial)
    os0 = ADCS.Orbital_State(
        ephem=ephemeris, J2000=epoch,
        R=7000.0 * radial, V=7.5 * np.cross([0.0, 0.0, 1.0], radial),
    )

    axes = np.eye(3)
    wheels = [RW(axis=axis, max_torque=0.002, J=0.0001, h=0.0, h_max=0.2)
              for axis in axes]
    # This wheel controller expects magnetic-field readings, although this
    # example has no magnetorquers and commands only the three wheels.
    magnetometers = [MTM(axis=axis) for axis in axes]
    satellite = ADCS.Satellite(
        mass=1.5, J_0=np.diag([0.02, 0.02, 0.025]),
        actuators=wheels, sensors=magnetometers,
        boresight=np.array([0.0, 0.0, 1.0]),
    )
    controller = MTQ_w_RW(
        est_sat=satellite, p_gain=0.002, d_gain=0.018, c_gain=0.0,
        h_target=np.zeros(3),
    )
    initial_state = ADCS.State(
        w=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]), h=np.zeros(3),
    )
    run = ADCS.simulate(
        x=initial_state, satellite=satellite, controller=controller,
        goal=goal, os0=os0, dt=dt, tf=duration_s,
    ).first()
    return run, goal


def main() -> None:
    run, goal = build_ground_tracking_run()
    print('Tracking ground target at 0° latitude, 0° longitude for 240 seconds.')
    animate_orbit_panda3d(
        time_hist=run.time_s, state_hist=run.state_hist, os_hist=run.os_hist,
        coord_goal=goal,
    )


if __name__ == "__main__":
    main()
