"""Open a short, controlled pointing run in the packaged Godot viewer.

Run: .venv/bin/python debug/debug_helpers/debug_animations/debug_orbit_godot_simulation.py
"""

import numpy as np

import ADCS
from ADCS.helpers.animations import animate_orbit_godot


def main() -> None:
    # Three-axis magnetorquers and reaction wheels give the controller enough
    # authority to follow the changing goals during the short run.
    mtqs = [ADCS.MTQ(axis=axis, max_torque=0.1) for axis in np.eye(3)]
    rws = [ADCS.RW(axis=axis, max_torque=4.51, J=0.22, h=0.0, h_max=3.8)
           for axis in np.eye(3)]
    mtms = [ADCS.MTM(axis=axis) for axis in np.eye(3)]
    satellite = ADCS.Satellite(
        mass=4.0,
        J_0=np.diag([3.4, 2.9, 1.3]),
        actuators=mtqs + rws,
        sensors=mtms,
        boresight=np.array([0.0, 0.0, 1.0]),
    )
    state = ADCS.State(
        w=np.zeros(3),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        h=np.zeros(3),
    )

    controller = ADCS.controller.MTQ_w_RW(
        est_sat=satellite,
        p_gain=0.04,
        d_gain=0.7,
        c_gain=0.1,
        h_target=np.zeros(3),
    )

    # Near-circular 500 km orbit, inclined so ground-target pointing changes
    # meaningfully as the spacecraft passes over Earth.
    orbital_radius_km = 6378.137 + 500.0
    inclination = np.deg2rad(51.6)
    orbital_speed_km_s = np.sqrt(398600.4418 / orbital_radius_km)
    orbital_state = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(), J2000=0.22,
        R=np.array([orbital_radius_km, 0.0, 0.0]),
        V=np.array([0.0, orbital_speed_km_s * np.cos(inclination),
                    orbital_speed_km_s * np.sin(inclination)]),
    )

    goal_timeline = {
        0.0: ADCS.goals.ECI_Goal(eci_vector=np.array([0.0, 0.0, 1.0])),
        400.0: ADCS.goals.Coordinate_Goal(
            lat=33.75, lon=-84.3885, alt=0.0),
        800.0: ADCS.goals.AntiVelocity_Goal(),
    }
    goals = ADCS.GoalList(
        goal_timeline=goal_timeline,
        time_units="seconds",
        start_juliantime=orbital_state.J2000,
    )

    results = ADCS.simulate(
        x=state,
        satellite=satellite,
        controller=controller,
        goal=goals,
        os0=orbital_state,
        dt=2.0,
        tf=1200.0,
    )
    run = results.first()
    animate_orbit_godot(results=run)


if __name__ == "__main__":
    main()
