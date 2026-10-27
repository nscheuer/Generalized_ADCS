"""Open a short orbit in the packaged Godot viewer.

Run: .venv/bin/python debug/debug_helpers/debug_animations/debug_orbit_godot_simulation.py
"""

import numpy as np

import ADCS
from ADCS.helpers.animations import animate_orbit_godot


def main() -> None:
    satellite = ADCS.Satellite(mass=4.0, J_0=np.diag([3.4, 2.9, 1.3]))
    state = ADCS.State(w=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]))
    orbital_state = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(), J2000=0.22,
        R=np.array([5000.0, 0.0, 5000.0]),
        V=np.array([0.0, -7.5, 0.0]),
    )
    run = ADCS.simulate(x=state, satellite=satellite, os0=orbital_state,
                        dt=0.1, tf=2.0).first()
    animate_orbit_godot(results=run)


if __name__ == "__main__":
    main()
