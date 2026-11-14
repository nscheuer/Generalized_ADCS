"""Show the intermediate-axis flip in the Godot viewer."""

import numpy as np

import ADCS
from ADCS.helpers.animations import animate_orbit_godot
from ADCS.helpers.animations.godot.overlays import Triad

inertia = np.array([
    [0.0333, 0.0060, 0.0],
    [0.0060, 0.0333, 0.0],
    [0.0,    0.0,    0.00667],
])
satellite = ADCS.Satellite(mass=4.0, J_0=inertia)
initial_state = ADCS.State(
    w=np.deg2rad([3.5, -3.5, 0.1]),
    q=np.array([1.0, 0.0, 0.0, 0.0]),
)
orbit = ADCS.Orbital_State(
    ephem=ADCS.Ephemeris(),
    J2000=0.22,
    R=np.array([7000.0, 0.0, 0.0]),
    V=np.array([0.0, 7.5, 0.0]),
)
results = ADCS.simulate(
    x=initial_state, satellite=satellite, os0=orbit,
    dt=1.0, tf=180.0,
)

animate_orbit_godot(
    results=results,
    overlays=[Triad(frame="body", px=110)],
    graphics_quality="high",
)
