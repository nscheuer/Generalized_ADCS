"""Animate a spacecraft initially at rest under residual magnetic torque."""

import numpy as np
import ADCS
from ADCS.helpers.animations import animate_orbit_godot
from ADCS.helpers.animations.godot.overlays import Arrow, Triad

environment = ADCS.Environment(magnetic_field=ADCS.IGRF14())
dipole = ADCS.disturbances.Dipole_Disturbance(
    dipole_torque=np.array([0.01, 0.0, 0.0]),  # A m^2, body frame
)
satellite = ADCS.Satellite(
    mass=4.0,
    J_0=np.diag([0.0333, 0.00667, 0.0333]),
    disturbances=[dipole],
)
initial_state = ADCS.State(
    q=np.array([1.0, 0.0, 0.0, 0.0]),
    w=np.zeros(3),
)
orbit = ADCS.Orbital_State(
    environment=environment, J2000=0.22,
    R=np.array([6878.1363, 0.0, 0.0]),
    V=np.array([0.0, np.sqrt(398600.4415 / 6878.1363), 0.0]),
)
results = ADCS.simulate(
    x=initial_state, satellite=satellite, os0=orbit,
    environment=environment, dt=5.0, tf=900.0,
)

animate_orbit_godot(
    results=results, model="mirata",
    overlays=[
        Triad(frame="body", px=110),
        Arrow("b_field", color="#facc15", scale="fixed_px", px=150),
    ],
    graphics_quality="high",
)
