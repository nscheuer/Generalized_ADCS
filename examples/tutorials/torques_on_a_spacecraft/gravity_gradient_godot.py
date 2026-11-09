"""Compare stable and unstable gravity-gradient orientations of a 3U satellite."""

import numpy as np
import ADCS
from ADCS.helpers.animations import animate_orbit_godot
from ADCS.helpers.animations.godot.overlays import Arrow, Triad


environment = ADCS.Environment()
gravity_gradient = ADCS.disturbances.GG_Disturbance()
satellite = ADCS.Satellite(
    mass=4.0,
    COM=np.zeros(3),
    J_0=np.diag([0.0333, 0.00667, 0.0333]),
    disturbances=[gravity_gradient],
)
radius = 6878.1363  # km: 500 km altitude
mu = 398600.4415  # km^3/s^2
orbital_rate = np.sqrt(mu / radius**3)
orbit = ADCS.Orbital_State(
    environment=environment, J2000=0.22,
    R=np.array([radius, 0.0, 0.0]),
    V=np.array([0.0, np.sqrt(mu / radius), 0.0]),
)

runs = {}
for label, angle_deg in (("Long axis near nadir", -80.0),
                         ("Wide axis near nadir", 10.0)):
    angle = np.deg2rad(angle_deg)
    initial_state = ADCS.State(
        q=np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]),
        w=np.array([0.0, 0.0, orbital_rate]),
    )
    results = ADCS.simulate(
        x=initial_state, satellite=satellite, os0=orbit,
        environment=environment, dt=20.0, tf=6000.0,
    )
    runs[label] = results.first()
    animate_orbit_godot(
        results=results, model="mirata",
        overlays=[
            Triad(frame="body", px=110),
            Arrow("@nadir", color="#facc15", scale="fixed_px", px=150),
        ],
        graphics_quality="high",
    )
