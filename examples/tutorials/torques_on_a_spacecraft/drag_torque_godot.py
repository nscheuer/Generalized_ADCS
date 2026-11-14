"""Animate a drag-driven 3U spacecraft in the Godot viewer."""

import numpy as np

import ADCS
from ADCS.helpers.animations import animate_orbit_godot
from ADCS.helpers.animations.godot.overlays import Arrow, Triad

environment = ADCS.Environment(atmosphere=ADCS.SMADAtmosphere())

size = np.array([0.1, 0.3, 0.1])
faces = []
for axis in range(3):
    for sign in (-1.0, 1.0):
        normal = sign * np.eye(3)[axis]
        faces.append(ADCS.disturbances.GeometryFace(
            area=np.prod(np.delete(size, axis)),
            centroid=0.5 * size[axis] * normal,
            normal=normal, CD=2.2,
        ))
geometry = ADCS.disturbances.GeometryConfig(faces)
drag = ADCS.disturbances.Drag_Disturbance(geometry)
satellite = ADCS.Satellite(
    mass=4.0,
    COM=np.array([0.0, 0.02, 0.0]),
    J_0=np.diag([0.0333, 0.00667, 0.0333]),
    disturbances=[drag],
)
initial_state = ADCS.State(
    w=np.zeros(3),
    q=np.array([np.cos(np.pi / 8), 0.0, 0.0, -np.sin(np.pi / 8)]),
)
orbit = ADCS.Orbital_State(
    environment=environment, J2000=0.22,
    R=np.array([6678.1363, 0.0, 0.0]),
    V=np.array([0.0, np.sqrt(398600.4415 / 6678.1363), 0.0]),
)
results = ADCS.simulate(
    x=initial_state, satellite=satellite, os0=orbit,
    environment=environment, dt=5.0, tf=390.0,
)

animate_orbit_godot(
    results=results,
    overlays=[
        Triad(frame="body", px=110),
        Arrow("@velocity", color="#facc15", scale="fixed_px", px=150),
    ],
    graphics_quality="high",
)
