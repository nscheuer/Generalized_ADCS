"""Show how aerodynamic drag torque changes the attitude of a 3U spacecraft."""

import matplotlib.pyplot as plt
import numpy as np

import ADCS

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

run = results.first()
time = np.asarray(run.time_s) / 60

def angle_to_velocity(states):
    velocity = np.array([
        orbital_state.get_state_vector(x=state)["v"]
        for state, orbital_state in zip(states, run.os_hist)
    ])
    cosine = velocity[:, 1] / np.linalg.norm(velocity, axis=1)
    return np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))

plt.figure()
plt.plot(time, angle_to_velocity(run.state_hist), label="With drag")
plt.plot(time, angle_to_velocity([initial_state] * len(run.os_hist)),
         "--", label="Without drag")
plt.xlabel("Time [min]")
plt.ylabel("Long axis to velocity [deg]")
plt.legend()

torques = np.array([
    drag.torque(satellite, state, orbital_state)
    for state, orbital_state in zip(run.state_hist, run.os_hist)
])
plt.figure()
for axis, label in enumerate(("x", "y", "z")):
    plt.plot(time, torques[:, axis] * 1e9, label=label)
plt.xlabel("Time [min]")
plt.ylabel("Drag torque [nN m]")
plt.legend()
plt.show()
