"""Compare four environmental torques on the same 3U spacecraft."""

import matplotlib.pyplot as plt
import numpy as np
import ADCS
from ADCS.helpers.animations import animate_orbit_godot

environment = ADCS.Environment(
    atmosphere=ADCS.SMADAtmosphere(), magnetic_field=ADCS.IGRF14(),
)
size = np.array([0.1, 0.3, 0.1])
faces = []
for axis in range(3):
    for sign in (-1.0, 1.0):
        normal = sign * np.eye(3)[axis]
        faces.append(ADCS.disturbances.GeometryFace(
            area=np.prod(np.delete(size, axis)),
            centroid=0.5 * size[axis] * normal,
            normal=normal, CD=2.2,
            eta_s=0.3, eta_d=0.3, eta_a=0.4,
        ))
geometry = ADCS.disturbances.GeometryConfig(faces)
drag = ADCS.disturbances.Drag_Disturbance(geometry)
srp = ADCS.disturbances.SRP_Disturbance(geometry)
gravity_gradient = ADCS.disturbances.GG_Disturbance()
dipole = ADCS.disturbances.Dipole_Disturbance(np.array([0.01, 0.0, 0.0]))
disturbances = [drag, srp, gravity_gradient, dipole]

inertia = np.diag([0.0333, 0.00667, 0.0333])
satellite = ADCS.Satellite(
    mass=4.0, COM=np.array([0.0, 0.02, 0.0]),
    J_0=inertia, disturbances=disturbances,
)
quiet_satellite = ADCS.Satellite(
    mass=4.0, COM=np.array([0.0, 0.02, 0.0]), J_0=inertia,
)
initial_state = ADCS.State(
    q=np.array([1.0, 0.0, 0.0, 0.0]), w=np.zeros(3),
)
radius = 6878.1363  # 500 km altitude
orbit = ADCS.Orbital_State(
    environment=environment, J2000=0.22,
    R=np.array([radius, 0.0, 0.0]),
    V=np.array([0.0, np.sqrt(398600.4415 / radius), 0.0]),
)
results = ADCS.simulate(
    x=initial_state, satellite=satellite, os0=orbit,
    environment=environment, dt=20.0, tf=5400.0,
)
quiet_results = ADCS.simulate(
    x=initial_state, satellite=quiet_satellite, os0=orbit,
    environment=environment, dt=20.0, tf=5400.0,
)

altitudes = np.geomspace(100.0, 100000.0, 24)  # km
models = (("Drag", drag), ("SRP", srp),
          ("Gravity gradient", gravity_gradient),
          ("Residual dipole", dipole))
mean_torques = np.empty((len(altitudes), len(models)))

for i, altitude in enumerate(altitudes):
    radius = 6378.1363 + altitude
    period = 2 * np.pi * np.sqrt(radius**3 / 398600.4415)
    orbit = ADCS.Orbital_State(
        environment=environment, J2000=0.22,
        R=np.array([radius, 0.0, 0.0]),
        V=np.array([0.0, np.sqrt(398600.4415 / radius), 0.0]),
    )
    sweep = ADCS.simulate(
        x=initial_state, satellite=satellite, os0=orbit,
        environment=environment, dt=period / 96, tf=period,
    )
    run = sweep.first()
    for j, (_, model) in enumerate(models):
        magnitudes = [
            np.linalg.norm(
                model.torque(state, orbital_state) if model is dipole else
                model.torque(satellite, state, orbital_state)
            )
            for state, orbital_state in zip(run.state_hist, run.os_hist)
        ]
        mean_torques[i, j] = np.mean(magnitudes)

for j, (label, _) in enumerate(models):
    # The atmosphere model has vanishingly small tails at high altitude.
    positive = mean_torques[:, j] > 1e-13
    plt.loglog(mean_torques[positive, j], altitudes[positive], label=label)
plt.xlabel("Mean torque magnitude [N m]")
plt.ylabel("Altitude [km]")
plt.xlim(1e-13, 1e-4)
plt.ylim(80, 150000)
plt.yticks([1e2, 1e3, 1e4, 1e5], [r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"])
plt.grid(True, which="major", alpha=0.3)
plt.legend(loc="upper right")
plt.show()

animate_orbit_godot(
    results=results, model="mirata", overlays=[], graphics_quality="high",
)
animate_orbit_godot(
    results=quiet_results, model="mirata", overlays=[], graphics_quality="high",
)
