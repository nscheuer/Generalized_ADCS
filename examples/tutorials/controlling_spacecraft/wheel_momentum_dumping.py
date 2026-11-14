"""Point at nadir while magnetorquers unload three reaction wheels."""

import matplotlib.pyplot as plt
import numpy as np

import ADCS
from ADCS.helpers.animations import animate_orbit_godot
from ADCS.helpers.animations.godot.overlays import Arrow, TargetLine
from ADCS.helpers.math_helpers import rot_mat

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
disturbances = [
    ADCS.disturbances.Drag_Disturbance(geometry),
    ADCS.disturbances.SRP_Disturbance(geometry),
    ADCS.disturbances.GG_Disturbance(),
    ADCS.disturbances.Dipole_Disturbance(np.array([0.01, 0.005, -0.007])),
]
wheels = [
    ADCS.RW(axis=axis, max_torque=1e-3, J=5e-6, h=0.0, h_max=0.02)
    for axis in np.eye(3)
]
magnetorquers = [ADCS.MTQ(axis=axis, max_torque=0.2) for axis in np.eye(3)]
magnetometers = [ADCS.MTM(axis=axis) for axis in np.eye(3)]
satellite = ADCS.Satellite(
    mass=4.0, COM=np.array([0.0, 0.02, 0.0]),
    J_0=np.diag([0.0333, 0.00667, 0.0333]),
    actuators=magnetorquers + wheels,
    sensors=magnetometers, disturbances=disturbances,
)
radius = 6878.1363  # 500 km altitude
orbit = ADCS.Orbital_State(
    environment=environment, J2000=0.22,
    R=np.array([0.0, -radius, 0.0]),
    V=np.array([np.sqrt(398600.4415 / radius), 0.0, 0.0]),
)
goal = ADCS.goals.Nadir_Goal()
nadir_ref, target_rate_eci = goal.to_ref(orbit)
nadir_eci = nadir_ref[1:]
q_scalar = np.sqrt((1.0 + nadir_eci[2]) / 2.0)
q_vector = np.cross([0.0, 0.0, 1.0], nadir_eci) / (2.0 * q_scalar)
q0 = np.r_[q_scalar, q_vector]
initial_state = ADCS.State(
    w=rot_mat(q0).T @ target_rate_eci, q=q0,
    h=np.array([0.004, -0.003, 0.002]),
)
controller = ADCS.controller.MTQ_w_RW(
    est_sat=satellite, p_gain=5e-5, d_gain=8e-4,
    c_gain=0.002, h_target=np.zeros(3),
)
results = ADCS.simulate(
    x=initial_state, satellite=satellite, os0=orbit,
    environment=environment, controller=controller, goal=goal,
    dt=5.0, tf=3600.0,
)
ADCS.plot(results, ADCS.plots.WheelMomentumPlot(),
          layout=(1, 1), figsize=(8, 4))

ADCS.plot(results, ADCS.plots.TargetPlot(modes=["real_target"],
                                        title="Nadir pointing error"),
          layout=(1, 1), figsize=(8, 4))

ADCS.plot(results, *[
    ADCS.plots.ControlPlotSingle(index=i, title=f"MTQ {axis} command",
                                 units="A m²", label=f"MTQ {axis}")
    for i, axis in enumerate("xyz")
], layout=(3, 1), figsize=(8, 9))
plt.tight_layout(pad=2.0)

ADCS.plot(results, *[
    ADCS.plots.ControlPlotSingle(index=i + 3, title=f"RW {axis} command",
                                 units="N m", label=f"RW {axis}",
                                 show_limits=False)
    for i, axis in enumerate("xyz")
], layout=(3, 1), figsize=(8, 9))
plt.tight_layout(pad=2.0)

if __name__ == "__main__":
    animate_orbit_godot(
        results=results, model="mirata", graphics_quality="high",
        overlays=[
            Arrow("@boresight:default", label="Body boresight", color="#facc15",
                  scale="fixed_px", px=120, width_px=5),
            TargetLine(color="#22c55e", length_px=150, width_px=5),
        ],
        camera_distance_m=20.0, camera_pitch_deg=60.0,
    )
