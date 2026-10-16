import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import ADCS as ADCS
import numpy as np
import matplotlib.pyplot as plt

acts: list[ADCS.Actuator] = [ADCS.RW(axis=np.array([1, 0, 0]), max_torque=0.0023, J=5.7e-6, h=0.0, h_max=0.0036)]
acts += [ADCS.MTQ(axis, max_torque=0.2) for axis in np.eye(3)]
sens = [ADCS.MTM(axis) for axis in np.eye(3)]

satellite = ADCS.Satellite(mass=4, J_0=np.diag([0.03, 0.03, 0.01]), actuators=acts, sensors=sens, boresight=np.array([0, 0, 1]))
x_0 = ADCS.State.from_array(np.array([0.01, -0.02, 0.01] + [1, 0, 0, 0] + [0.0]))  # w, q, h

controller = ADCS.controller.MTQ_w_RW_LP(est_sat=satellite, p_gain=0.00005, d_gain=0.002, c_gain=0.001, h_target=np.array([0, 0, 0]))

goal = ADCS.goals.Coordinate_Goal(lat=42.36, lon=-71.06, alt=0)
environment = ADCS.Environment(
    magnetic_field=ADCS.Environment.MagneticField.IGRF14Numba(),
    atmosphere=ADCS.Environment.Atmosphere.SMADAtmosphere(),
    ephemeris=ADCS.Environment.Ephemeris.DE421(),
)
os0 = ADCS.Orbital_State(
    environment=environment,
    J2000=0.22,
    R=np.array([5000, 0, 5000]),
    V=np.array([0, 7.5, 0]),
)
print(f"Initial magnetic field (ECEF): {environment.magnetic_field.field(os0.ECEF, os0.datetime)} T")
print(f"Initial density: {environment.atmosphere.density(os0.R)} kg/m³")

results = ADCS.simulate(
    x=x_0,
    satellite=satellite,
    controller=controller,
    goal=goal,
    os0=os0,
    dt=1.0,
    tf=500.0,
    environment=environment,
)

ADCS.plot(
    results,
    ADCS.plots.AnimationPlot(goal=goal),
    layout=(1,1),
    title="Underactuated Control Animation",
)

ADCS.plot(
    results,
    ADCS.plots.TargetPlot(),
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(),
    ADCS.plots.ControlPlotSingle(index=0, title="RW Control Torque"),
    layout=(2,2),
    title="Underactuated Control with MTQ and RW",
)
plt.show()
