import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
import ADCS as ADCS
import numpy as np
import matplotlib.pyplot as plt

acts = [ADCS.RW(axis, max_torque=0.2, J=0.01, h=0.0, h_max=0.1) for axis in np.eye(3)]
sens = [ADCS.MTM(axis) for axis in np.eye(3)]
satellite = ADCS.Satellite(mass=10, J_0=np.diag([1, 1.2, 0.8]), actuators=acts, sensors=sens, boresight=np.array([0, 0, 1]))
x_0 = np.array([0.1, 0.1, 0.1, 1, 0, 0, 0, 0, 0, 0])

controller = ADCS.MTQ_w_RW(est_sat=satellite, p_gain=0.0, d_gain=0.0, c_gain=0.0, h_target=np.array([0, 0, 0]))

goal = ADCS.ECI_Goal(np.array([0, 0, 1]))
os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(),J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 7.5, 1]))

results = ADCS.simulate(
    x=x_0,
    satellite=satellite,
    controller=controller,
    goal=goal,
    os0=os0,
    dt=1.0,
    tf=100.0
)

from ADCS.helpers.plot import plot, AngularVelocityPlot, AngularVelocityPlotSingle, AngularVelocityPlotCombined

plot(
    results,
    AngularVelocityPlot(),
    AngularVelocityPlotSingle(component="m"),
    AngularVelocityPlotCombined(),
    layout=(3,1),
    title="Angular Rates",
)

from ADCS.helpers.plot import QuaternionPlot, QuaternionPlotSingle, QuaternionPlotCombined

plot(
    results,
    QuaternionPlot(),
    QuaternionPlotSingle(component=0),
    QuaternionPlotCombined(),
    layout=(3,1),
    title="Attitude Quaternions",
)

from ADCS.helpers.plot import ControlPlot, ControlPlotSingle, ControlPlotCombined

plot(
    results,
    ControlPlot(labels=["RW_1", "RW_2", "RW_3"]),
    ControlPlotSingle(index=0, label="RW_1"),
    ControlPlotCombined(labels=["RW_1", "RW_2", "RW_3"]),
    layout=(3,1),
    title="Control Torques",
)


plt.show()