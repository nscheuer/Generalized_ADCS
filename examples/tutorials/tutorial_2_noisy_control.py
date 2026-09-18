import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import ADCS as ADCS
import numpy as np
import matplotlib.pyplot as plt

acts_noise = ADCS.Noise(std_noise=0.005)
acts = [ADCS.MTQ(axis, max_torque=0.1, noise=acts_noise) for axis in np.eye(3)]
sens_noise = ADCS.Noise(std_noise=1e-7)
sens = [ADCS.MTM(axis, noise=sens_noise) for axis in np.eye(3)]

satellite = ADCS.Satellite(mass=10, J_0=np.diag([0.003, 0.003, 0.003]), actuators=acts, sensors=sens)
x_0 = ADCS.State.from_array(np.array([0.01, 0.05, 0] + [1, 0, 0, 0])) # w, q

controller = ADCS.controller.BDot(est_sat=satellite, gain=5e4)

os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(),J2000=0.22, R=np.array([5000, 0, 5000]), V=np.array([0, 7.5, 0]))

results = ADCS.simulate(
    x=x_0,
    satellite=satellite,
    controller=controller,
    os0=os0,
    dt=10.0,
    tf=5000.0
)

ADCS.plot(
    results,
    ADCS.plots.AttitudePlot(sources=["real"]),
    layout=(1,1),
    title="B-Dot Magnetic Detumbling Control",
)

ADCS.plot(
    results,
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
    ADCS.plots.SensorsPlotCombined(title="Magnetometer Readings", sources=["clean"], units="T"),
    ADCS.plots.SensorsPlotCombined(title="Magnetometer Readings (Noisy)", sources=["real"], units="T"),
    layout=(2,2),
    title="B-Dot Magnetic Detumbling Control",
)
plt.show()