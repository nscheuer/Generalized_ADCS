import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import ADCS as ADCS
import numpy as np
import matplotlib.pyplot as plt

satellite = ADCS.satellite_factory.create_beavercube1_cubesat()
x_0 = ADCS.State.from_array(np.array([0.01, 0.05, 0] + [1, 0, 0, 0])) # w, q

controller = ADCS.controller.MTQ_Lovera(est_sat=satellite, p_gain=0.001, d_gain=0.005, eps=1.0)

os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(),J2000=0.22, R=np.array([5000, 0, 5000]), V=np.array([0, 7.5, 0]))
goal = ADCS.goals.ECI_Goal(eci_vector=np.array([0, 0, 1]))

results = ADCS.simulate(
    x=x_0,
    satellite=satellite,
    controller=controller,
    goal=goal,
    os0=os0,
    dt=1.0,
    tf=4000.0
)

ADCS.plot(
    results,
    ADCS.plots.AttitudePlot(sources=["real", "reference"]),
    layout=(1,1),
    title="Lovera Reduced Pointing Control",
)

ADCS.plot(
    results,
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
    ADCS.plots.SensorsPlot(title="MTM Readings", sources=["clean"], units="T"),
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    layout=(2,2),
    title="Lovera Reduced Pointing Control",
)

plt.show()