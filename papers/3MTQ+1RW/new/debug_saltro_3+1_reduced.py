import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

import ADCS as ADCS
import matplotlib.pyplot as plt
import numpy as np
from ADCS.controller.neo_planner.NEO_planner_settings import PlannerSettings

real_sat = ADCS.satellite_factory.create_beavercube2_cubesat(estimated=False)
x_0 = np.array([0.0, 0.0, 0.0] + [1, 0, 0, 0] + [0.0])  # w, q, h

planner_settings = PlannerSettings(est_sat=real_sat)
planner_settings.passes[0].dt = 10.0

controller = ADCS.controller.SALTRO(est_sat=real_sat, planner_settings=planner_settings)

os0 = ADCS.Orbital_State(
    ephem=ADCS.Ephemeris(),
    J2000=0.22,
    R=np.array([7000.0, 0.0, 0.0]),
    V=np.array([0.0, 7.5, 0.0]),
)

goal = ADCS.goals.ECI_Goal(np.array([0, 0, -1]))

results = ADCS.simulate(
    x=x_0,
    satellite=real_sat,
    controller=controller,
    goal=goal,
    os0=os0,
    dt=1.0,
    tf=1000.0,
)

ADCS.plot(
    results,
    ADCS.plots.AnimationPlot(),
    layout=(1, 1),
    title="3+1 SALTRO Reduced",
)

ADCS.plot(
    results,
    ADCS.plots.AttitudePlot(sources=["real", "reference"]),
    layout=(1, 1),
    title="3+1 SALTRO Reduced",
)

ADCS.plot(
    results,
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    layout=(2, 2),
    title="3+1 SALTRO Reduced",
)

plt.show()
