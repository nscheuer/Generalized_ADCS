import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import ADCS as ADCS
import numpy as np
import matplotlib.pyplot as plt

satellite = ADCS.satellite_factory.create_beavercube2_cubesat()

x_0 = ADCS.State.from_array(np.array([0, 0, 0] + [1, 0, 0, 0] + [0.0])) # w, q, h

planner_settings = ADCS.controller.saltro.PlannerSettings(est_sat=satellite)
controller = ADCS.controller.SALTRO(est_sat=satellite, planner_settings=planner_settings)

os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(), J2000=0.22, R=np.array([5000, 0, 5000]), V=np.array([0, 7.5, 0]))
goal_timeline = {0.0: ADCS.goals.ECI_Goal(np.array([1, 0, 0])), 300.0: ADCS.goals.No_Goal(), 400.0: ADCS.goals.ECI_Goal(np.array([0, 1, 0])), 700.0: ADCS.goals.No_Goal(), 800.0: ADCS.goals.ECI_Goal(np.array([0, 0, 1]))}
goallist = ADCS.GoalList(goal_timeline=goal_timeline, time_units="seconds", start_juliantime=0.22)

results = ADCS.simulate(
    x=x_0,
    satellite=satellite,
    controller=controller,
    goal=goallist,
    os0=os0,
    dt=1.0,
    tf=1000.0
)

ADCS.plot(
    results,
    ADCS.plots.AnimationPlot(),
    layout=(1,1),
    title="3+1 SALTRO Reduced",
)

ADCS.plot(
    results,
    ADCS.plots.AttitudePlot(sources=["real", "reference"]),
    layout=(1,1),
    title="3+1 SALTRO Mixed",
)

ADCS.plot(
    results,
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    layout=(2,2),
    title="3+1 SALTRO Mixed",
)

ADCS.plot(
    results,
    ADCS.plots.ControlPlotSingle(index=0, title="Magnetorquer 1", units="Am²"),
    ADCS.plots.ControlPlotSingle(index=1, title="Magnetorquer 2", units="Am²"),
    ADCS.plots.ControlPlotSingle(index=2, title="Magnetorquer 3", units="Am²"),
    layout=(3,1),
    title="3+1 SALTRO Mixed",
)

plt.show()