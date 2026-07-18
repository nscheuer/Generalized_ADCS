import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import ADCS as ADCS
import numpy as np
import matplotlib.pyplot as plt

real_sat = ADCS.satellite_factory.create_beavercube2_cubesat()

x_0 = ADCS.State.from_array(np.array([0, 0, 0] + [1, 0, 0, 0] + [0])) # w, q, h

controller = ADCS.controller.MTQ_w_RW_LP(est_sat=real_sat, p_gain=0.00005, d_gain=0.002, c_gain=0.001, h_target=np.array([0.0, 0.0, 0.0]))

os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(),J2000=0.22, R=np.array([5000, 0, 5000]), V=np.array([0, 7.5, 0]))
goal_timeline = {0.0: ADCS.goals.Coordinate_Goal(lat=33.75, lon=-84.3885, alt=0), 500.0: ADCS.goals.AntiSun_Goal()}
goallist = ADCS.GoalList(goal_timeline=goal_timeline, time_units="seconds", start_juliantime=0.22)

results = ADCS.simulate(
    x=x_0,
    satellite=real_sat,
    controller=controller,
    goal=goallist,
    os0=os0,
    dt=1.0,
    tf=1000.0
)

ADCS.plot(
    results,
    ADCS.plots.AttitudePlot(sources=["real", "reference"]),
    layout=(1,1),
    title="MTQ with Reaction Wheels Reduced Pointing Control",
)

ADCS.plot(
    results,
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
    ADCS.plots.SensorsPlot(title="MTM & RW Readings", sources=["clean"], units="T"),
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    layout=(2,2),
    title="MTQ with Reaction Wheels Reduced Pointing Control",
)

ADCS.plot(
    results,
    ADCS.plots.AnimationPlot(goal=ADCS.goals.Coordinate_Goal(lat=33.75, lon=-84.3885, alt=0)),
    layout=(1,1),
    title="Satellite Animation",
)

plt.show()