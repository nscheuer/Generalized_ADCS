import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

import ADCS as ADCS
import matplotlib.pyplot as plt
import numpy as np
from ADCS.controller.neo_planner.NEO_planner_settings import PlannerSettings

unitvecs = np.eye(3)

mtqs = [ADCS.MTQ(axis=axis, max_torque=0.2) for axis in unitvecs]
rws = [
    ADCS.RW(axis=axis, max_torque=0.001, J=1e-5, h=0.0, h_max=0.02)
    for axis in unitvecs
]
mtms = [ADCS.MTM(axis=axis) for axis in unitvecs]

real_sat = ADCS.Satellite(
    mass=4.0,
    J_0=np.diagflat([0.067, 0.071, 0.069]),
    actuators=mtqs + rws,
    sensors=mtms,
    boresight=np.array([0.0, 0.0, 1.0]),
)

x_0 = np.array([0.01, 0.01, 0.01] + [1.0, 0.0, 0.0, 0.0] + [0.0, 0.0, 0.0])

planner_settings = PlannerSettings(est_sat=real_sat)

controller = ADCS.controller.SALTRO(est_sat=real_sat, planner_settings=planner_settings)

os0 = ADCS.Orbital_State(
    ephem=ADCS.Ephemeris(),
    J2000=0.22,
    R=np.array([7000.0, 0.0, 0.0]),
    V=np.array([0.0, 7.5, 0.0]),
)

q_goal = np.array([np.sqrt(2.0) / 2.0, 0.0, 0.0, np.sqrt(2.0) / 2.0])
goal = ADCS.goals.Fixed_Attitude_Goal(q_ref=q_goal)
goal_list = ADCS.GoalList({os0.J2000: goal})

trajectory = controller.calculate_trajectory(
    t_start=os0.J2000,
    duration=400.0,
    x_0=x_0,
    os_0=os0,
    goals=goal_list,
    verbose=False,
)
controller.set_active_trajectory(trajectory)

results = ADCS.simulate(
    x=x_0,
    satellite=real_sat,
    controller=controller,
    goal=goal_list,
    os0=os0,
    dt=1.0,
    tf=400.0,
)

ADCS.plot(
    results,
    ADCS.plots.AnimationPlot(),
    layout=(1, 1),
    title="SALTRO 3+3 Clean",
)

ADCS.plot(
    results,
    ADCS.plots.AttitudePlot(sources=["real", "reference"]),
    layout=(1, 1),
    title="SALTRO 3+3 Clean",
)

ADCS.plot(
    results,
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(title="Control Commands"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    layout=(2, 2),
    title="SALTRO 3+3 Clean",
)

plt.show()
