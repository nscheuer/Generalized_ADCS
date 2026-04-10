import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import ADCS as ADCS
import numpy as np
import matplotlib.pyplot as plt

satellite = ADCS.satellite_factory.create_beavercube1_cubesat()

x_0 = np.array([0, 0, 0] + [1, 0, 0, 0]) # w, q, h

planner_settings = ADCS.controller.plan_and_track.PlannerSettings(est_sat=satellite, bdot_on=0, dt_tp=50, dt_tvlqr=1.0)

planner_settings.verbosity = False
planner_settings.cost_main.use_full_cost_hessian = True
planner_settings.pass1.regularization.use_dynamics_hess = 1
planner_settings.init_traj.bdot_gain = 500
planner_settings.pass1.aug_lag.penalty_init = 1e-3
planner_settings.pass1.aug_lag.penalty_scale = 10
planner_settings.pass1.convergence.max_outer_iter = 15
planner_settings.pass1.convergence.max_inner_iter = 40
planner_settings.pass2.aug_lag.penalty_init = 1e5
planner_settings.pass2.aug_lag.penalty_scale = 10
planner_settings.pass2.convergence.max_outer_iter = 8
planner_settings.pass2.convergence.max_inner_iter = 20

planner_settings.cost_main = ADCS.controller.plan_and_track.CostWeights(
        angle=1e1,
        angle_N=1e1,
        ang_vel=1e5,
        ang_vel_N=1e5,
        ang_vel_err_dir=1e2,
        ang_vel_err_dir_N=0.0,
        ang_vel_mag=0.0,
        ang_vel_mag_N=0.0,
        control_mult=1.0,
        ang_cost_func_type=2,
    )

planner_settings.cost_second = planner_settings.cost_main

planner_settings.cost_tvlqr = ADCS.controller.plan_and_track.CostWeights(
        angle=1e5,
        angle_N=1e6,
        ang_vel=1e6,
        ang_vel_N=1e8,
        ang_vel_mag=0.0,
        ang_vel_mag_N=0.0,
        control_mult=1.0,
        ang_cost_func_type=2,
    )

controller = ADCS.controller.Plan_and_Track_LQR(est_sat=satellite, planner_settings=planner_settings)

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
    title="3+1 ALTRO Reduced",
)

ADCS.plot(
    results,
    ADCS.plots.AttitudePlot(sources=["real", "reference"]),
    layout=(1,1),
    title="3+0 ALTRO Mixed",
)

ADCS.plot(
    results,
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    layout=(2,2),
    title="3+0 ALTRO Mixed",
)

ADCS.plot(
    results,
    ADCS.plots.ControlPlotSingle(index=0, title="Magnetorquer 1", units="Am²"),
    ADCS.plots.ControlPlotSingle(index=1, title="Magnetorquer 2", units="Am²"),
    ADCS.plots.ControlPlotSingle(index=2, title="Magnetorquer 3", units="Am²"),
    layout=(3,1),
    title="3+0 ALTRO Mixed",
)

plt.show()