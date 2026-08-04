import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
import ADCS as ADCS
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
real_sat = ADCS.satellite_factory.create_beavercube2_cubesat(estimated=False)
x_0 = np.array([0.0, 0.0, 0.0] + [1, 0, 0, 0] + [0.0]) # w, q, h


planner_settings = ADCS.controller.plan_and_track.PlannerSettings(est_sat=real_sat, bdot_on=0, dt_tp=50, dt_tvlqr=1.0)

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

controller = ADCS.controller.Plan_and_Track_LQR(est_sat=real_sat, planner_settings=planner_settings)
os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(),J2000=0.22, R=7000*np.array([0, np.sqrt(2)/2, np.sqrt(2)/2]), V=np.array([8, 0, 0]))

def make_random_os(rng: np.random.Generator) -> ADCS.Orbital_State:
    return ADCS.orbits.create_random_circular_os(radius_km=7000.0, J2000=0.22, rng=rng)

mc_config = ADCS.MCConfig(
    w = lambda rng: ADCS.helpers.normalize(rng.standard_normal(3)) * (rng.uniform(0.1, 1.0) * np.pi / 180.0),
    q = lambda rng: ADCS.helpers.normalize(rng.standard_normal(4)),
    h = lambda rng: rng.uniform(-0.0001, 0.0001, size=1),
    goal = lambda rng: ADCS.goals.ECI_Goal(eci_vector=ADCS.helpers.normalize(rng.standard_normal(3))),
    orbit = make_random_os
)

results = ADCS.simulate_mc(
    x=x_0,
    satellite=real_sat,
    controller=controller,
    os0=os0,
    dt=1.0,
    tf=1000.0,
    mc_config=mc_config,
    num_runs=100,
    base_seed=42
)

results.save("mc100_altro_3+1_reduced", out_dir="papers/3MTQ+1RW/new/output")

# ADCS.plot(
#     results,
#     ADCS.plots.AnimationPlot(),
#     layout=(1,1),
#     title="3+1 ALTRO Reduced",
# )

ADCS.plot(
    results,
    ADCS.plots.AttitudePlot(sources=["real", "reference"]),
    layout=(1,1),
    title="3+1 ALTRO Reduced",
)

ADCS.plot(
    results,
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    layout=(2,2),
    title="3+1 ALTRO Reduced",
)

plt.show()