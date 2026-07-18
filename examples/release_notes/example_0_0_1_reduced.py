import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import ADCS as ADCS
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
real_sat = ADCS.satellite_factory.create_beavercube2_cubesat(estimated=False)
x_0 = ADCS.State.from_array(np.array([0, 0, 0] + [1, 0, 0, 0] + [0]))

controller = ADCS.controller.MTQ_w_RW_LP(est_sat=real_sat, p_gain=0.00005, d_gain=0.002, c_gain=0.001, h_target=np.array([0.0, 0.0, 0.0]))
goal = ADCS.goals.ECI_Goal(eci_vector=np.array([1, 0, 0]))

os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(),J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 7.5, 0]))

def make_random_os(rng: np.random.Generator) -> ADCS.Orbital_State:
    return ADCS.orbits.create_random_circular_os(radius_km=7000.0, J2000=0.22, rng=rng)

mc_config = ADCS.MCConfig(
    w = lambda rng: ADCS.helpers.normalize(rng.standard_normal(3)) * (rng.uniform(0.1, 1.0) * np.pi / 180.0),
    q = lambda rng: ADCS.helpers.normalize(rng.standard_normal(4)),
    h = lambda rng: rng.uniform(-0.0001, 0.0001, size=1),
    goal = lambda rng: ADCS.goals.ECI_Goal(eci_vector=ADCS.helpers.normalize(rng.standard_normal(3))),
    orbit = make_random_os
)

mc_results = ADCS.simulate_mc(
    x=x_0,
    satellite=real_sat,
    controller=controller,
    goal=goal,
    os0=os0,
    dt=1.0,
    tf=1000.0,
    mc_config=mc_config,
    num_runs=12,
    base_seed=42,
)

ADCS.plot(
    mc_results,
    ADCS.plots.ControlPlot(),
    ADCS.plots.TargetPlot(modes=["real_target"]),
    ADCS.plots.TargetHistogram(),
    ADCS.plots.IlluminationPlot(),
    layout=(2,2),
    title="Control Plot",
)

ADCS.plot(
    mc_results,
    ADCS.plots.AnimationPlot(),
    layout=(1,1),
    title="Animation Plot",
)

ADCS.plot(
    mc_results,
    ADCS.plots.OrbitPlot(),
    layout=(1,1),
    title="Orbit Plot",
)

ADCS.plot(
    mc_results,
    ADCS.plots.OrbitPositionPlot(),
    ADCS.plots.OrbitPositionPlotSingle(component="m"),
    ADCS.plots.OrbitPositionPlotCombined(),
    ADCS.plots.OrbitVelocityPlot(),
    ADCS.plots.OrbitVelocityPlotSingle(component="m"),
    ADCS.plots.OrbitVelocityPlotCombined(),
    layout=(3,2),
    title="Orbit Position and Velocity Plot",
)

ADCS.plot(
    mc_results,
    ADCS.plots.QuaternionPlot(),
    ADCS.plots.QuaternionPlotSingle(component=0),
    ADCS.plots.QuaternionPlotCombined(),
    ADCS.plots.AngularVelocityPlot(),
    ADCS.plots.AngularVelocityPlotSingle(component="m"),
    ADCS.plots.AngularVelocityPlotCombined(),
    layout=(3,2),
    title="Quaternion and Angular Velocity Plot",
)



plt.show()