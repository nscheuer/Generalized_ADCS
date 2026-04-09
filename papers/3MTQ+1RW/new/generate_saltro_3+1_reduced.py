import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

import ADCS as ADCS
import numpy as np
import matplotlib.pyplot as plt
from ADCS.controller.neo_planner.NEO_planner_settings import PlannerSettings

np.random.seed(43)
real_sat = ADCS.satellite_factory.create_beavercube2_cubesat(estimated=False)
x_0 = np.array([0.0, 0.0, 0.0] + [1, 0, 0, 0] + [0.0]) # w, q, h

planner_settings = PlannerSettings(est_sat=real_sat)
planner_settings.passes[0].dt = 10.0

controller = ADCS.controller.SALTRO(est_sat=real_sat, planner_settings=planner_settings)
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
    num_runs=10,
    base_seed=43
)

# results.save("mc100_saltro_3+1_reduced", out_dir="papers/3MTQ+1RW/new/output")

# ADCS.plot(
#     results,
#     ADCS.plots.AnimationPlot(),
#     layout=(1,1),
#     title="3+1 SALTRO Reduced",
# )

ADCS.plot(
    results,
    ADCS.plots.AttitudePlot(sources=["real", "reference"]),
    layout=(1,1),
    title="3+1 SALTRO Reduced",
)

ADCS.plot(
    results,
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    layout=(2,2),
    title="3+1 SALTRO Reduced",
)

plt.show()
