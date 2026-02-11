import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
import ADCS as ADCS
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
real_sat = ADCS.satellite_factory.create_beavercube1_cubesat(estimated=False)
x_0 = np.array([-0.00874868,  0.00209214,  0.005936770] + [0.86698928, 0.29417644, 0.34385383, 0.20869681]) # w, q

controller = ADCS.controller.MTQ_Lovera(est_sat=real_sat, p_gain=0.0001, d_gain=0.001, eps=1.0)
os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(),J2000=0.22, R=7000*np.array([0, np.sqrt(2)/2, np.sqrt(2)/2]), V=np.array([8, 0, 0]))

def make_random_os(rng: np.random.Generator) -> ADCS.Orbital_State:
    return ADCS.orbits.create_random_circular_os(radius_km=7000.0, J2000=0.22, rng=rng)

mc_config = ADCS.MCConfig(
    w = lambda rng: ADCS.helpers.normalize(rng.standard_normal(3)) * (rng.uniform(0.1, 1.0) * np.pi / 180.0),
    q = lambda rng: ADCS.helpers.normalize(rng.standard_normal(4)),
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

results.save("mc100_lovera_3+0_reduced", out_dir="papers/Planner/new/output")

# ADCS.plot(
#     results,
#     ADCS.plots.AnimationPlot(),
#     layout=(1,1),
#     title="3+0 Lovera Reduced",
# )

ADCS.plot(
    results,
    ADCS.plots.AttitudePlot(sources=["real", "reference"]),
    layout=(1,1),
    title="3+0 Lovera Reduced",
)

ADCS.plot(
    results,
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    layout=(2,2),
    title="3+0 Lovera Reduced",
)

plt.show()