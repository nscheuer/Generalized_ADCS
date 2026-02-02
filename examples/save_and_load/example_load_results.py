import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import ADCS as ADCS
import numpy as np
import matplotlib.pyplot as plt

results = ADCS.SimulationResults.load("examples/output/example_save_results_20260202_155358.sim", ephem=ADCS.Ephemeris())

ADCS.plot(
    results,
    ADCS.plots.ControlPlot(),
    ADCS.plots.TargetPlot(modes=["real_target"]),
    ADCS.plots.TargetHistogram(),
    ADCS.plots.IlluminationPlot(),
    layout=(2,2),
    title="Control Plot",
)

ADCS.plot(
    results,
    ADCS.plots.OrbitPlot(),
    layout=(1,1),
    title="Orbit Plot",
)
plt.show()