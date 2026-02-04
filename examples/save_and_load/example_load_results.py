import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import ADCS as ADCS
import numpy as np
import matplotlib.pyplot as plt

results = ADCS.SimulationResults.load("papers/Planner/new/output/altro_3+1_full_20260204_144357.sim", ephem=ADCS.Ephemeris())

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

ADCS.plot(
    results,
    ADCS.plots.ControlPlotSingle(index=0, title="Magnetorquer 1", units="Am²"),
    ADCS.plots.ControlPlotSingle(index=1, title="Magnetorquer 2", units="Am²"),
    ADCS.plots.ControlPlotSingle(index=2, title="Magnetorquer 3", units="Am²"),
    ADCS.plots.ControlPlotSingle(index=3, title="Reaction Wheel", units="Nms"),
    layout=(2,2),
    title="3+1 ALTRO Reduced",
)

plt.show()