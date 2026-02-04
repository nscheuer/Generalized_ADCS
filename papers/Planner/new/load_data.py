import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
import ADCS as ADCS
import numpy as np
import matplotlib.pyplot as plt


results_30_full = ADCS.SimulationResults.load("papers/Planner/new/output/mc12_altro_3+0_full_20260204_155915.sim", ephem=ADCS.Ephemeris())
results_30_reduced = ADCS.SimulationResults.load("papers/Planner/new/output/mc12_altro_3+0_reduced_20260204_160316.sim", ephem=ADCS.Ephemeris())
results_31_full = ADCS.SimulationResults.load("papers/Planner/new/output/mc12_altro_3+1_full_20260204_155150.sim", ephem=ADCS.Ephemeris())
results_31_reduced = ADCS.SimulationResults.load("papers/Planner/new/output/mc12_altro_3+1_reduced_20260204_161032.sim", ephem=ADCS.Ephemeris())

ADCS.plot(
    results_30_full,
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    layout=(2,2),
    title="3+0 ALTRO Full",
)

ADCS.plot(
    results_30_reduced,
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    layout=(2,2),
    title="3+0 ALTRO Reduced",
)

ADCS.plot(
    results_31_full,
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    layout=(2,2),
    title="3+1 ALTRO Full",
)

ADCS.plot(
    results_31_reduced,
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    layout=(2,2),
    title="3+1 ALTRO Reduced",
)
plt.show()