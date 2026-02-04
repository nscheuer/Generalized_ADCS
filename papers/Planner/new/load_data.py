import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
import ADCS as ADCS
import numpy as np
import matplotlib.pyplot as plt


results_30_full = ADCS.SimulationResults.load("papers/Planner/new/output/mc100_altro_3+0_full_20260204_162440.sim", ephem=ADCS.Ephemeris())
results_30_reduced = ADCS.SimulationResults.load("papers/Planner/new/output/mc100_altro_3+0_reduced_20260204_185357.sim", ephem=ADCS.Ephemeris())
results_31_full = ADCS.SimulationResults.load("papers/Planner/new/output/mc100_altro_3+1_full_20260204_235053.sim", ephem=ADCS.Ephemeris())
results_31_reduced = ADCS.SimulationResults.load("papers/Planner/new/output/mc100_altro_3+1_reduced_20260205_001823.sim", ephem=ADCS.Ephemeris())

results_30_lovera_full = ADCS.SimulationResults.load("papers/Planner/new/output/mc100_lovera_3+0_full_20260205_003347.sim", ephem=ADCS.Ephemeris())
results_30_lovera_reduced = ADCS.SimulationResults.load("papers/Planner/new/output/mc100_lovera_3+0_reduced_20260205_003107.sim", ephem=ADCS.Ephemeris())
results_31_lp_full = ADCS.SimulationResults.load("papers/Planner/new/output/mc100_lp_3+1_full_20260205_000958.sim", ephem=ADCS.Ephemeris())
results_31_lp_reduced = ADCS.SimulationResults.load("papers/Planner/new/output/mc100_lp_3+1_reduced_20260205_000536.sim", ephem=ADCS.Ephemeris())

ADCS.plot(
    results_30_full,
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetHistogram(bin_width=5.0, threshold=5),
    ADCS.plots.TargetHistogram(bin_width=5.0, threshold=10),
    layout=(2,2),
    title="3+0 ALTRO Full",
)

ADCS.plot(
    results_30_reduced,
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetHistogram(bin_width=5.0, threshold=5),
    ADCS.plots.TargetHistogram(bin_width=5.0, threshold=10),
    layout=(2,2),
    title="3+0 ALTRO Reduced",
)

ADCS.plot(
    results_31_full,
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetHistogram(bin_width=5.0, threshold=5),
    ADCS.plots.TargetHistogram(bin_width=5.0, threshold=10),
    layout=(2,2),
    title="3+1 ALTRO Full",
)

ADCS.plot(
    results_31_reduced,
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetHistogram(bin_width=5.0, threshold=5),
    ADCS.plots.TargetHistogram(bin_width=5.0, threshold=10),
    layout=(2,2),
    title="3+1 ALTRO Reduced",
)

ADCS.plot(
    results_31_lp_full,
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetHistogram(bin_width=5.0, threshold=5),
    ADCS.plots.TargetHistogram(bin_width=5.0, threshold=10),
    layout=(2,2),
    title="3+1 LP Full",
)

ADCS.plot(
    results_31_lp_reduced,
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetHistogram(bin_width=5.0, threshold=5),
    ADCS.plots.TargetHistogram(bin_width=5.0, threshold=10),
    layout=(2,2),
    title="3+1 LP Reduced",
)

ADCS.plot(
    results_30_lovera_full,
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetHistogram(bin_width=5.0, threshold=5),
    ADCS.plots.TargetHistogram(bin_width=5.0, threshold=10),
    layout=(2,2),
    title="3+0 Lovera Full",
)

ADCS.plot(
    results_30_lovera_reduced,
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetHistogram(bin_width=5.0, threshold=5),
    ADCS.plots.TargetHistogram(bin_width=5.0, threshold=10),
    layout=(2,2),
    title="3+0 Lovera Reduced",
)

plt.show()