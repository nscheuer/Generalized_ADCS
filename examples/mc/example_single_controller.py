import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import ADCS as ADCS
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
real_sat = ADCS.satellite_factory.create_beavercube2_cubesat(estimated=False)
x_0 = np.array([0, 0, 0] + [1, 0, 0, 0] + [0])

controller = ADCS.controller.MTQ_w_RW_LP(est_sat=real_sat, p_gain=0.00005, d_gain=0.002, c_gain=0.001, h_target=np.array([0.0, 0.0, 0.0]))
goal = ADCS.goals.ECI_Goal(eci_vector=np.array([1, 0, 0]))

os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(),J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 7.5, 0]))

results = ADCS.simulate(
    x=x_0,
    satellite=real_sat,
    controller=controller,
    goal=goal,
    os0=os0,
    dt=1.0,
    tf=500.0,
)

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
    ADCS.plots.AttitudePlot(sources=["real", "reference"]),
    layout=(1,1),
    title="Attitude Plot",
)

ADCS.plot(
    results,
    ADCS.plots.AnimationPlot(),
    layout=(1,1),
    title="Animation Plot",
)

ADCS.plot(
    results,
    ADCS.plots.OrbitDensityPlot(),
    ADCS.plots.OrbitDensityModelPlot(),
    layout=(1,2),
    title="Orbit Density Plot",
)

ADCS.plot(
    results,
    ADCS.plots.OrbitMagneticPlot(),
    ADCS.plots.OrbitMagneticPlotSingle(component="m"),
    ADCS.plots.OrbitMagneticPlotCombined(),
    layout=(3,1),
    title="Orbit Magnetic Field Plot",
)

ADCS.plot(
    results,
    ADCS.plots.OrbitPlot(),
    layout=(1,1),
    title="Orbit Plot",
)

ADCS.plot(
    results,
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
    results,
    ADCS.plots.BiasPlot(),
    ADCS.plots.BiasPlotSingle(index=0),
    ADCS.plots.BiasPlotCombined(),
    ADCS.plots.SensorsPlot(),
    ADCS.plots.SensorsPlotSingle(index=0),
    ADCS.plots.SensorsPlotCombined(),
    layout=(3,2),
    title="Bias and Sensors Plot",
)

ADCS.plot(
    results,
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