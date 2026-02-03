import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
import ADCS as ADCS
import numpy as np
import matplotlib.pyplot as plt

real_sat = ADCS.satellite_factory.create_beavercube2_cubesat(estimated=False)
x_0 = np.array([-0.00874868,  0.00209214,  0.005936770] + [0.86698928, 0.29417644, 0.34385383, 0.20869681] + [-9.76622366e-05]) # w, q, h

controller = ADCS.controller.MTQ_Lovera(est_sat=real_sat, p_gain=0.0001, d_gain=0.001, eps=1.0)

os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(),J2000=0.22, R=7000*np.array([0, np.sqrt(2)/2, np.sqrt(2)/2]), V=np.array([8, 0, 0]))
goal = ADCS.goals.ECI_Goal(np.array([-0.13901563, -0.36955661, -0.91875055]))

results = ADCS.simulate(
    x=x_0,
    satellite=real_sat,
    controller=controller,
    goal=goal,
    os0=os0,
    dt=2.0,
    tf=1000.0
)

ADCS.plot(
    results,
    ADCS.plots.AnimationPlot(goal=goal),
    layout=(1,1),
    title="3+1 LP Reduced",
)

ADCS.plot(
    results,
    ADCS.plots.AttitudePlot(sources=["real", "reference"]),
    layout=(1,1),
    title="3+1 LP Reduced",
)

ADCS.plot(
    results,
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    layout=(2,2),
    title="3+1 LP Reduced",
)

plt.show()