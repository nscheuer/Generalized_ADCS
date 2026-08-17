import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
import ADCS as ADCS
import numpy as np
import matplotlib.pyplot as plt

mtq_max = 0.4
        
acts = [ADCS.MTQ(axis=j, max_torque=mtq_max) for j in np.eye(3)]

mtms = [ADCS.MTM(axis=j) for j in np.eye(3)]

real_sat = ADCS.Satellite(
    mass=1.2, 
    J_0=np.diagflat([0.022, 0.022, 0.004]), 
    actuators=acts, 
    sensors=mtms, 
    boresight=np.array([0, 0, 1])
)

x_0 = ADCS.State.from_array(np.array([0.0, 0.0, 0.0] + [1, 0, 0, 0])) # w, q

controller = ADCS.controller.MTQ_w_RW_QPW(est_sat=real_sat, p_gain=0.00005, d_gain=0.001, c_gain=0.001, h_target=np.array([0.004, 0.0, 0.0]))

os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(),J2000=0.22, R=7000*np.array([0, np.sqrt(2)/2, np.sqrt(2)/2]), V=np.array([8, 0, 0]))
goal = ADCS.goals.ECI_Goal(np.array([-0.13901563, -0.36955661, -0.91875055]))

results = ADCS.simulate(
    x=x_0,
    satellite=real_sat,
    controller=controller,
    goal=goal,
    os0=os0,
    dt=1.0,
    tf=1000.0
)

ADCS.plot(
    results,
    ADCS.plots.AnimationPlot(goal=goal),
    layout=(1,1),
    title="3+0 QPW Reduced",
)

ADCS.plot(
    results,
    ADCS.plots.AttitudePlot(sources=["real", "reference"]),
    layout=(1,1),
    title="3+0 QPW Reduced",
)

ADCS.plot(
    results,
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    layout=(2,2),
    title="3+0 QPW Reduced",
)

plt.show()