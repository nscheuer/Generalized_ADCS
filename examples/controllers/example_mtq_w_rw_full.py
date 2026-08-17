import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import ADCS as ADCS
import numpy as np
import matplotlib.pyplot as plt

mtm_max_torque = 0.1
mtqs = [ADCS.MTQ(axis=axes, max_torque=mtm_max_torque) for axes in np.eye(3)]

rw_max_torque = 4.51
rw_J = 0.22
rw_h0 = 1
rw_hmax = 3.8
rws = [ADCS.RW(axis=axes, max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax) for axes in np.eye(3)]

acts = mtqs+rws

mtms = [ADCS.MTM(axis=axes) for axes in np.eye(3)]

real_sat = ADCS.Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]), actuators=acts, sensors=mtms, boresight=np.array([0, 0, 1]))

x_0 = ADCS.State.from_array(np.array([0, 0, 0] + [1, 0, 0, 0] + [0, 0, 0])) # w, q, h

controller = ADCS.controller.MTQ_w_RW(est_sat=real_sat, p_gain=0.1, d_gain=0.7, c_gain=0.1, h_target=np.array([0, 0, 0]))

os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(),J2000=0.22, R=np.array([5000, 0, 5000]), V=np.array([0, 7.5, 0]))
goal = ADCS.goals.Fixed_Attitude_Goal(q_ref=np.array([0, 0, 1, 0]))

results = ADCS.simulate(
    x=x_0,
    satellite=real_sat,
    controller=controller,
    goal=goal,
    os0=os0,
    dt=1.0,
    tf=100.0
)

ADCS.plot(
    results,
    ADCS.plots.AttitudePlot(sources=["real", "reference"]),
    layout=(1,1),
    title="MTQ with Reaction Wheels Full Pointing Control",
)

ADCS.plot(
    results,
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
    ADCS.plots.SensorsPlot(title="MTM & RW Readings", sources=["clean"], units="T"),
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    layout=(2,2),
    title="MTQ with Reaction Wheels Full Pointing Control",
)

plt.show()