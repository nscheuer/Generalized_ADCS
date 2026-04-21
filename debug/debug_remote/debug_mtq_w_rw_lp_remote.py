import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import ADCS as ADCS

np.random.seed(42)
real_sat = ADCS.satellite_factory.create_beavercube2_cubesat(estimated=False)
x_0 = np.array([0.0, 0.0, 0.0] + [1, 0, 0, 0] + [0.0])

controller = ADCS.controller.MTQ_w_RW_LP(
    est_sat=real_sat,
    p_gain=0.00005,
    d_gain=0.002,
    c_gain=0.001,
    h_target=np.array([0.0, 0.0, 0.0]),
)

os0 = ADCS.Orbital_State(
    ephem=ADCS.Ephemeris(),
    J2000=0.22,
    R=7000 * np.array([0, np.sqrt(2) / 2, np.sqrt(2) / 2]),
    V=np.array([8, 0, 0]),
)

goal = ADCS.goals.ECI_Goal(eci_vector=ADCS.helpers.normalize(np.array([1.0, 1.0, 1.0])))

remote_host = os.getenv("ADCS_REMOTE_HOST", "10.77.0.5")
remote_port = int(os.getenv("ADCS_REMOTE_PORT", "5000"))

results = ADCS.simulate_remote(
    x=x_0,
    satellite=real_sat,
    os0=os0,
    controller=controller,
    goal=goal,
    dt=1.0,
    tf=1000.0,
    remote=ADCS.remote.RemoteSimulationConfig(
        controller=ADCS.remote.ComponentLocation.REMOTE,
        estimator=ADCS.remote.ComponentLocation.LOCAL,
        orbit_estimator=ADCS.remote.ComponentLocation.LOCAL,
        host=remote_host,
        port=remote_port,
        timeout_s=0.5,
        retries=2,
    ),
)

ADCS.plot(
    results,
    ADCS.plots.AttitudePlot(sources=["real", "reference"]),
    layout=(1, 1),
    title="Remote Controller: 3+1 LP Reduced",
)

ADCS.plot(
    results,
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    layout=(2, 2),
    title="Remote Controller: 3+1 LP Reduced",
)

plt.show()