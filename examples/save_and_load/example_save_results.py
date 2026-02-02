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
    dt=2.0,
    tf=500.0,
)

results.save("example_save_results", out_dir="examples/output")