"""Simulate and plot the attitude of a freely rotating spacecraft."""

import numpy as np
import matplotlib.pyplot as plt

import ADCS

satellite = ADCS.Satellite(mass=4.0, J_0=np.diag([0.0333, 0.0333, 0.00667]))
initial_state = ADCS.State(
    w=np.deg2rad([5.0, 0.0, 0.0]),
    q=np.array([1.0, 0.0, 0.0, 0.0]),
)
orbit = ADCS.Orbital_State(
    ephem=ADCS.Ephemeris(),
    J2000=0.22,
    R=np.array([7000.0, 0.0, 0.0]),
    V=np.array([0.0, 7.5, 0.0]),
)
results = ADCS.simulate(
    x=initial_state, satellite=satellite, os0=orbit,
    dt=1.0, tf=30.0,
)

ADCS.plot(results, ADCS.plots.AttitudePlot(), layout=(1, 1))
plt.show()
