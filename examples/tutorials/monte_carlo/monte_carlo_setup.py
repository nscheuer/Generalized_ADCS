"""Run identical spacecraft cases to establish a Monte Carlo baseline."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import ADCS

satellite = ADCS.Satellite(
    mass=4.0, J_0=np.diag([0.03, 0.04, 0.05]),
    boresight=np.array([0.0, 0.0, 1.0]),
)
initial_state = ADCS.State(
    w=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])
)
orbit = ADCS.Orbital_State(
    ephem=ADCS.Ephemeris(), J2000=0.22,
    R=np.array([7000.0, 0.0, 0.0]),
    V=np.array([0.0, 7.5, 0.0]),
)
results = ADCS.simulate_mc(
    x=initial_state, satellite=satellite, os0=orbit,
    goal=ADCS.goals.ECI_Goal(eci_vector=np.array([0.0, 0.0, 1.0])),
    dt=2.0, tf=120.0, mc_config=None,
    num_runs=8, base_seed=42, max_workers=2,
)
ADCS.plot(results, ADCS.plots.AnimationPlot(), layout=(1, 1))
