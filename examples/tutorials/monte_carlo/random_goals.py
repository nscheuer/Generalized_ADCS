"""Sample a goal for each uncontrolled spacecraft."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ADCS

ROOT = Path(__file__).resolve().parents[3]
IMAGES = ROOT / "docs/source/_static/tutorials/monte_carlo"

satellite = ADCS.Satellite(mass=4.0, J_0=np.diag([0.03, 0.04, 0.05]),
                           boresight=np.array([0.0, 0.0, 1.0]))
initial_state = ADCS.State(
    w=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])
)
orbit = ADCS.Orbital_State(
    ephem=ADCS.Ephemeris(), J2000=0.22,
    R=np.array([7000.0, 0.0, 0.0]),
    V=np.array([0.0, 7.5, 0.0]),
)
config = ADCS.MCConfig(
    orbit=lambda rng: ADCS.orbits.create_random_circular_os(
        radius_km=7000.0, J2000=0.22, rng=rng
    ),
    q=lambda rng: ADCS.helpers.normalize(rng.standard_normal(4)),
    w=lambda rng: np.deg2rad(rng.uniform(-3.0, 3.0, size=3)),
    goal=lambda rng: ADCS.goals.ECI_Goal(
        eci_vector=ADCS.helpers.normalize(rng.standard_normal(3))
    ),
)
results = ADCS.simulate_mc(
    x=initial_state, satellite=satellite, os0=orbit,
    dt=2.0, tf=120.0, mc_config=config,
    num_runs=8, base_seed=42, max_workers=2,
)
ADCS.plot(results, ADCS.plots.AnimationPlot(
    screenshot_path=IMAGES / "random_goals.png"), layout=(1, 1))
ADCS.plot(results, ADCS.plots.TargetPlot(modes=["real_target"],
          title="Pointing error without control"), layout=(1, 1))
plt.savefig(IMAGES / "random_goals_error.png", dpi=160)
plt.close("all")
