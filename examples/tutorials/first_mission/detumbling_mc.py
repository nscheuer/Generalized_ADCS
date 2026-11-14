"""Monte Carlo B-dot detumbling after deployment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt
import numpy as np
import ADCS

magnetorquers = [ADCS.MTQ(axis=axis, max_torque=0.2) for axis in np.eye(3)]
magnetometers = [ADCS.MTM(axis=axis) for axis in np.eye(3)]
satellite = ADCS.Satellite(
    mass=4.0, J_0=np.diag([0.0333, 0.0333, 0.00667]),
    actuators=magnetorquers, sensors=magnetometers,
)
initial_state = ADCS.State(w=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]))
orbit = ADCS.Orbital_State(
    ephem=ADCS.Ephemeris(), J2000=0.22,
    R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]),
)
controller = ADCS.controller.BDot(est_sat=satellite, gain=5e4)
config = ADCS.MCConfig(
    orbit=lambda rng: ADCS.orbits.create_random_circular_os(
        radius_km=7000.0, J2000=0.22, rng=rng
    ),
    q=lambda rng: ADCS.helpers.normalize(rng.standard_normal(4)),
    w=lambda rng: np.deg2rad(
        rng.uniform(1.0, 5.0) * ADCS.helpers.normalize(rng.standard_normal(3))
    ),
)
results = ADCS.simulate_mc(
    x=initial_state, satellite=satellite, controller=controller,
    os0=orbit, mc_config=config, dt=10.0, tf=4000.0,
    num_runs=8, base_seed=42, max_workers=2,
)

ADCS.plot(
    results,
    ADCS.plots.AngularVelocityPlotCombined(
        title="B-dot detumbling across eight runs",
        units="rad/s",
    ),
    layout=(1, 1), figsize=(8, 4.5),
)

plt.show()
