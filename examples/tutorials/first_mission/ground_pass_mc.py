"""Monte Carlo ground-pass pointing with three reaction wheels."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt
import numpy as np
import ADCS
from ADCS.helpers.math_helpers import quat_mult

wheels = [ADCS.RW(axis=axis, max_torque=1e-3, J=5e-6,
                  h=0.0, h_max=0.02) for axis in np.eye(3)]
# This controller accepts MTQs for momentum unloading. c_gain=0 below
# leaves them idle; the three wheels provide pointing torque.
magnetorquers = [ADCS.MTQ(axis=axis, max_torque=0.1) for axis in np.eye(3)]
magnetometers = [ADCS.MTM(axis=axis) for axis in np.eye(3)]
satellite = ADCS.Satellite(
    mass=4.0, J_0=np.diag([0.0333, 0.0333, 0.00667]),
    actuators=magnetorquers + wheels, sensors=magnetometers,
    boresight=np.array([0.0, 0.0, 1.0]),
)
orbit = ADCS.Orbital_State(
    ephem=ADCS.Ephemeris(), J2000=0.22,
    R=np.array([0.0, -7000.0, 0.0]), V=np.array([7.5, 0.0, 0.0]),
)
yaw = np.array([np.cos(np.pi / 6), 0.0, 0.0, np.sin(np.pi / 6)])
pitch = np.array([np.cos(np.pi / 3), 0.0, np.sin(np.pi / 3), 0.0])
nominal_q = quat_mult(yaw, pitch)
initial_state = ADCS.State(w=np.zeros(3), q=nominal_q, h=np.zeros(3))
controller = ADCS.controller.MTQ_w_RW(
    est_sat=satellite, p_gain=1e-4, d_gain=5e-3,
    c_gain=0.0, h_target=np.zeros(3),
)
goal = ADCS.goals.Coordinate_Goal(lat=0.0, lon=-176.6, alt=0.0)

def sampled_pass_orbit(rng):
    """Small injection errors keep each run near the same ground pass."""
    return ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(), J2000=0.22,
        R=orbit.R + rng.normal(0.0, 10.0, 3),
        V=orbit.V + rng.normal(0.0, 0.01, 3),
    )

config = ADCS.MCConfig(
    orbit=sampled_pass_orbit,
    q=lambda rng: ADCS.helpers.normalize(nominal_q + rng.normal(0.0, 0.06, 4)),
    w=lambda rng: np.deg2rad(rng.uniform(-0.1, 0.1, 3)),
)
results = ADCS.simulate_mc(
    x=initial_state, satellite=satellite, controller=controller,
    goal=goal, os0=orbit, mc_config=config,
    dt=1.0, tf=480.0, num_runs=8, base_seed=42, max_workers=2,
)
ADCS.plot(results, ADCS.plots.TargetPlot(modes=["real_target"],
          title="Ground-pass pointing error"), layout=(1, 1), figsize=(8, 4.5))

plt.show()
