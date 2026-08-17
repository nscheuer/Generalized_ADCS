"""
Generalized ADCS Framework — 3U CubeSat nadir-pointing simulation.
Scenario: 3U CubeSat, 3 MTQ + 1 RW, nadir pointing, ISS orbit, 1 orbit.
"""
import os, sys, time
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
import numpy as np
import ADCS

# ── Spacecraft ────────────────────────────────────────────────────────
sat = ADCS.satellite_factory.create_beavercube2_cubesat()

# ── Initial state: [omega(3), quaternion(4), h_rw(1)] ────────────────
x_0 = ADCS.State.from_array(np.array([0.5*np.pi/180, -1.0*np.pi/180, 2.0*np.pi/180,   # tumble rates [rad/s]
                 0.9624, 0.1451, -0.0960, 0.2101,                  # ~equiv to sigma=[0.3,-0.2,0.1]
                 0.0]))                                             # RW momentum

# ── ISS orbit (408 km, 51.6 deg) ─────────────────────────────────────
R_iss = np.array([6779.0, 0.0, 0.0])           # km, ECI
V_iss = np.array([0.0, 4.825, 5.694])          # km/s (inclined 51.6 deg)
os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(), J2000=0.25, R=R_iss, V=V_iss)

# ── Goal ──────────────────────────────────────────────────────────────
goal = ADCS.goals.Nadir_Goal()

# ── Controller (PD + cross-product allocation, matches Basilisk mrpFeedback) ─
controller = ADCS.controller.MTQ_w_RW(
    est_sat=sat, p_gain=5e-5, d_gain=2e-3, c_gain=1e-3,
    h_target=np.array([0, 0, 0])
)

# ── Run ───────────────────────────────────────────────────────────────
t_start = time.perf_counter()
results = ADCS.simulate(
    x=x_0, satellite=sat, controller=controller,
    goal=goal, os0=os0, dt=1.0, tf=5400.0
)
elapsed = time.perf_counter() - t_start
print(f"ADCS Framework — elapsed: {elapsed:.2f} s  (sim time: 5400 s)")
