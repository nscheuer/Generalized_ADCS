"""
Generalized ADCS Framework — 3U CubeSat nadir pointing WITH estimation.
Scenario: 3U CubeSat, 3 MTQ + 1 RW, nadir pointing, ISS orbit, 1 orbit.
         SRUAKF attitude estimator in the loop.

Note: The BeaverCube 2 satellite factory includes magnetometers, gyroscopes,
sun sensors (SunPair on two solar panels), reaction wheel, magnetorquers,
and disturbances (gravity gradient, drag, SRP). Eclipse is handled
automatically — sun sensors return NaN when the spacecraft is in shadow.
"""
import os, sys, time
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
import numpy as np
from scipy.linalg import block_diag
import ADCS

# ── Spacecraft (true and estimated) ───────────────────────────────────
# True satellite from factory (includes sensor noise, disturbances, sun sensors).
# Estimated satellite matches but without bias estimation — fair comparison
# since Basilisk's simpleNav provides perfect-knowledge nav.
sat     = ADCS.satellite_factory.create_beavercube2_cubesat(estimated=False)
est_sat = ADCS.satellite_factory.create_beavercube2_cubesat(estimated=True)

# ── Initial state: [omega(3), quaternion(4), h_rw(1)] ────────────────
x_0 = ADCS.State.from_array(np.array([0.5*np.pi/180, -1.0*np.pi/180, 2.0*np.pi/180,
                 0.9624, 0.1451, -0.0960, 0.2101,
                 0.0]))

# ── ISS orbit ─────────────────────────────────────────────────────────
R_iss = np.array([6779.0, 0.0, 0.0])
V_iss = np.array([0.0, 4.825, 5.694])
dt = 1.0
os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(), J2000=0.25, R=R_iss, V=V_iss)

# ── Estimator (SRUAKF) ───────────────────────────────────────────────
# State: [omega(3), quat(4), h_rw(1), act_bias(4), sens_bias(8)] = 20
# P_hat, Q_hat: (N-1)x(N-1) = 19x19  (quat(4) → MRP(3))
x_hat = ADCS.EstimatorState(
    w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0], h=np.zeros(1),
    act_bias=np.zeros(4), sens_bias=np.zeros(8),
)
P_hat = block_diag(
    np.eye(3) * 0.01**2,       # angular velocity
    np.eye(3) * 1.0,           # attitude (MRP)
    np.eye(1) * 0.001**2,      # RW momentum
    np.eye(4) * 0.01**2,       # actuator biases
    np.eye(8) * 0.01**2,       # sensor biases
)
Q_hat = block_diag(
    np.eye(3) * 1e-8**2,       # process noise: rates
    np.eye(3) * 1e-8,          # process noise: attitude
    np.eye(1) * 1e-10**2,      # process noise: RW
    np.eye(4) * 1e-10**2,      # process noise: act biases
    np.eye(8) * 1e-10**2,      # process noise: sens biases
)
estimator = ADCS.UAKF(
    est_sat=est_sat, J2000=os0.J2000,
    x_hat=x_hat, P_hat=P_hat, Q_hat=Q_hat, dt=dt,
    cross_term=True
)

# ── Goal + Controller ─────────────────────────────────────────────────
goal = ADCS.goals.Nadir_Goal()
controller = ADCS.controller.MTQ_w_RW_LP(
    est_sat=est_sat, p_gain=5e-5, d_gain=2e-3, c_gain=1e-3
)

# ── Run ───────────────────────────────────────────────────────────────
t_start = time.perf_counter()
results = ADCS.simulate(
    x=x_0, satellite=sat, est_satellite=est_sat,
    controller=controller, estimator=estimator,
    goal=goal, os0=os0, dt=dt, tf=5400.0
)
elapsed = time.perf_counter() - t_start
print(f"ADCS Framework (with estimation) — elapsed: {elapsed:.2f} s  (sim time: 5400 s)")
