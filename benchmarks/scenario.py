"""Shared benchmark scenario: one definition used by every runner.

All tools propagate the same rigid spacecraft on the same orbit with the same
physics and the same fixed time step:

* 12 kg spacecraft, principal inertia diag(0.13, 0.10, 0.05) kg m^2, no torques
  (torque-free rigid-body rotation), no wheels.
* 500 km circular orbit (r = 6,878,137 m) inclined 51.6 deg, central gravity
  plus J2 about the inertial z axis.
* Earth constants from GenADCS 3.0 (mu = 3.986004418e14 m^3/s^2,
  R = 6,378,137 m, J2 = 1.08262668e-3); each runner sets these where the tool
  allows it.
* Fixed-step classic RK4 with dt = 0.25 s, the state recorded at every step.

This module only needs the Python standard library, so the orchestrator can
import it from any interpreter. ``reference_solution`` needs NumPy and SciPy.
"""

from __future__ import annotations

import math

MU = 3.986004418e14  # m^3/s^2
R_EARTH = 6378137.0  # m
J2 = 1.08262668e-3

MASS = 12.0  # kg
INERTIA_DIAG = (0.13, 0.10, 0.05)  # kg m^2

DT = 0.25  # s

ALTITUDE = 500_000.0  # m
INCLINATION_DEG = 51.6

R0 = (R_EARTH + ALTITUDE, 0.0, 0.0)  # m, inertial
_V_CIRC = math.sqrt(MU / R0[0])
V0 = (0.0, _V_CIRC * math.cos(math.radians(INCLINATION_DEG)),
      _V_CIRC * math.sin(math.radians(INCLINATION_DEG)))  # m/s, inertial

# Attitude: q_b_from_i, scalar first (identity), body rates in rad/s.
Q0 = (1.0, 0.0, 0.0, 0.0)
W0 = (0.01, -0.015, 0.02)

JULIAN_DATE0 = 2_461_000.5  # epoch used by GenADCS 3.0 (time-independent physics)

ORBIT_PERIOD = 2.0 * math.pi * math.sqrt(R0[0] ** 3 / MU)  # s, two-body period

# Durations as fractions of one orbit, for the scaling chart. The last one is
# the headline run used for speed and accuracy.
DURATION_FRACTIONS = (0.125, 0.25, 0.5, 1.0)


def steps_for(fraction: float) -> int:
    """Number of dt steps covering ``fraction`` of an orbit."""
    return int(round(fraction * ORBIT_PERIOD / DT))


SCENARIOS = {
    "coupled": "Attitude + orbit: torque-free rigid body and J2 orbit, integrated together",
    "orbit": "Orbit only: J2 orbit propagation",
}


def quat_b_from_i_to_dcm(q):
    """Inertial-to-body DCM from a scalar-first q_b_from_i (Hamilton formula).

    GenADCS 3.0 propagates q_b_from_i with qdot = -1/2 [0, w] (x) q, which is
    the conjugate of the body-to-inertial Hamilton quaternion, so the standard
    Hamilton rotation matrix of q is the inertial-to-body DCM.
    """
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def dcm_angle_deg(a, b) -> float:
    """Rotation angle between two DCMs, in degrees.

    Uses ||A - B||_F = 2*sqrt(2)*sin(theta/2), which stays accurate for the
    tiny angles of an accuracy comparison (acos of the trace does not).
    """
    frob = math.sqrt(sum((a[i][k] - b[i][k]) ** 2 for i in range(3) for k in range(3)))
    return math.degrees(2.0 * math.asin(min(1.0, frob / (2.0 * math.sqrt(2.0)))))


def reference_solution(steps: int, scenario: str = "coupled"):
    """High-precision reference: DOP853 at rtol = atol-scaled 1e-13.

    Integrates exactly the scenario's equations (central gravity + J2 about the
    inertial z axis; torque-free Euler equations; GenADCS 3.0 quaternion
    kinematics) and returns the final state at ``steps * DT``.
    """
    import numpy as np
    from scipy.integrate import solve_ivp

    inertia = np.array(INERTIA_DIAG)

    def rhs(_t, y):
        q, w, r, v = y[0:4], y[4:7], y[7:10], y[10:13]
        x_, y_, z_ = r
        r2 = x_ * x_ + y_ * y_ + z_ * z_
        rn = math.sqrt(r2)
        k = 1.5 * J2 * MU * R_EARTH ** 2 / rn ** 5
        zz = 5.0 * z_ * z_ / r2
        a = -MU * r / rn ** 3 + k * np.array([x_ * (zz - 1.0), y_ * (zz - 1.0), z_ * (zz - 3.0)])
        wdot = (-np.cross(w, inertia * w)) / inertia
        qdot = 0.5 * np.array([
            np.dot(q[1:4], w),
            -(q[0] * w[0] + w[1] * q[3] - w[2] * q[2]),
            -(q[0] * w[1] + w[2] * q[1] - w[0] * q[3]),
            -(q[0] * w[2] + w[0] * q[2] - w[1] * q[1]),
        ])
        return np.concatenate([qdot, wdot, v, a])

    y0 = np.array(Q0 + W0 + R0 + V0, dtype=float)
    tf = steps * DT
    sol = solve_ivp(rhs, (0.0, tf), y0, method="DOP853", rtol=1e-13,
                    atol=np.array([1e-15] * 7 + [1e-7] * 3 + [1e-10] * 3))
    y = sol.y[:, -1]
    q = y[0:4] / np.linalg.norm(y[0:4])
    return {
        "t": tf,
        "q_b_from_i": q.tolist(),
        "w": y[4:7].tolist(),
        "r": y[7:10].tolist(),
        "v": y[10:13].tolist(),
    }
