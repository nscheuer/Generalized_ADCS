"""Roll guidance -- NOT USED IN THE CAMPAIGN. Recorded as an observation, with the reason it is rejected.

.. warning::

   **This couples control to estimation, and the campaign must not do that.**

   The paper's frontier argument is that the limit is *actuation*, not estimation. Steering
   roll to satisfy the star tracker makes the control command a function of the estimator's
   needs, and the two can no longer be separated by construction -- the very separation the
   campaign is built to demonstrate.

   It also breaks Campaign A directly. Commanding roll converts the reduced-attitude cells
   into full-attitude cells with a particular roll target, so the "reduced vs full" comparison
   would no longer compare goal types: both arms would be full-attitude. That is the
   comparison Campaign A exists to make.

   The right fix for tracker availability is hardware and mounting -- a second tracker, which
   reaches 0.909 with no coupling at all -- or accepting one tracker's availability and
   reporting it honestly as a sensor-suite limit.

   Kept here because the geometry is a real and non-obvious observation worth a sentence in
   future work, not because the campaign should use it.

Original note follows.

Spend the reduced-attitude task's free degree of freedom on star-tracker visibility.

A boresight-pointing (reduced-attitude) task constrains two degrees of freedom and leaves the
third -- roll about the boresight -- entirely free. Campaign A left it uncommanded, so it
drifted wherever the dynamics took it, and the star tracker was blinded 55% of the orbit.

Spending that freedom deliberately is nearly free and beats adding hardware:

===================================  ==========
configuration                        availability
===================================  ==========
1 tracker, roll uncommanded          0.456
2 trackers opposed, roll uncommanded 0.909
**1 tracker, roll chosen**           **0.998**
===================================  ==========

The geometry is simple. With the tracker mounted at ``beta`` from the payload boresight, roll
sweeps it around a cone of half-angle ``beta`` about the boresight, so its angle from nadir
ranges over ``[|gamma - beta|, gamma + beta]`` where ``gamma`` is the target's angle from
nadir. A fix exists for *some* roll whenever ``gamma + beta > keepout``. At ``beta = 90 deg``
and a 95.2 deg keep-out that means any target more than **5.2 deg off nadir** -- essentially
always.

**This is available to the reduced-attitude task only.** A full-attitude goal dictates roll,
so it cannot be spent, and a single tracker stays at 0.456. Goal type therefore affects
pointing *knowledge*, not just controllability -- which is a sharper version of the
"goal type is a design lever" argument than the control-side one.

The cost is that roll is no longer free for anything else, and the manoeuvre to hold a
commanded roll consumes control authority that a purely reduced-attitude task would not spend.
Campaign A measures that rather than assuming it is negligible.
"""

from __future__ import annotations

__all__ = ["roll_optimal_quaternion", "TrackerAwareGoal"]

from typing import Optional, Sequence

import numpy as np

from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.helpers.math_helpers import normalize
from ADCS.orbits.universal_constants import EarthConstants


def _basis_about(b_hat: np.ndarray):
    """Two unit vectors completing a right-handed frame with ``b_hat``."""
    seed = np.array([1.0, 0.0, 0.0]) if abs(b_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(seed, b_hat)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(b_hat, e1)


def roll_optimal_quaternion(target_eci: np.ndarray,
                            boresight_body: np.ndarray,
                            tracker_body: np.ndarray,
                            R_eci: np.ndarray,
                            n_roll: int = 72) -> np.ndarray:
    """Attitude putting ``boresight_body`` on ``target_eci`` with roll maximising tracker elevation.

    Returns a quaternion (body->ECI). Roll is searched on a coarse grid -- the objective is a
    single smooth maximum over the circle, so a grid is both adequate and cheap, and it avoids
    an optimiser inside the control loop.

    :param target_eci: Desired inertial direction for the payload boresight.
    :param boresight_body: Payload boresight in the body frame.
    :param tracker_body: Star-tracker boresight in the body frame.
    :param R_eci: Spacecraft position (ECI), for the nadir direction.
    """
    t_hat = normalize(np.asarray(target_eci, float))
    b_hat = normalize(np.asarray(boresight_body, float))
    s_hat = normalize(np.asarray(tracker_body, float))
    R = np.asarray(R_eci, float)
    r = float(np.linalg.norm(R))
    nadir = -R / r

    # Any attitude taking b_hat -> t_hat, then roll about t_hat.
    v = np.cross(b_hat, t_hat)
    c = float(np.dot(b_hat, t_hat))
    if np.linalg.norm(v) < 1e-12:
        C0 = np.eye(3) if c > 0 else -np.eye(3) + 2 * np.outer(b_hat, b_hat)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        C0 = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))

    e1, e2 = _basis_about(t_hat)
    best_ang, best_C = -np.inf, C0
    for psi in np.linspace(0.0, 2.0 * np.pi, n_roll, endpoint=False):
        # Rodrigues rotation about the (fixed) target direction.
        K = np.array([[0, -t_hat[2], t_hat[1]],
                      [t_hat[2], 0, -t_hat[0]],
                      [-t_hat[1], t_hat[0], 0]])
        Rr = np.eye(3) + np.sin(psi) * K + (1 - np.cos(psi)) * (K @ K)
        C = Rr @ C0
        s_eci = C @ s_hat
        ang = np.arccos(np.clip(float(s_eci @ nadir), -1.0, 1.0))   # from nadir
        if ang > best_ang:
            best_ang, best_C = ang, C

    # Rotation matrix -> quaternion (scalar first).
    C = best_C
    tr = np.trace(C)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (C[2, 1] - C[1, 2]) / s,
                      (C[0, 2] - C[2, 0]) / s, (C[1, 0] - C[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(C)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + C[i, i] - C[j, j] - C[k, k]) * 2
        q = np.zeros(4)
        q[0] = (C[k, j] - C[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (C[j, i] + C[i, j]) / s
        q[k + 1] = (C[k, i] + C[i, k]) / s
    return normalize(q)


class TrackerAwareGoal:
    """Reduced-attitude goal that also commands the roll the tracker wants.

    Presents as a full-attitude goal whose target quaternion is recomputed as the geometry
    changes, so the existing controllers need no modification. The boresight constraint is
    identical to the plain reduced-attitude goal; only the otherwise-free roll differs.
    """

    def __init__(self, target_eci, boresight_body, tracker_body,
                 recompute_every_s: float = 60.0):
        self.target_eci = normalize(np.asarray(target_eci, float))
        self.boresight_body = normalize(np.asarray(boresight_body, float))
        self.tracker_body = normalize(np.asarray(tracker_body, float))
        # Roll is recomputed periodically rather than every step: chasing it continuously
        # would inject a command the tracker geometry does not actually require and would
        # spend control authority on tracking numerical jitter.
        self.recompute_every_s = float(recompute_every_s)
        self._last_t = -np.inf
        self._goal: Optional[Fixed_Attitude_Goal] = None

    def goal_at(self, t_s: float, os) -> Fixed_Attitude_Goal:
        if self._goal is None or (t_s - self._last_t) >= self.recompute_every_s:
            q = roll_optimal_quaternion(self.target_eci, self.boresight_body,
                                        self.tracker_body, np.asarray(os.R, float))
            self._goal = Fixed_Attitude_Goal(q)
            self._last_t = t_s
        return self._goal
