from __future__ import annotations

import numpy as np

from ADCS.estimators.old_attitude_estimators import SRUAKF
from ADCS.state import EstimatorState
from testing.test_estimators.ukf.helpers import *  # noqa: F401,F403


def make_srukf(
    est_sat,
    *,
    x_hat: EstimatorState | None = None,
    P_hat: np.ndarray | None = None,
    Q_hat: np.ndarray | None = None,
    dt: float = 5.0,
    cross_term: bool = False,
    quat_as_vec: bool = False,
) -> SRUAKF:
    guess = make_estimate_guess(est_sat) if x_hat is None else x_hat.copy()
    if quat_as_vec:
        P = full_state_cov(est_sat) if P_hat is None else np.asarray(P_hat, dtype=float)
        Q = np.eye(guess.augmented_size) * 1.0e-6 if Q_hat is None else np.asarray(Q_hat, dtype=float)
    else:
        P = reduced_state_cov(est_sat) if P_hat is None else np.asarray(P_hat, dtype=float)
        Q = reduced_process_cov(est_sat, dt=dt) if Q_hat is None else np.asarray(Q_hat, dtype=float)
    return SRUAKF(est_sat=est_sat, J2000=0.22, x_hat=guess, P_hat=P, Q_hat=Q, dt=dt, cross_term=cross_term, quat_as_vec=quat_as_vec)
