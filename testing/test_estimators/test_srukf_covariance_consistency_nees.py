"""
Rigorous SRUKF covariance-consistency (NEES) detection test.

WHY THIS EXISTS
---------------
The pre-existing ``test_srukf_covariance_consistency`` only checks the *rate*
block on a single seed, so it never caught that the SRUKF's reported attitude
covariance is meaningless. Verified by probe (in the filter's OWN mode-6
parametrisation, so this is not a units artefact):

* rate-block NEES ~ 2.4            -> consistent (healthy)
* attitude-block NEES ~ 3.5e12     -> expected ~3; the posterior attitude
                                      covariance collapses to the process-
                                      noise floor (``P+`` attitude eigenvalues
                                      == ``Q_est`` 1e-4 at EVERY step, all
                                      seeds), while the true attitude error
                                      grows to ~23 deg.

LOCALISATION (verified)
-----------------------
The collapse is produced inside ``SRUAKF.update_core`` itself: the value it
returns in ``EstimatedArray.cov`` is already collapsed (the
``set_indices``/``int_cov`` write-back path is faithful and is NOT the cause).

REFUTED hypotheses (by probe, do not re-chase without new evidence):
  * ``quat_to_mrp``/``mrp_to_quat`` factor-2 inverse mismatch  -> sound
  * scaled-sigma ``alpha=1e-3`` -> ``|W0c|~1e6`` downdate       -> alpha=1e-1
    did not help (still ~2.9e12)

NEXT DIAGNOSTIC for the scoped SRUKF rework: instrument ``update_core`` to log
the attitude block of the predicted square-root covariance ``srcov1`` vs
``S_Q``; suspected mechanism is that the predicted attitude unscented spread
is ~0 (quaternion reduction in ``new_post_state`` relative to the 0th
propagated sigma vs the attitude spread injected by ``make_pts_and_wts``), so
``P+`` reduces to the process-noise residual.

This test is ``xfail(strict=True)``: it documents the verified bug without
breaking CI, and will turn into an enforced failure (XPASS) the moment the
SRUKF is fixed, forcing this marker to be removed.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_estimator_srukf import run_srukf  # noqa: E402

from ADCS.helpers.math_helpers import quat_inv, quat_mult, quat_to_vec3  # noqa: E402

VEC_MODE = 6  # SRUAKF.vec_mode -- compute the attitude error in the SAME param


def _nees(seed: int = 3):
    np.random.seed(seed)
    out = run_srukf(verbose=False, tf=1000, dt=50, real_orbit=True)
    state_hist, est_hist, cov_hist = out[1], out[2], out[7]
    full, rate, att = [], [], []
    for k in range(len(cov_hist)):
        xt, xe, P = state_hist[k], est_hist[k], cov_hist[k]
        if xt is None or xe is None or P is None:
            continue
        if np.any(np.isnan(xt)) or np.any(np.isnan(xe)):
            continue
        w_err = xt[0:3] - xe[0:3]
        dq = quat_mult(quat_inv(xe[3:7]), xt[3:7])
        a3 = quat_to_vec3(dq, VEC_MODE)
        dx = np.concatenate([w_err, a3])
        Pr = np.asarray(P, dtype=float)[:6, :6]
        try:
            full.append(dx @ np.linalg.inv(Pr) @ dx)
            rate.append(w_err @ np.linalg.inv(Pr[:3, :3]) @ w_err)
            att.append(a3 @ np.linalg.inv(Pr[3:6, 3:6]) @ a3)
        except np.linalg.LinAlgError:
            continue
    assert full, "no valid SRUKF steps to evaluate"
    return float(np.mean(full)), float(np.mean(rate)), float(np.mean(att))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "VERIFIED BUG: SRUAKF.update_core posterior attitude covariance "
        "collapses to the Q process-noise floor; attitude NEES ~3.5e12 vs "
        "expected ~3 (rate block consistent ~2.4). Localised to update_core. "
        "Scoped for SRUKF rework -- see module docstring for the next "
        "diagnostic. Remove this marker when fixed."
    ),
)
def test_srukf_full_state_nees_is_consistent():
    full_nees, rate_nees, att_nees = _nees(seed=3)

    # Methodology sanity: the rate block IS consistent, proving the NEES
    # computation/units are correct and the defect is attitude-specific.
    assert 0.2 < rate_nees < 9.0, (
        f"rate NEES {rate_nees:.3g} out of chi2(3) sanity band -- the test "
        f"harness itself is suspect, not just the attitude block"
    )

    # The actual consistency requirement. chi2(6): mean 6, 95% band ~[1.2, 14].
    # Currently fails hard (full ~ att ~ 3.5e12) -> xfail.
    assert att_nees < 14.0, (
        f"attitude-block NEES {att_nees:.3g} (expected ~3): SRUKF attitude "
        f"covariance is not consistent with the actual error"
    )
    assert 1.2 < full_nees < 14.4, (
        f"full-state NEES {full_nees:.3g} outside chi2(6) 95% band [1.2, 14.4]"
    )
