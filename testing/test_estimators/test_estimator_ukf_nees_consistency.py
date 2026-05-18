"""
Plain (non-square-root) UAKF covariance consistency via NEES
(test-hardening backlog #9).

PR #39 added a rigorous NEES-in-parameter-space guard for the SR-UKF, but
the plain UAKF -- which uses a DIFFERENT covariance path (direct
P^- = sum w_c (dx)(dx)^T + Q, then P^+ = P^- - K^T P_yy K) rather than the
SR-UKF's choldate rank-1 factor updates -- had only the weak rate-only
">70% inside 3-sigma" check (test_estimator_ukf*::test_ukf_covariance_
consistency). That marginal check passes at ~100% even when the attitude
covariance is grossly over-confident, exactly the blind spot PR #39
addressed for the SR-UKF.

NEES = e^T P^-1 e in the filter's own reduced vec_mode=6 attitude-error
space. For a consistent filter NEES ~ chi^2(dof) (attitude dof=3 -> ~3).

PROBED NEGATIVE RESULT (recorded so it is not re-investigated): the plain
UAKF covariance reconstruction is NOT broken. Swapping ONLY the filter
class in one fixed harness (same Q/P/scenario) yields BIT-IDENTICAL
attitude NEES for UAKF and SR-UKF (1278.1 both) -- the two paths are
mathematically equivalent (the square root is just a factorisation). The
large attitude NEES in this harness is the SAME under-modeled-attitude-Q
theme as the SR-UKF's documented residual (PR #39 / process-noise backlog),
only with this harness's looser Q tuning -- it is NOT a UAKF-specific bug
and is deliberately not force-fixed (open Q-convention API decision).

This file closes the coverage asymmetry: it gives the plain UAKF the same
NEES tripwire the SR-UKF got -- enforcing the alpha sigma-spread choice and
guarding the rate block and the covariance path against gross regression --
and documents the residual with the measured number via a non-strict xfail.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

import ADCS.estimators.attitude_estimators.attitude_UAKF as _UAKF_MOD
from ADCS.helpers.math_helpers import quat_to_vec3, quat_mult, quat_inv

pytestmark = pytest.mark.slow

_HARNESS = pathlib.Path(__file__).with_name("test_estimator_ukf.py")
_VEC_MODE = 6  # must match UAKF.vec_mode


def _run():
    # alpha=1 is OPT-IN (default is the legacy 1e-3 so the shared estimator
    # suite is bit-for-bit unchanged -- see PR #39). This guard opts in, as
    # PR #39's SR-UKF guard does, so it enforces the sigma-spread choice.
    spec = importlib.util.spec_from_file_location("_ukf_h", _HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _orig = _UAKF_MOD.UAKF.__init__

    def _alpha1_init(self, *a, **k):
        _orig(self, *a, **k)
        self.al = 1.0

    _UAKF_MOD.UAKF.__init__ = _alpha1_init
    try:
        return mod.run_ukf(verbose=False, tf=1000, dt=50, real_orbit=True)
    finally:
        _UAKF_MOD.UAKF.__init__ = _orig


def _nees(results):
    (_, sh, esh, _, _, _, _, ch) = results
    P = np.array(ch)
    N = len(P)
    s = N // 2 if N // 2 != N else 0
    qt, qe = sh[:, 3:7], esh[:, 3:7]

    att, rate_in = [], []
    for k in range(s, N):
        dq = quat_mult(quat_inv(qe[k] / np.linalg.norm(qe[k])),
                       qt[k] / np.linalg.norm(qt[k]))
        if dq[0] < 0:
            dq = -dq
        e = quat_to_vec3(dq, _VEC_MODE)            # SAME space/gain as P[3:6,3:6]
        Pa = P[k, 3:6, 3:6]
        try:
            att.append(float(e @ np.linalg.solve(Pa, e)))
        except np.linalg.LinAlgError:
            pass
        rsig = 3.0 * np.sqrt(np.diag(P[k, 0:3, 0:3]))
        rerr = np.abs(sh[k, 0:3] - esh[k, 0:3])
        rate_in.append(bool(np.all(rerr <= rsig)))
    return np.array(att), float(np.mean(rate_in))


@pytest.fixture(scope="module")
def nees_results():
    return _nees(_run())


def test_ukf_attitude_nees_is_finite_and_bounded(nees_results):
    """Covariance-path health + alpha enforcement. The plain UAKF's direct
    P reconstruction must stay finite and PSD enough that NEES is finite,
    and with the alpha=1 sigma-spread the attitude NEES must be materially
    below the legacy-alpha collapse (~1645) -- a bound of 1450 is GREEN at
    alpha=1 (~1278) and RED if alpha regresses or the covariance path
    grossly degrades."""
    att, _ = nees_results
    assert att.size > 0, "no finite NEES samples -> covariance path produced singular P"
    mean_nees = float(att.mean())
    assert np.isfinite(mean_nees)
    print(f"plain-UAKF attitude NEES (param space, dof=3): mean={mean_nees:.1f} "
          f"median={np.median(att):.1f}")
    assert mean_nees < 1450.0, (
        f"attitude NEES {mean_nees:.1f} >= 1450 -> sigma-spread collapsed "
        f"(alpha regressed) or the UAKF covariance reconstruction degraded.")


def test_ukf_rate_block_consistency_regression_guard(nees_results):
    """Rate block must stay within 3-sigma most of the time (the property
    the legacy weak check verified) -- kept as an explicit regression
    tripwire on the plain UAKF path."""
    _, rate_in = nees_results
    assert rate_in > 0.70, f"rate-block 3-sigma consistency regressed: {rate_in:.0%}"


@pytest.mark.xfail(strict=False, reason=(
    "Residual attitude over-confidence (~1278 NEES vs consistent ~3) under "
    "this harness's Q tuning. VERIFIED (probed) to be IDENTICAL for the "
    "plain UAKF and the SR-UKF -- i.e. the UAKF covariance reconstruction "
    "is correct/equivalent, not a path bug; it is the shared under-modeled-"
    "attitude-Q theme tracked in the SRUKF process-noise backlog and "
    "deliberately not force-fixed (open Q-convention API decision)."))
def test_ukf_attitude_fully_consistent(nees_results):
    att, _ = nees_results
    assert att.mean() < 10.0, (
        f"attitude NEES {att.mean():.1f} not within ~dof+margin")
