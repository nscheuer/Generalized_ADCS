"""
SRUKF covariance-consistency checks for rate and attitude uncertainty.

This module reuses the existing deterministic ``run_srukf()`` harness from
``test_estimator_srukf.py`` and inspects how often the true error stays inside
the filter's reported 3-sigma bounds after the initial transient.

The helper path is:

1. Run the seeded SRUKF scenario once through ``_run_srukf()``.
2. Compute per-axis rate errors directly from the true and estimated state
   histories.
3. Convert quaternion attitude error into a small-angle rotation vector, then
   compare each axis against the attitude covariance block ``P[:, 3:6, 3:6]``.

The assertions are intentionally split by status:

* ``test_srukf_rate_block_consistency_regression_guard`` is a hard regression
  test that requires each rate axis to stay inside its reported 3-sigma bound
  more than 70% of the time.
* ``test_srukf_attitude_block_consistency`` is an ``xfail`` tracker for the
  known over-confident attitude covariance block; it preserves a reproducible
  measurement of the gap without forcing a speculative filter fix here.
"""

import importlib.util
import pathlib
import numpy as np
import pytest

pytestmark = pytest.mark.slow

_SRUKF_FILE = pathlib.Path(__file__).with_name("test_estimator_srukf.py")


def _run_srukf():
    spec = importlib.util.spec_from_file_location("_srukf_harness", _SRUKF_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Same configuration as the existing srukf_results fixture.
    return mod.run_srukf(verbose=False, tf=1000, dt=50, real_orbit=True)


def _attitude_error_angles(q_true, q_est):
    """Per-axis small-angle attitude error (rad), scalar-first quaternions."""
    def conj(q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def mult(a, b):
        w0, x0, y0, z0 = a
        w1, x1, y1, z1 = b
        return np.array([
            w0*w1 - x0*x1 - y0*y1 - z0*z1,
            w0*x1 + x0*w1 + y0*z1 - z0*y1,
            w0*y1 - x0*z1 + y0*w1 + z0*x1,
            w0*z1 + x0*y1 - y0*x1 + z0*w1,
        ])

    out = np.zeros((len(q_true), 3))
    for k in range(len(q_true)):
        qe = mult(q_true[k] / np.linalg.norm(q_true[k]),
                  conj(q_est[k] / np.linalg.norm(q_est[k])))
        if qe[0] < 0:
            qe = -qe
        out[k] = 2.0 * qe[1:4]      # small-angle rotation vector
    return out


def _consistency(results):
    (_, state_hist, est_state_hist, _, _, _, _, cov_hist) = results
    P = np.array(cov_hist)
    N = len(P)
    s = N // 2 if N // 2 != N else 0

    rate_sig3 = 3.0 * np.sqrt(P[:, 0:3, 0:3].diagonal(axis1=1, axis2=2))
    rate_err = np.abs(state_hist[:, 0:3] - est_state_hist[:, 0:3])
    rate_pct = [
        np.mean(rate_err[s:, a] <= rate_sig3[s:, a]) for a in range(3)
    ]

    att_sig3 = 3.0 * np.sqrt(P[:, 3:6, 3:6].diagonal(axis1=1, axis2=2))
    att_err = np.abs(_attitude_error_angles(state_hist[:, 3:7],
                                            est_state_hist[:, 3:7]))
    att_pct = [
        np.mean(att_err[s:, a] <= att_sig3[s:, a]) for a in range(3)
    ]
    return rate_pct, att_pct


def test_srukf_rate_block_consistency_regression_guard():
    rate_pct, att_pct = _consistency(_run_srukf())
    print(f"rate in-3sigma per axis : {['%.0f%%' % (100*p) for p in rate_pct]}")
    print(f"attitude in-3sigma/axis : {['%.0f%%' % (100*p) for p in att_pct]}")
    for a, p in enumerate(rate_pct):
        assert p > 0.70, f"rate-block consistency regressed on axis {a}: {p:.0%}"


@pytest.mark.xfail(strict=False, reason=(
    "Known: SRUKF attitude covariance block is over-confident (the analytic "
    "attitude error-state process noise / S-factor reconstruction under-"
    "represents attitude uncertainty). Tracked, not yet fixed; this test "
    "reports the measured in-3-sigma percentage."))
def test_srukf_attitude_block_consistency():
    _, att_pct = _consistency(_run_srukf())
    worst = min(att_pct)
    # A consistent filter keeps the true error inside 3-sigma the large
    # majority of the time. The attitude block currently does not.
    assert worst > 0.70, (
        f"attitude-block 3-sigma consistency too low: per-axis {att_pct} "
        f"(worst {worst:.0%}); attitude covariance is over-confident."
    )
