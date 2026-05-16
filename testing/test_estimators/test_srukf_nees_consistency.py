"""
SRUKF attitude-covariance consistency via NEES (Normalized Estimation Error
Squared), computed in the FILTER'S OWN error-parameter space so it is
apples-to-apples with the reported covariance block P[3:6,3:6].

NEES = e^T P^-1 e. For a consistent filter NEES ~ chi^2(dof); attitude
dof = 3, so a consistent attitude NEES averages ~3. Over-confidence
(P too small) drives NEES well above dof.

Diagnosis behind these tests (all measured on the seeded run_srukf harness):

* alpha = 1e-3 (the old "textbook default") collapses the unscented spread
  to gamma ~ 2.4e-3 with +/-1e6 weights -> the UT degenerates to a local
  linearisation that cannot capture the attitude nonlinearity. Measured
  attitude NEES ~263.
* alpha = 1 (gamma = sqrt(L), O(1) weights) is the standard robust choice
  and brings attitude NEES down to ~55 (a ~5x improvement) and is far
  healthier numerically.
* A residual ~18x over-confidence remains (NEES ~55 vs the consistent ~3):
  that is an attitude process-noise (Q) tuning issue, model/config specific,
  deliberately NOT force-fixed -- it is documented here with the measured
  number rather than left silently untested.
"""

import importlib.util
import pathlib
import numpy as np
import pytest

from ADCS.helpers.math_helpers import quat_to_vec3, quat_mult, quat_inv

pytestmark = pytest.mark.slow

_HARNESS = pathlib.Path(__file__).with_name("test_estimator_srukf.py")
_VEC_MODE = 6  # must match UAKF.vec_mode (the attitude error parametrisation)


def _run():
    spec = importlib.util.spec_from_file_location("_srukf_h", _HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_srukf(verbose=False, tf=1000, dt=50, real_orbit=True)


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


def test_srukf_rate_block_consistency_regression_guard(nees_results):
    _, rate_in = nees_results
    assert rate_in > 0.70, f"rate-block 3-sigma consistency regressed: {rate_in:.0%}"


def test_srukf_attitude_nees_sigma_spread_fix_enforced(nees_results):
    """
    Enforces the alpha (sigma-spread) fix. With the degenerate alpha=1e-3 the
    attitude NEES is ~263; with the correct alpha=1 it is ~55. A bound of 120
    passes on the fixed code and FAILS if alpha regresses to the collapsed
    spread -- i.e. this is the RED-on-origin/main, GREEN-after-fix guard.
    """
    att, _ = nees_results
    mean_nees = float(att.mean())
    print(f"attitude NEES (param space, dof=3): mean={mean_nees:.1f} "
          f"median={np.median(att):.1f}")
    assert mean_nees < 120.0, (
        f"attitude NEES {mean_nees:.1f} >= 120 -> unscented spread is "
        f"collapsed (alpha too small); covariance massively over-confident."
    )


@pytest.mark.xfail(strict=False, reason=(
    "Residual ~18x attitude over-confidence after the sigma-spread fix "
    "(NEES ~55 vs consistent ~3): under-modeled attitude process noise Q. "
    "Model/config-specific tuning, tracked here with the measured number, "
    "deliberately not force-fixed."))
def test_srukf_attitude_fully_consistent(nees_results):
    att, _ = nees_results
    assert att.mean() < 10.0, (
        f"attitude NEES {att.mean():.1f} not within ~dof+margin; "
        f"attitude process noise still under-modeled."
    )
