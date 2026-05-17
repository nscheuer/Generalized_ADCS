"""
Regression guard for the SRUKF Van Loan continuous->discrete process-noise
discretisation (`vanloan_Q=True`).

Defect (raw per-step Q_hat, no dt scaling): the attitude block is the
integral of the body rate, so it was short by ~dt and the filter was
catastrophically attitude-over-confident -- attitude NEES ~= 55 (consistent
value = dof = 3), unbounded-growing with dt.

Fix: exact Van Loan with the nilpotent kinematic F
    Q_d = M*dt + (F M + M F^T) dt^2/2 + (F M F^T) dt^3/3
(generalises Crassidis to the full augmented state). Validated here under
*genuine, physical* unmodeled dynamics (a truth-only white torque the
filter's model does not contain) with a SINGLE fixed physical process PSD:

  - the rate block is consistent AND dt-invariant (NEES ~ dof at every dt),
  - the attitude block is in the consistent band (~1-3), the ~55
    over-confidence is gone, with only a mild safe-direction residual.

NEES = e^T P^-1 e in the filter's own vec_mode-6 error space (apples-to-
apples with P), dof = 3. See ~/adcs-srukf-process-noise-notes.md (paper).

Marked slow (multi-dt full SRUKF runs).
"""

import importlib.util
import pathlib
import numpy as np
import pytest

from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
import ADCS.estimators.attitude_estimators.attitude_SRUAKF as SRMOD
from ADCS.helpers.math_helpers import quat_to_vec3, quat_mult, quat_inv

pytestmark = pytest.mark.slow

_HARNESS = pathlib.Path(__file__).with_name("test_estimator_srukf.py")
_SIGMA = 2e-6      # N*m truth-only white torque (genuine unmodeled dynamics)
_QR = 1e-12        # fixed physical rate process PSD (one value, ALL dt)
_DTS = (25, 50, 100)


def _nees_curve():
    """Returns {dt: (attitude_NEES_mean, rate_NEES_mean)} with the Van Loan
    path active and a physical truth-only disturbance, ONE fixed PSD."""
    orig_dt = Satellite.dist_torques
    orig_init = SRMOD.SRUAKF.__init__

    def patched_dist_torques(self, *a, **k):
        t = np.asarray(orig_dt(self, *a, **k), float).reshape(3)
        if not isinstance(self, EstimatedSatellite):       # truth only
            t = t + np.random.normal(0.0, _SIGMA, 3)
        return t

    def patched_init(self, *a, **k):
        # Force the Van Loan path and a fixed physical rate-only PSD.
        Q = np.array(k['Q_hat'], float) if 'Q_hat' in k else np.array(a[4], float)
        Qp = np.zeros_like(Q)
        Qp[0:3, 0:3] = _QR * np.eye(3)
        if 'Q_hat' in k:
            k['Q_hat'] = Qp
        else:
            a = list(a); a[4] = Qp; a = tuple(a)
        k['vanloan_Q'] = True
        orig_init(self, *a, **k)
        # The Van Loan dt-invariance result was validated at alpha=1, which is
        # now OPT-IN (default legacy 1e-3, see #39 / notes 7i). Opt in here so
        # this guard reproduces the validated regime.
        self.al = 1.0

    out = {}
    try:
        Satellite.dist_torques = patched_dist_torques
        SRMOD.SRUAKF.__init__ = patched_init
        for dt in _DTS:
            spec = importlib.util.spec_from_file_location(f"_h{dt}", _HARNESS)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            res = m.run_srukf(verbose=False, tf=2000, dt=dt, real_orbit=True)
            (_, sh, esh, _, _, _, _, ch) = res
            P = np.array(ch); N = len(P); s = N // 2
            att, rate = [], []
            for k in range(s, N):
                qe = esh[k, 3:7] / np.linalg.norm(esh[k, 3:7])
                qt = sh[k, 3:7] / np.linalg.norm(sh[k, 3:7])
                dq = quat_mult(quat_inv(qe), qt)
                if dq[0] < 0:
                    dq = -dq
                e = quat_to_vec3(dq, 6)
                try:
                    att.append(e @ np.linalg.solve(P[k, 3:6, 3:6], e))
                except np.linalg.LinAlgError:
                    pass
                re = sh[k, 0:3] - esh[k, 0:3]
                rate.append(re @ np.linalg.solve(P[k, 0:3, 0:3], re))
            out[dt] = (float(np.mean(att)), float(np.mean(rate)))
    finally:
        Satellite.dist_torques = orig_dt
        SRMOD.SRUAKF.__init__ = orig_init
    return out


def test_srukf_vanloan_dt_invariant_consistency():
    curve = _nees_curve()
    for dt, (a, r) in curve.items():
        print(f"dt={dt}: attitude NEES={a:.2f}  rate NEES={r:.2f}")

    rates = np.array([curve[d][1] for d in _DTS])
    atts = np.array([curve[d][0] for d in _DTS])

    # (1) Rate block: consistent AND dt-invariant (dof=3). Generous band
    # around the validated 3.21 / 2.35 / 3.56; the key point is it is neither
    # crushed (~0.05, the blanket-fix failure) nor drifting unboundedly.
    for dt in _DTS:
        assert 1.0 < curve[dt][1] < 8.0, \
            f"rate NEES {curve[dt][1]:.2f} at dt={dt} not consistent (dof=3)"
    assert rates.max() / rates.min() < 3.0, \
        f"rate NEES not dt-invariant across {_DTS}: {rates}"

    # (2) Attitude block: the ~55x over-confidence MUST be gone at every dt
    # (this is the headline defect), and it stays in the consistent band on
    # the safe side. Validated: 2.88 / 1.73 / 1.02.
    for dt in _DTS:
        assert curve[dt][0] < 12.0, (
            f"attitude NEES {curve[dt][0]:.2f} at dt={dt}: over-confidence "
            f"not resolved (defect was ~55)."
        )
        assert curve[dt][0] > 0.2, \
            f"attitude NEES {curve[dt][0]:.2f} at dt={dt}: implausibly tiny"


def test_vanloan_flag_default_off_is_legacy():
    """Default (no flag) must NOT enable Van Loan (zero-regression contract)."""
    import inspect
    sig = inspect.signature(SRMOD.SRUAKF.__init__)
    assert sig.parameters["vanloan_Q"].default is False
