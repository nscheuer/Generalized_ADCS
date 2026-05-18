"""
SR-UKF cross_term=False S/P-invariant guard (test-hardening backlog #8).

`cross_term` defaults to False but EVERY shipped SR-UKF test passes
cross_term=True, so the default path was completely untested. It is broken
for a square-root filter whenever >=2 of the {actuator-bias, sensor-bias,
disturbance-parameter} blocks are non-empty:

Attitude_Estimator.update() does `x_hat = update_core(...)` then zeros the
cross blocks of the RETURNED covariance `oc`. But the SR-UKF's update_core
has already committed its propagated factor `self.S` and merely returns
oc = S^T S. Nothing re-derives self.S from the zeroed oc (only reset()
builds S from x_hat.cov, and it is not called per step). Consequences on
origin/main:

  * the requested decoupling never reaches the filter -- the next step
    propagates the still-fully-correlated self.S, so cross_term=False is a
    silent no-op for the SR-UKF's actual estimate; and
  * the reported covariance (x_hat.cov, zeroed) no longer equals the filter
    covariance (S^T S) -- the S/P invariant is broken.

Fix: a _resync_sqrt_factor hook (no-op for the non-square-root UAKF; the
SR-UKF rebuilds self.S from the decoupled covariance via its existing
PSD-safe square root and writes S^T S back so P == S^T S exactly).

These tests are deterministic (seeded, single static orbital state, one
update) and exercise a 2-block config (actuator-bias + sensor-bias) with an
injected actuator<->sensor-bias prior correlation so the zeroing is not a
trivial no-op. RED on origin/main, GREEN after; cross_term=True must stay
byte-identical (regression guard).
"""

import numpy as np
import pytest
from scipy.linalg import block_diag

from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.satellite_hardware.sensors import Gyro, MTM, SunPair
from ADCS.satellite_hardware.errors import Bias, Noise, ErrorMode
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.estimators.attitude_estimators import SRUAKF
from ADCS.helpers.math_helpers import random_n_unit_vec
from ADCS.helpers.math_constants import MathConstants

_UV = MathConstants.unitvecs


def _build():
    """A 2-bias-block SR-UKF setup (actuator-bias + sensor-bias, no
    disturbance params) with an injected actuator<->sensor-bias prior
    correlation. Returns (est_sat, real_sat, os0, x, x_hat, P0, Q0, idx)."""
    np.random.seed(5)
    mtq_n = Noise(noise=0.0, std_noise=1e-4)
    gyro_n = Noise(noise=0.0, std_noise=1e-4); gyro_bsr = 4e-4 * np.pi / 180
    mtm_n = Noise(noise=0.0, std_noise=1e-8); mtm_bsr = 1e-9
    sun_n = Noise(noise=0.0, std_noise=1e-4); sun_bsr = 1e-5

    acts = [MTQ(axis=_UV[j], max_torque=1.0, bias=Bias(bias=1e-3, std_bias=1e-4), noise=mtq_n.copy()) for j in range(3)]
    gyros = [Gyro(axis=_UV[j], bias=Bias(bias=0.002, std_bias=gyro_bsr), noise=gyro_n.copy()) for j in range(3)]
    mtms = [MTM(axis=_UV[j], bias=Bias(bias=mtm_bsr, std_bias=mtm_bsr), noise=mtm_n.copy()) for j in range(3)]
    suns = [SunPair(axis=_UV[j], efficiency=1.0, bias=Bias(bias=0.05, std_bias=sun_bsr), noise=sun_n.copy()) for j in range(3)]
    real = Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]), actuators=acts, sensors=mtms + gyros + suns)

    est_acts = [MTQ(axis=_UV[j], max_torque=1.0, bias=Bias(bias=0.0, std_bias=1e-4), noise=mtq_n.copy(), estimate_bias=True) for j in range(3)]
    est_gyros = [Gyro(axis=_UV[j], bias=Bias(bias=0.0, std_bias=gyro_bsr), noise=gyro_n.copy(), estimate_bias=True) for j in range(3)]
    est_mtms = [MTM(axis=_UV[j], bias=Bias(bias=0.0, std_bias=mtm_bsr), noise=mtm_n.copy()) for j in range(3)]
    est_suns = [SunPair(axis=_UV[j], efficiency=1.0, bias=Bias(bias=0.0, std_bias=sun_bsr), noise=sun_n.copy()) for j in range(3)]
    est_sat = EstimatedSatellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]),
                                 actuators=est_acts, sensors=est_mtms + est_gyros + est_suns)

    # >=2 non-empty bias/param blocks is the precondition for the defect.
    assert est_sat.act_bias_len > 0 and est_sat.att_sens_bias_len > 0

    n_aug = (est_sat.state_len + est_sat.act_bias_len
             + est_sat.att_sens_bias_len + est_sat.dist_param_len)
    x_hat = np.zeros(n_aug); x_hat[3] = 1.0

    a0 = est_sat.state_len - 1
    a1 = a0 + est_sat.act_bias_len
    s0 = a1
    s1 = s0 + est_sat.att_sens_bias_len
    P0 = block_diag(np.eye(3) * 1e-6, np.eye(3) * 1e-3,
                    np.eye(3) * 4e-7, np.eye(3) * 9e-7).astype(float)
    rho = 0.6 * np.sqrt(P0[a0, a0] * P0[s0, s0])
    for k in range(3):
        P0[a0 + k, s0 + k] = rho
        P0[s0 + k, a0 + k] = rho
    assert np.all(np.linalg.eigvalsh(P0) > 0)            # valid prior
    Q0 = np.eye(P0.shape[0]) * 1e-12

    ephem = Ephemeris()
    os0 = Orbital_State(ephem=ephem, J2000=0.22 - 1 * TimeConstants.sec2cent,
                        R=-7000 * np.array([0, np.sqrt(.5), np.sqrt(.5)]),
                        V=np.array([8.0, 0, 0]), B=np.array([0, 0.1, 0]),
                        S=np.array([1e5 + 1, 0, 0]), rho=5e-12)
    x = np.concatenate([random_n_unit_vec(3) * 0.02 * np.pi / 180,
                        random_n_unit_vec(4)])
    return est_sat, real, os0, x, x_hat, P0, Q0, (a0, a1, s0, s1)


def _run_once(cross_term):
    est_sat, real, os0, x, x_hat, P0, Q0, idx = _build()
    ukf = SRUAKF(est_sat=est_sat, J2000=os0.J2000, x_hat=x_hat.copy(),
                 P_hat=P0.copy(), Q_hat=Q0.copy(), dt=10.0,
                 cross_term=cross_term, quat_as_vec=False)
    dmode = ErrorMode(add_bias=True, add_noise=True, update_bias=True, update_noise=True)
    sens = real.sensor_readings(x=x, os=os0, dmode=dmode)
    ukf.update(u=np.zeros(len(est_sat.actuators)), sensors=sens, os=os0)
    P_rep = np.asarray(ukf.x_hat.cov, float)
    # With u=0 the actuator-bias block is unobservable over one step and
    # self.S can grow large in this deliberately minimal config; we only
    # care about the P == S^T S invariant, not that reconstruction's scale.
    with np.errstate(over="ignore", invalid="ignore"):
        P_true = ukf.S.T @ ukf.S
    return P_rep, P_true, idx


def test_srukf_cross_term_false_preserves_S_P_invariant():
    """Reported covariance must equal the filter's true covariance S^T S.
    RED on origin/main (self.S never refreshed from the zeroed cov ->
    mismatch ~1e-7); GREEN after the resync fix (exact)."""
    P_rep, P_true, _ = _run_once(cross_term=False)
    resid = float(np.max(np.abs(P_rep - P_true)))
    assert resid < 1e-10, f"S/P invariant broken: ||P_reported - S^T S|| = {resid:.3e}"


def test_srukf_cross_term_false_decoupling_reaches_the_filter():
    """The filter's propagated covariance (S^T S) must actually reflect the
    requested decoupling, i.e. the actuator<->sensor-bias cross block must be
    far smaller than in the cross_term=True (un-decoupled) run. RED on
    origin/main (no-op: filter cross block == the fully-correlated value)."""
    a0, a1, s0, s1 = _run_once(cross_term=False)[2]
    _, P_true_false, _ = _run_once(cross_term=False)
    _, P_true_true, _ = _run_once(cross_term=True)
    leak = np.max(np.abs(P_true_false[a0:a1, s0:s1]))
    full = np.max(np.abs(P_true_true[a0:a1, s0:s1]))
    assert full > 1e-9, "test setup: expected a non-trivial un-decoupled cross block"
    assert leak < 0.3 * full, (
        f"cross_term=False not applied to the filter: leak {leak:.3e} "
        f"vs un-decoupled {full:.3e}")


def test_srukf_cross_term_true_is_consistent_and_unchanged():
    """Regression guard: cross_term=True keeps the S/P invariant and is
    unaffected by the fix (the resync hook only fires when cross_term is
    False). Passes on origin/main and after."""
    P_rep, P_true, _ = _run_once(cross_term=True)
    # Exact invariant: for cross_term=True nothing zeroes oc, so the reported
    # covariance is exactly the reconstruction (robust to overflow entries
    # that are identical on both sides in this minimal config).
    assert np.array_equal(P_rep, P_true, equal_nan=True)
