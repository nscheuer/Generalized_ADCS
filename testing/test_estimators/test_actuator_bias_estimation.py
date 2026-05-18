"""
Actuator-bias estimation coverage guard (critique pass).

Audit finding: sensor-bias estimation is exercised by the bias-convergence
suite (test_estimator_*_bias*.py, gyro/MTM/sun with estimate_bias=True --
verified sound). Actuator-bias estimation uses the SAME real,
working interface (a default Bias object is always present; match_estimate
writes act.bias.bias / act.bias.std_bias by act.input_len) -- but NO test
in the suite ever sets an actuator's estimate_bias=True, so the
actuator-bias path was completely uncovered.

It currently works (verified: the estimate propagates through
reset()->match_estimate for both UAKF and SR-UKF). This is a PR #37-model
guard: GREEN on origin/main, locking the working behaviour so it cannot
silently rot the way disturbance-parameter estimation did (that path used
an unimplemented dist.main_param/std API and was dead-on-arrival).
"""

import numpy as np
import pytest

from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.satellite_hardware.sensors import Gyro
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.estimators.attitude_estimators import UAKF, SRUAKF
from ADCS.helpers.math_constants import MathConstants

_UV = MathConstants.unitvecs


def _est_sat():
    return EstimatedSatellite(
        mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]),
        actuators=[MTQ(axis=_UV[j], max_torque=0.1,
                       bias=Bias(bias=0.0, std_bias=1e-4),
                       estimate_bias=True) for j in range(3)],
        sensors=[Gyro(axis=_UV[j], bias=Bias(bias=0.0, std_bias=1e-4),
                      noise=Noise(noise=0.0, std_noise=1e-4))
                 for j in range(3)])


def test_default_bias_present_when_estimate_bias_without_explicit_bias():
    """estimate_bias=True with no explicit Bias must still yield a usable
    Bias object (the match_estimate write path must not hit a None)."""
    m = MTQ(axis=_UV[0], max_torque=0.1, estimate_bias=True)
    assert m.estimate_bias is True
    assert m.bias is not None
    assert int(m.input_len) >= 1
    # match_estimate-style write must not raise
    m.bias.bias = np.array([0.01] * m.input_len)
    m.bias.std_bias = np.eye(m.input_len) * 1e-4


@pytest.mark.parametrize("Filter", [UAKF, SRUAKF])
def test_actuator_bias_estimate_propagates_via_match_estimate(Filter):
    es = _est_sat()
    assert es.act_bias_len == 3 and es.att_sens_bias_len == 0

    SL = es.state_len                       # [w,q,(h_rw)] -> 7 (MTQ-only)
    aug = SL + es.act_bias_len + es.att_sens_bias_len + es.dist_param_len
    red = (SL - 1) + es.act_bias_len + es.att_sens_bias_len + es.dist_param_len
    a0 = SL                                  # actuator-bias slot starts after state

    x_hat = np.zeros(aug)
    x_hat[3] = 1.0
    abias = np.array([2.0e-3, -1.0e-3, 3.0e-3])
    x_hat[a0:a0 + 3] = abias

    filt = Filter(est_sat=es, J2000=0.22, x_hat=x_hat,
                  P_hat=np.eye(red) * 1e-3, Q_hat=np.eye(red) * 1e-9,
                  dt=1.0, cross_term=True, quat_as_vec=False)
    assert filt is not None

    got = np.concatenate([np.atleast_1d(a.bias.bias) for a in es.actuators])
    np.testing.assert_allclose(
        got, abias, rtol=0, atol=0,
        err_msg="estimated actuator bias was not written into the "
                "actuator Bias models via match_estimate")
