"""
Disturbance-parameter estimation restored (critique pass).

This worked in the original PhD/thesis code and rotted in the port:
EstimatedSatellite.match_estimate references a `dist.active` /
`dist.main_param` / `dist.std` interface that NO disturbance class
implemented. Constructing ANY attitude estimator over an EstimatedSatellite
that estimates a disturbance parameter (estimate_dist=True) raised
`AttributeError: '..._Disturbance' object has no attribute 'active'`
(then 'main_param'), so the whole feature was dead-on-arrival and there was
no test exercising it.

Restored: base Disturbance now provides `active` (default True) and `std`
(length estimated_vector_length) and a fail-loud `main_param` contract;
Dipole_Disturbance implements `main_param` bound to its residual dipole so
the estimator's write-back actually drives the disturbance torque.

RED on origin/main (AttributeError on estimator construction); GREEN after.
"""

import numpy as np
import pytest

from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.satellite_hardware.sensors import Gyro
from ADCS.satellite_hardware.disturbances import Dipole_Disturbance, GG_Disturbance
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.estimators.attitude_estimators import UAKF, SRUAKF
from ADCS.helpers.math_constants import MathConstants

_UV = MathConstants.unitvecs


def _est_sat():
    return EstimatedSatellite(
        mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]),
        actuators=[MTQ(axis=_UV[j], max_torque=0.1) for j in range(3)],
        sensors=[Gyro(axis=_UV[j], bias=Bias(bias=0.0, std_bias=1e-4),
                      noise=Noise(noise=0.0, std_noise=1e-4),
                      estimate_bias=True) for j in range(3)],
        disturbances=[Dipole_Disturbance(dipole_torque=np.zeros(3),
                                         estimate_dist=True)])


def test_dipole_main_param_drives_the_torque_model():
    """The estimated parameter must actually parameterise the disturbance:
    setting main_param changes the produced torque."""
    d = Dipole_Disturbance(dipole_torque=np.zeros(3), estimate_dist=True)
    assert d.main_param.size == d.estimated_vector_length == 3
    assert d.std.shape == (3,)

    class _OS:
        def get_state_vector(self, x):
            return {"b": np.array([1e-5, -2e-5, 3e-5])}

    os = _OS()
    x = np.concatenate([np.zeros(3), [1.0, 0, 0, 0]])
    t0 = np.asarray(d.torque(x, os), float)
    np.testing.assert_allclose(t0, 0.0, atol=0)        # zero dipole -> zero torque
    d.main_param = np.array([2.0e-4, -1.0e-4, 5.0e-5])
    t1 = np.asarray(d.torque(x, os), float)
    assert np.linalg.norm(t1) > 0.0                    # estimate now drives torque
    np.testing.assert_allclose(d.main_param, [2.0e-4, -1.0e-4, 5.0e-5])


@pytest.mark.parametrize("Filter", [UAKF, SRUAKF])
def test_estimating_dipole_param_constructs_and_propagates(Filter):
    es = _est_sat()
    assert es.dist_param_len == 3

    SL = es.state_len
    aug = SL + es.act_bias_len + es.att_sens_bias_len + es.dist_param_len
    red = (SL - 1) + es.act_bias_len + es.att_sens_bias_len + es.dist_param_len
    d0 = SL + es.act_bias_len + es.att_sens_bias_len

    x_hat = np.zeros(aug)
    x_hat[3] = 1.0
    dipole_est = np.array([1.0e-6, -2.0e-6, 3.0e-6])
    x_hat[d0:d0 + 3] = dipole_est

    # RED on origin/main: AttributeError inside reset()->match_estimate.
    filt = Filter(est_sat=es, J2000=0.22, x_hat=x_hat,
                  P_hat=np.eye(red) * 1e-3, Q_hat=np.eye(red) * 1e-9,
                  dt=1.0, cross_term=True, quat_as_vec=False)
    assert filt is not None
    np.testing.assert_allclose(
        np.asarray(es.disturbances[0].main_param, float).reshape(3),
        dipole_est, rtol=0, atol=0,
        err_msg="estimated dipole parameter was not propagated into the "
                "Dipole_Disturbance model via match_estimate")


def test_base_disturbance_main_param_is_fail_loud_not_silent():
    """An estimable disturbance that does NOT implement main_param must
    raise a clear NotImplementedError (not silently mis-estimate)."""
    gg = GG_Disturbance()
    # GG has estimated_vector_length 0 (not estimated) but still must expose
    # active/std without crashing, and the base main_param contract is loud.
    assert gg.active is True
    assert hasattr(gg, "std")
    with pytest.raises(NotImplementedError):
        _ = gg.main_param
