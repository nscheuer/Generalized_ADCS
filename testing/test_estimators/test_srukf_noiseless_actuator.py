"""
SR-UKF must run with actuators that have no/zero noise (critique pass).

Every shipped SR-UKF test gives its actuators an explicit non-zero noise
(MTQ(noise=Noise(std_noise=1e-4))), so the extremely common case of
*noiseless* actuators (e.g. MTQ(axis=..., max_torque=...) with no noise
argument -- as in the tutorials, controller tests, and the integrated
sim) was never exercised by the SR-UKF.

It crashed. SRUAKF.make_pts_and_wts set
`L_q = control_cov.shape[0] if control_cov.size > 0 else 0`, so a
present-but-ALL-ZERO control covariance (3x3 of zeros for 3 noiseless
MTQs) inflated L / num_sigma. The control block then did
`cholesky(zeros)` -> LinAlgError -> `except: pass`, appending NO control
sigma points, so `len(pts) = 1 + 2*L_x` while `num_sigma = 2*(L_x+L_q)+1`
-> `pts[j]` IndexError in update_core. Even with L_q fixed, the
per-sigma-point control-noise vector was `np.zeros(0)`, so
`u (n_act,) + control_noise_j (0,)` raised a broadcast ValueError. The
plain UAKF was unaffected (its determine_covariances_to_use excludes an
all-zero control_cov, and its zeros_control is the actuator dimension).

Fix: gate L_q on the UAKF "used" predicate (non-empty AND not all-zero)
and size the control-noise vector by the actuator dimension regardless.

Reference: the plain UAKF -- whose bias-convergence tests validate it as
correct -- produces the answer the SR-UKF must match on this path. RED on
origin/main (IndexError), GREEN after; SR-UKF == UAKF state.
"""

import numpy as np
import pytest

from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.satellite_hardware.sensors import Gyro, MTM
from ADCS.satellite_hardware.errors import Bias, Noise, ErrorMode
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.estimators.attitude_estimators import UAKF, SRUAKF
from ADCS.helpers.math_constants import MathConstants

_UV = MathConstants.unitvecs


def _est_sat():
    # Noiseless MTQ actuators (no noise= argument) -> control_cov is a
    # present-but-all-zero matrix: the configuration that crashed the SR-UKF.
    return EstimatedSatellite(
        mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]),
        actuators=[MTQ(axis=_UV[j], max_torque=0.1) for j in range(3)],
        sensors=[MTM(axis=_UV[j], noise=Noise(noise=0.0, std_noise=1e-8))
                 for j in range(3)]
        + [Gyro(axis=_UV[j], bias=Bias(bias=0.0, std_bias=1e-12),
                noise=Noise(noise=0.0, std_noise=1e-4)) for j in range(3)])


def _measurement():
    ephem = Ephemeris()
    os0 = Orbital_State(ephem=ephem, J2000=0.22,
                        R=-7000.0 * np.array([0, np.sqrt(.5), np.sqrt(.5)]),
                        V=np.array([7.55, 0.0, 0.0]),
                        B=np.array([0.0, 0.1, 0.0]),
                        S=np.array([1e5 + 1, 0.0, 0.0]), rho=5e-12)
    real = Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]),
                     actuators=[MTQ(axis=_UV[j], max_torque=0.1) for j in range(3)],
                     sensors=[MTM(axis=_UV[j], noise=Noise(noise=0.0, std_noise=1e-8))
                              for j in range(3)]
                     + [Gyro(axis=_UV[j], noise=Noise(noise=0.0, std_noise=1e-4))
                        for j in range(3)])
    x = np.concatenate([[0.005, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    sens = real.sensor_readings(
        x=x, os=os0, dmode=ErrorMode(add_bias=True, add_noise=True,
                                     update_bias=True, update_noise=True))
    return sens, os0


def _run(Filter):
    es = _est_sat()
    SL = es.state_len
    x_hat = np.zeros(SL)
    x_hat[3] = 1.0
    f = Filter(est_sat=es, J2000=0.22, x_hat=x_hat,
               P_hat=np.diag([1e-6] * 3 + [1e-3] * 3),
               Q_hat=np.eye(SL - 1) * 1e-12, dt=10.0,
               cross_term=True, quat_as_vec=False)
    sens, os0 = _measurement()
    out = f.update(u=np.zeros(3), sensors=sens, os=os0)   # RED on main here
    return np.asarray(out, float)


def test_srukf_runs_with_noiseless_actuators():
    """SR-UKF update() must not crash when actuators have no/zero noise."""
    x = _run(SRUAKF)
    assert x.shape[0] >= 7 and np.all(np.isfinite(x))
    np.testing.assert_allclose(np.linalg.norm(x[3:7]), 1.0, atol=1e-6)


def test_srukf_matches_uakf_on_noiseless_actuator_path():
    """The (correct) plain UAKF is the reference: with identical
    configuration the SR-UKF must produce the same post-update state."""
    np.testing.assert_allclose(_run(SRUAKF), _run(UAKF), rtol=0, atol=1e-6)
