"""
End-to-end integrated-GNC closed-loop test (test-hardening backlog #13).

Coverage gap this closes (established by evidence earlier in the hardening
effort): the attitude-estimator tests run the filter open-loop (u=0, no
controller); the controller tests run the controller with the TRUE state
fed in as x_hat (no estimator in the loop). NOTHING exercised
`ADCS/simulate.py::simulate()`, the production pipeline that actually wires
truth -> sensors -> ESTIMATOR -> CONTROLLER -> actuators -> truth. So the
estimator<->controller interaction (the separation principle: a controller
driven by the filter's noisy estimate, and a filter that must stay
consistent while its own estimate excites the plant) was completely
untested -- exactly the blind spot that lets bugs like the covariance-units
mismatch / cross_term no-op hide.

This runs the SAME detumble scenario the existing controller suite proves
(MTQ_w_RW, p=0/d=1/c=0, No_Goal, static orbit) but with a UAKF placed in
the loop via simulate(). Assertions are smoke + closed-loop consistency
against INDEPENDENT references (the physical true-state trajectory and
actuator limits), not code-vs-itself.
"""

import numpy as np
import pytest

from ADCS.CONOPS.goals import No_Goal
from ADCS.controller import MTQ_w_RW
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import MTM, Gyro
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.estimators.attitude_estimators import UAKF
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import random_n_unit_vec
from ADCS.simulate import simulate

pytestmark = pytest.mark.slow
_UV = MathConstants.unitvecs


@pytest.fixture(scope="module")
def gnc_run():
    np.random.seed(11)
    # True plant: 3 MTQ + 3 RW, MTM + Gyro (noise+bias so the filter has
    # real work), mirroring the proven controller-suite config.
    acts = [MTQ(axis=_UV[j], max_torque=0.1) for j in range(3)]
    acts += [RW(axis=_UV[j], max_torque=4.51, J=0.22, h=0.0, h_max=3.8)
             for j in range(3)]
    mtms = [MTM(axis=_UV[j], noise=Noise(noise=0.0, std_noise=1e-8),
                bias=Bias(bias=1e-9, std_bias=1e-9)) for j in range(3)]
    gyros = [Gyro(axis=_UV[j], noise=Noise(noise=0.0, std_noise=1e-4),
                  bias=Bias(bias=2e-3, std_bias=4e-4 * np.pi / 180))
             for j in range(3)]
    real_sat = Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]),
                         actuators=acts, sensors=mtms + gyros)
    est_sat = EstimatedSatellite.from_satellite(real_sat)

    ephem = Ephemeris()
    os0 = Orbital_State(ephem=ephem, J2000=0.22,
                        R=-7000.0 * np.array([0, np.sqrt(.5), np.sqrt(.5)]),
                        V=np.array([8.0, 0.0, 0.0]),
                        B=np.array([0.0, 0.1, 0.0]),
                        S=np.array([1e5 + 1, 0.0, 0.0]), rho=5e-12)

    SL = real_sat.state_len                       # [w(3), q(4), h(3)] = 10
    w0 = random_n_unit_vec(3) * np.random.uniform(1.0, 2.0) * np.pi / 180.0
    q0 = random_n_unit_vec(4)
    x0 = np.concatenate([w0, q0, np.zeros(SL - 7)])

    x_hat0 = np.concatenate([np.zeros(3), [1.0, 0, 0, 0], np.zeros(SL - 7)])
    red = SL - 1                                   # reduced cov dim
    P0 = np.diag(np.concatenate([[1e-3] * 3, [1e-2] * 3, [1e-4] * (red - 6)]))
    Q0 = np.eye(red) * 1e-8

    ukf = UAKF(est_sat=est_sat, J2000=os0.J2000, x_hat=x_hat0,
               P_hat=P0, Q_hat=Q0, dt=1.0, cross_term=True, quat_as_vec=False)
    ctrl = MTQ_w_RW(est_sat=est_sat, p_gain=0.0, d_gain=1.0, c_gain=0.0,
                    h_target=np.zeros(3))

    results = simulate(x=x0, satellite=real_sat, est_satellite=est_sat,
                       controller=ctrl, estimator=ukf, goal=No_Goal(),
                       os0=os0, dt=1.0, tf=100.0)
    return results, x0


def _run(gnc_run):
    results, x0 = gnc_run
    r = results[0]
    return (r, np.asarray(r.state_hist, float),
            np.asarray(r.est_state_hist, float),
            np.asarray(r.control_hist, float), x0)


def test_gnc_smoke_all_logged_quantities_finite_and_physical(gnc_run):
    """The integrated pipeline runs and every logged quantity is finite;
    quaternions stay unit; MTQ commands respect the hardware limit."""
    r, sh, esh, ch, _ = _run(gnc_run)
    assert sh.ndim == 2 and sh.shape[0] > 10
    assert np.all(np.isfinite(sh)) and np.all(np.isfinite(esh))
    assert np.all(np.isfinite(ch))
    qn = np.linalg.norm(sh[:, 3:7], axis=1)
    np.testing.assert_allclose(qn, 1.0, atol=1e-3)
    # First 3 actuators are the MTQs (max_torque 0.1); commands must respect it.
    assert np.all(np.abs(ch[:, 0:3]) <= 0.1 + 1e-6)
    cov = [np.asarray(P, float) for P in r.state_cov_hist if P is not None]
    assert cov, "estimator covariance was not logged"
    for P in cov:
        assert np.all(np.isfinite(P))
        assert np.min(np.linalg.eigvalsh(0.5 * (P + P.T))) > -1e-6  # ~PSD


def test_gnc_estimator_stays_consistent_with_estimate_in_the_loop(gnc_run):
    """Separation-principle path: the controller is driven ONLY by the
    UAKF estimate, yet the estimate must still track the true state. Mean
    second-half attitude error (independent ref = the true quaternion) is
    bounded well below a tumble."""
    _, sh, esh, _, _ = _run(gnc_run)
    n = sh.shape[0]
    s = n // 2
    qt = sh[s:, 3:7] / np.linalg.norm(sh[s:, 3:7], axis=1, keepdims=True)
    qe = esh[s:, 3:7] / np.linalg.norm(esh[s:, 3:7], axis=1, keepdims=True)
    # attitude error angle = 2*acos(|<qt,qe>|)
    dots = np.abs(np.sum(qt * qe, axis=1)).clip(0, 1)
    err_deg = np.degrees(2.0 * np.arccos(dots))
    mean_err = float(np.mean(err_deg))
    assert np.isfinite(mean_err)
    assert mean_err < 30.0, (
        f"estimator did not track truth in closed loop: mean attitude "
        f"error {mean_err:.1f} deg")


def test_gnc_closed_loop_detumbles_end_to_end(gnc_run):
    """With ONLY the estimate feeding the controller, the TRUE angular rate
    must still decrease (the detumble works end-to-end through the full GNC
    stack). Independent ref = the physical true-rate trajectory."""
    _, sh, _, _, x0 = _run(gnc_run)
    w_init = float(np.linalg.norm(x0[0:3]))
    w_final = float(np.linalg.norm(np.mean(sh[-5:, 0:3], axis=0)))
    assert w_final < w_init, (
        f"closed-loop detumble failed: |w| {w_init:.4e} -> {w_final:.4e} rad/s")
