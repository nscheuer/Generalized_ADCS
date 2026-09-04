"""Recursive behaviour of the EKF/MEKF and agreement with the legacy UAKF.

The unit tests in ``test_ekf_mekf.py`` exercise single ``predict``/``correct``
calls. These tests run the filters as a recursion, which is where a wrong
reset, a dropped process-noise term, or a mis-timed measurement shows up.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import block_diag

import ADCS
from ADCS.estimators.attitude_estimators import EKF, MEKF
from ADCS.estimators.old_attitude_estimators import UAKF
from ADCS.estimators.process_model import propagate_state
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import Gyro, StarTrackerQuaternion
from ADCS.state import EstimatorState, State


@pytest.fixture(scope="module")
def orbital_state() -> Orbital_State:
    return Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]),
    )


def _tracker_satellite() -> EstimatedSatellite:
    gyros = [Gyro(axis=axis, noise=Noise(noise=0.0, std_noise=1.0e-4)) for axis in np.eye(3)]
    tracker = StarTrackerQuaternion(min_stars=0, noise=Noise(noise=np.zeros(4), std_noise=np.full(4, 1.0e-3)))
    return EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=gyros + [tracker])


def _attitude_error_deg(a: State, b: State) -> float:
    return float(2.0 * np.degrees(np.arccos(min(1.0, abs(float(np.dot(a.q, b.q)))))))


@pytest.mark.parametrize("filter_type, full", [(EKF, True), (MEKF, False)])
def test_step_recursion_converges_and_keeps_covariance_valid(filter_type, full, orbital_state):
    """Thirty predict+correct cycles on noiseless measurements must converge,
    keep P symmetric and PSD, and (EKF) keep the quaternion block projected."""
    satellite = _tracker_satellite()
    stack = satellite.measurement_stack
    dt = 0.1
    truth = EstimatorState(w=[0.02, -0.015, 0.01], q=[0.9, 0.2, -0.3, 0.1]).normalized()
    n = truth.full_size if full else truth.tangent_size
    guess = EstimatorState(
        w=truth.w + [0.005, -0.004, 0.003],
        q=(truth.q + [0.0, 0.03, -0.02, 0.01]),
        cov=np.eye(n) * 1.0e-2,
        int_cov=np.zeros((n, n)),
    ).normalized()
    estimator = filter_type(satellite, guess, dt=dt, unmodeled_dynamics_psd=1.0e-9)
    control = np.empty(0)

    errors = [_attitude_error_deg(truth, estimator.state)]
    for _ in range(30):
        truth = propagate_state(truth, satellite, control, dt, orbital_state, orbital_state)
        measurements = stack.predict(truth, orbital_state)
        estimate = estimator.step(control, measurements, orbital_state, orbital_state)
        cov = estimate.covariance.as_matrix()

        assert np.isclose(np.linalg.norm(estimate.q), 1.0)
        assert np.allclose(cov, cov.T)
        assert np.linalg.eigvalsh(cov).min() > -1.0e-12
        if full:
            attitude = estimate.slice("attitude", coordinates="full")
            assert np.linalg.norm(cov[attitude, attitude] @ estimate.q) < 1.0e-10
        errors.append(_attitude_error_deg(truth, estimate))

    assert errors[-1] < 0.05 * errors[0]
    assert np.mean(errors[-5:]) < 0.02


@pytest.mark.parametrize("filter_type, full", [(EKF, True), (MEKF, False)])
def test_process_noise_actually_enters_the_prediction(filter_type, full, orbital_state):
    """P^- must exceed Phi P Phi^T when a non-zero PSD is configured."""
    satellite = _tracker_satellite()
    n = 7 if full else 6
    state = EstimatorState(w=[0.01, -0.02, 0.015], q=[1.0, 0.0, 0.0, 0.0], cov=np.eye(n) * 1.0e-2, int_cov=np.zeros((n, n)))
    traces = {}
    for psd in (0.0, 1.0e-6):
        estimator = filter_type(satellite, state, dt=1.0, unmodeled_dynamics_psd=psd)
        predicted = estimator.predict(np.empty(0), orbital_state, orbital_state)
        traces[psd] = np.trace(predicted.covariance.as_matrix())
    assert traces[1.0e-6] > traces[0.0] + 1.0e-7


def test_reset_replaces_the_estimate_and_clears_diagnostics(orbital_state):
    satellite = _tracker_satellite()
    state = EstimatorState(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0], cov=np.eye(6) * 1.0e-2, int_cov=np.zeros((6, 6)))
    estimator = MEKF(satellite, state, dt=0.5)
    estimator.predict(np.empty(0), orbital_state, orbital_state)
    assert estimator.diagnostics

    replacement = EstimatorState(w=[0.01, 0.0, 0.0], q=[0.0, 2.0, 0.0, 0.0], cov=np.eye(6) * 0.5, int_cov=np.zeros((6, 6)))
    out = estimator.reset(replacement)

    assert np.isclose(np.linalg.norm(out.q), 1.0)  # normalized on the way in
    assert np.allclose(out.w, [0.01, 0.0, 0.0])
    assert not estimator.diagnostics
    with pytest.raises(ValueError):
        estimator.reset(EstimatorState(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0], cov=np.eye(7) * 0.5, int_cov=np.zeros((7, 7))))


# --- legacy cross-check ----------------------------------------------------

class _StepAdapter:
    """Adapt ``predict``/``correct`` filters to the ``update(u, sensors, os)``
    protocol that :func:`ADCS.simulate` speaks."""

    def __init__(self, inner):
        self.inner = inner
        self._previous = None

    def update(self, u, sensors, os):
        if self._previous is None:
            self._previous = os
            return self.inner.correct(sensors, os)
        start, self._previous = self._previous, os
        return self.inner.step(u, sensors, start, os, midpoint_orbital_state=os)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _cross_check_scenario():
    mtm_noise = Noise(std_noise=5.0e-8)
    gyro_noise = Noise(std_noise=5.0e-7)
    sun_noise = Noise(std_noise=1.0e-3)

    def sensors():
        out = [ADCS.MTM(axis, noise=mtm_noise.copy()) for axis in np.eye(3)]
        out += [ADCS.Gyro(axis, noise=gyro_noise.copy()) for axis in np.eye(3)]
        out += [ADCS.SunPair(axis, efficiency=0.3, noise=sun_noise.copy()) for axis in np.eye(3)]
        return out

    satellite = ADCS.Satellite(mass=3000.0, J_0=np.diag([500.0, 1500.0, 1500.0]), sensors=sensors(), disturbances=[ADCS.disturbances.GG_Disturbance()])
    est_satellite = ADCS.EstimatedSatellite(mass=3200.0, J_0=np.diag([450.0, 1400.0, 1400.0]), sensors=sensors(), disturbances=[ADCS.disturbances.GG_Disturbance()])
    x_0 = State.from_array(np.array([0.001, 0.001, -0.002, 0.2588, 0.0, 0.9659, 0.0]))
    os0 = Orbital_State(ephem=Ephemeris(), J2000=0.22, R=np.array([5000.0, 0.0, 5000.0]), V=np.array([0.0, -7.5, 0.0]))
    return satellite, est_satellite, x_0, os0


def _attitude_error_history_deg(run) -> np.ndarray:
    truth, estimate = run.state_hist, run.est_state_hist
    return np.array([_attitude_error_deg(truth[i], estimate[i]) for i in range(len(run.time_s))])


def test_ekf_and_mekf_agree_with_legacy_uakf_given_equivalent_process_noise():
    """Same truth, same seed, same discrete process noise: all three filters
    must converge to a few degrees and agree with each other.

    The legacy UAKF takes a *discrete* per-step Q; the new filters take a
    *continuous* PSD, so ``psd * dt`` is matched to the UAKF's diagonal. The
    EKF keeps a four-component quaternion block, but in the quaternion_vector
    chart the vector part of the error quaternion *is* the tangent coordinate,
    so the per-component PSD carries over unscaled (the scalar component is
    projected out by the normalization sandwich).
    """
    dt, tf = 20.0, 2000.0
    P6 = block_diag(np.eye(3) * 0.01**2, np.eye(3))
    Q6 = block_diag(np.eye(3) * 1.0e-16, np.eye(3) * 1.0e-8)
    psd_tangent = np.concatenate((np.full(3, 1.0e-16 / dt), np.full(3, 1.0e-8 / dt)))
    psd_full = np.concatenate((np.full(3, 1.0e-16 / dt), np.full(4, 1.0e-8 / dt)))

    results = {}

    np.random.seed(0)
    satellite, est_satellite, x_0, os0 = _cross_check_scenario()
    ukf = UAKF(est_sat=est_satellite, J2000=0.22, x_hat=EstimatorState(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0]), P_hat=P6, Q_hat=Q6, dt=dt, cross_term=True, quat_as_vec=False)
    results["uakf"] = _attitude_error_history_deg(ADCS.simulate(x=x_0, satellite=satellite, est_satellite=est_satellite, estimator=ukf, os0=os0, dt=dt, tf=tf).first())

    np.random.seed(0)
    satellite, est_satellite, x_0, os0 = _cross_check_scenario()
    mekf_state = EstimatorState(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0], cov=P6, int_cov=np.zeros((6, 6)))
    results["mekf"] = _attitude_error_history_deg(ADCS.simulate(x=x_0, satellite=satellite, est_satellite=est_satellite, estimator=_StepAdapter(MEKF(est_satellite, mekf_state, dt=dt, unmodeled_dynamics_psd=psd_tangent)), os0=os0, dt=dt, tf=tf).first())

    np.random.seed(0)
    satellite, est_satellite, x_0, os0 = _cross_check_scenario()
    ekf_state = EstimatorState(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0], cov=np.diag([1.0e-4, 1.0e-4, 1.0e-4, 0.0, 1.0, 1.0, 1.0]), int_cov=np.zeros((7, 7)))
    results["ekf"] = _attitude_error_history_deg(ADCS.simulate(x=x_0, satellite=satellite, est_satellite=est_satellite, estimator=_StepAdapter(EKF(est_satellite, ekf_state, dt=dt, unmodeled_dynamics_psd=psd_full)), os0=os0, dt=dt, tf=tf).first())

    tails = {name: float(np.mean(err[-10:])) for name, err in results.items()}
    for name, tail in tails.items():
        assert tail < 5.0, f"{name} settled at {tail:.2f} deg"
    assert abs(tails["mekf"] - tails["uakf"]) < 1.5, tails
    assert abs(tails["ekf"] - tails["uakf"]) < 1.5, tails
    assert abs(tails["ekf"] - tails["mekf"]) < 1.0, tails
