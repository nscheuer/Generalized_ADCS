import numpy as np
import pytest

from ADCS.CONOPS.goals import No_Goal
from ADCS.controller import MTQ_w_RW
from ADCS.estimators.attitude_estimators import UAKF
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import random_n_unit_vec
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import Gyro, MTM
from ADCS.simulate import simulate


pytestmark = pytest.mark.slow
UNIT_VECTORS = MathConstants.unitvecs


@pytest.fixture(scope="module")
def gnc_run():
    np.random.seed(11)
    actuators = [MTQ(axis=UNIT_VECTORS[index], max_torque=0.1) for index in range(3)]
    actuators += [RW(axis=UNIT_VECTORS[index], max_torque=4.51, J=0.22, h=0.0, h_max=3.8) for index in range(3)]
    sensors = [
        *[
            MTM(
                axis=UNIT_VECTORS[index],
                noise=Noise(noise=0.0, std_noise=1e-8),
                bias=Bias(bias=1e-9, std_bias=1e-9),
            )
            for index in range(3)
        ],
        *[
            Gyro(
                axis=UNIT_VECTORS[index],
                noise=Noise(noise=0.0, std_noise=1e-4),
                bias=Bias(bias=2e-3, std_bias=4e-4 * np.pi / 180),
            )
            for index in range(3)
        ],
    ]
    real_satellite = Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]), actuators=actuators, sensors=sensors)
    estimated_satellite = EstimatedSatellite.from_satellite(real_satellite)

    orbital_state = Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=-7000.0 * np.array([0.0, np.sqrt(0.5), np.sqrt(0.5)]),
        V=np.array([8.0, 0.0, 0.0]),
        B=np.array([0.0, 0.1, 0.0]),
        S=np.array([1e5 + 1.0, 0.0, 0.0]),
        rho=5e-12,
    )

    state_length = real_satellite.state_len
    initial_rate = random_n_unit_vec(3) * np.random.uniform(1.0, 2.0) * np.pi / 180.0
    initial_quaternion = random_n_unit_vec(4)
    x0 = np.concatenate([initial_rate, initial_quaternion, np.zeros(state_length - 7)])

    x_hat0 = np.concatenate([np.zeros(3), [1.0, 0.0, 0.0, 0.0], np.zeros(state_length - 7)])
    reduced_length = state_length - 1
    covariance0 = np.diag(np.concatenate([[1e-3] * 3, [1e-2] * 3, [1e-4] * (reduced_length - 6)]))
    process_noise0 = np.eye(reduced_length) * 1e-8

    estimator = UAKF(
        est_sat=estimated_satellite,
        J2000=orbital_state.J2000,
        x_hat=x_hat0,
        P_hat=covariance0,
        Q_hat=process_noise0,
        dt=1.0,
        cross_term=True,
        quat_as_vec=False,
    )
    controller = MTQ_w_RW(
        est_sat=estimated_satellite,
        p_gain=0.0,
        d_gain=1.0,
        c_gain=0.0,
        h_target=np.zeros(3),
    )

    result = simulate(
        x=x0,
        satellite=real_satellite,
        est_satellite=estimated_satellite,
        controller=controller,
        estimator=estimator,
        goal=No_Goal(),
        os0=orbital_state,
        dt=1.0,
        tf=100.0,
    )[0]
    return result, x0


def state_histories(gnc_run):
    result, x0 = gnc_run
    return (
        result,
        np.asarray(result.state_hist, dtype=float),
        np.asarray(result.est_state_hist, dtype=float),
        np.asarray(result.control_hist, dtype=float),
        x0,
    )


def test_integrated_gnc_produces_finite_state_history(gnc_run):
    _, state_history, estimate_history, control_history, _ = state_histories(gnc_run)
    assert state_history.ndim == 2 and state_history.shape[0] > 10
    assert np.all(np.isfinite(state_history))
    assert np.all(np.isfinite(estimate_history))
    assert np.all(np.isfinite(control_history))


def test_integrated_gnc_keeps_true_quaternions_normalized(gnc_run):
    _, state_history, _, _, _ = state_histories(gnc_run)
    quaternion_norms = np.linalg.norm(state_history[:, 3:7], axis=1)
    np.testing.assert_allclose(quaternion_norms, 1.0, atol=1e-3)


def test_integrated_gnc_respects_mtq_command_limits(gnc_run):
    _, _, _, control_history, _ = state_histories(gnc_run)
    assert np.all(np.abs(control_history[:, 0:3]) <= 0.1 + 1e-6)


def test_integrated_gnc_logs_finite_positive_semidefinite_covariances(gnc_run):
    result, _, _, _, _ = state_histories(gnc_run)
    covariances = [np.asarray(covariance, dtype=float) for covariance in result.state_cov_hist if covariance is not None]
    assert covariances
    for covariance in covariances:
        assert np.all(np.isfinite(covariance))
        assert np.min(np.linalg.eigvalsh(0.5 * (covariance + covariance.T))) > -1e-6


def test_integrated_gnc_estimate_tracks_truth_in_second_half(gnc_run):
    _, state_history, estimate_history, _, _ = state_histories(gnc_run)
    start = state_history.shape[0] // 2
    q_true = state_history[start:, 3:7] / np.linalg.norm(state_history[start:, 3:7], axis=1, keepdims=True)
    q_est = estimate_history[start:, 3:7] / np.linalg.norm(estimate_history[start:, 3:7], axis=1, keepdims=True)
    dots = np.abs(np.sum(q_true * q_est, axis=1)).clip(0.0, 1.0)
    mean_error_deg = float(np.mean(np.degrees(2.0 * np.arccos(dots))))
    assert np.isfinite(mean_error_deg)
    assert mean_error_deg < 30.0


def test_integrated_gnc_detumbles_true_rate_end_to_end(gnc_run):
    _, state_history, _, _, x0 = state_histories(gnc_run)
    initial_rate_norm = float(np.linalg.norm(x0[0:3]))
    final_rate_norm = float(np.linalg.norm(np.mean(state_history[-5:, 0:3], axis=0)))
    assert final_rate_norm < initial_rate_norm
