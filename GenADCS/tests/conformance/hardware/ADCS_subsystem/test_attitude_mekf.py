"""Conformance tests for AttitudeMEKF across Python, C++, and Rust backends."""

from importlib import import_module
from typing import Any

import numpy as np
import pytest

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def _estimator_module(adcs_api: Any) -> Any:
    return import_module(f"{adcs_api.__name__}.estimator.attitude_estimator")


def test_constructor_and_prepare_contract(adcs_api: Any) -> None:
    record_arg: Any = (
        {"normalized_innovation_squared"}
        if ".cpp." in adcs_api.__name__
        else ("normalized_innovation_squared",)
    )
    est = adcs_api.AttitudeMEKF(
        np.array([3.0, 0.0, 0.0, 0.0]),
        np.array([0.05, -0.1, 0.2]),
        np.eye(6) * 0.5,
        1e-5,
        record=record_arg,
    )
    np.testing.assert_allclose(est.quaternion, IDENTITY, atol=1e-14)
    np.testing.assert_allclose(est.rate, [0.05, -0.1, 0.2], atol=1e-14)
    np.testing.assert_allclose(est.covariance, np.eye(6) * 0.5, atol=1e-14)
    np.testing.assert_allclose(est.acceleration_psd, np.eye(3) * 1e-5, atol=1e-14)
    assert "normalized_innovation_squared" in est.record

    q, w, p, a = est.prepare()
    q[0] = -10.0
    w[0] = -10.0
    p[0, 0] = -10.0
    a[0, 0] = -10.0
    np.testing.assert_allclose(est.quaternion, IDENTITY, atol=1e-14)
    np.testing.assert_allclose(est.rate, [0.05, -0.1, 0.2], atol=1e-14)
    np.testing.assert_allclose(est.covariance, np.eye(6) * 0.5, atol=1e-14)
    np.testing.assert_allclose(est.acceleration_psd, np.eye(3) * 1e-5, atol=1e-14)


def test_predict_and_covariance_properties(adcs_api: Any) -> None:
    mod = _estimator_module(adcs_api)
    psd = np.eye(3) * 2e-5
    q0, w0, p0 = mod.predict(IDENTITY, np.zeros(3), np.zeros((6, 6)), psd, 2.0)
    np.testing.assert_allclose(q0, IDENTITY, atol=1e-14)
    np.testing.assert_allclose(w0, np.zeros(3), atol=1e-14)
    np.testing.assert_allclose(p0[:3, :3], psd * 8.0 / 3.0, atol=1e-14)
    np.testing.assert_allclose(p0[3:, 3:], psd * 2.0, atol=1e-14)
    np.testing.assert_allclose(p0[:3, 3:], -psd * 2.0, atol=1e-14)
    np.testing.assert_allclose(p0, p0.T, atol=1e-14)
    assert np.linalg.eigvalsh(p0).min() >= -1e-12

    q_turn, w_turn, p_turn = mod.predict(
        IDENTITY, np.array([0.0, 0.0, 0.2]), np.eye(6) * 0.1, psd, 1.0
    )
    np.testing.assert_allclose(np.linalg.norm(q_turn), 1.0, atol=1e-14)
    np.testing.assert_allclose(q_turn[0], np.cos(0.1), atol=1e-12)
    np.testing.assert_allclose(q_turn[3], -np.sin(0.1), atol=1e-12)
    np.testing.assert_allclose(w_turn, [0.0, 0.0, 0.2], atol=1e-14)
    assert np.linalg.eigvalsh(p_turn).min() >= -1e-12


def test_correct_gyro_and_star_tracker_behavior(adcs_api: Any) -> None:
    mod = _estimator_module(adcs_api)
    q_init = np.array([np.cos(0.05), np.sin(0.05), 0.0, 0.0])
    p_init = np.eye(6) * 0.1

    q_st, _, p_st = mod.correct_star_tracker(
        q_init, np.zeros(3), p_init, IDENTITY, IDENTITY, np.eye(3) * 1e-5
    )
    assert np.linalg.norm(q_st[1:]) < np.linalg.norm(q_init[1:]) / 100.0
    assert np.trace(p_st[:3, :3]) < np.trace(p_init[:3, :3])
    np.testing.assert_allclose(p_st, p_st.T, atol=1e-14)
    assert np.linalg.eigvalsh(p_st).min() >= -1e-12

    mounting = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    q_g, w_g, p_g = mod.correct_gyro(
        IDENTITY, np.zeros(3), p_init, np.array([0.0, 0.5, 0.0]), mounting, np.eye(3) * 1e-5
    )
    np.testing.assert_allclose(q_g, IDENTITY, atol=1e-14)
    np.testing.assert_allclose(w_g, [0.5, 0.0, 0.0], atol=1e-3)
    assert np.trace(p_g[3:, 3:]) < np.trace(p_init[3:, 3:])
    np.testing.assert_allclose(p_g, p_g.T, atol=1e-14)


def test_invalid_inputs_rejected(adcs_api: Any) -> None:
    mod = _estimator_module(adcs_api)
    with pytest.raises(ValueError):
        adcs_api.AttitudeMEKF(np.zeros(4), np.zeros(3), np.eye(6), 1e-4)
    with pytest.raises(ValueError):
        adcs_api.AttitudeMEKF(IDENTITY, np.array([np.inf, 0.0, 0.0]), np.eye(6), 1e-4)
    asym = np.eye(6)
    asym[0, 1] = 0.25
    with pytest.raises(ValueError):
        adcs_api.AttitudeMEKF(IDENTITY, np.zeros(3), asym, 1e-4)
    with pytest.raises(ValueError):
        adcs_api.AttitudeMEKF(IDENTITY, np.zeros(3), -np.eye(6), 1e-4)
    bogus_record: Any = {"bogus"} if ".cpp." in adcs_api.__name__ else ("bogus",)
    with pytest.raises(ValueError):
        adcs_api.AttitudeMEKF(IDENTITY, np.zeros(3), np.eye(6), 1e-4, record=bogus_record)
    with pytest.raises(ValueError):
        mod.predict(IDENTITY, np.zeros(3), np.eye(6), np.eye(3) * 1e-4, -0.5)

    h = np.zeros((3, 6))
    h[:, :3] = np.eye(3)
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        mod.correct(IDENTITY, np.zeros(3), np.zeros((6, 6)), np.zeros(3), h, np.zeros((3, 3)))
