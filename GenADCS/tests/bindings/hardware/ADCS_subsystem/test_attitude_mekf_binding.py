"""Python-facing binding tests for C++ and Rust AttitudeMEKF extensions."""

from importlib import import_module
from typing import Any

import numpy as np
import pytest


def _estimator_module(backend_name: str) -> Any:
    return import_module(
        f"GenADCS.backends.{backend_name}.hardware.ADCS_subsystem.estimator.attitude_estimator"
    )


def test_numpy_roundtrip_and_state_access(
    subsystem_backend: Any, subsystem_backend_name: str
) -> None:
    q_in = [2.0, 0.0, 0.0, 0.0]
    w_in = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    p_in = np.eye(6, dtype=np.float64) * 0.25
    record_arg: Any = (
        {"estimated_attitude", "estimated_rate"}
        if subsystem_backend_name == "cpp"
        else ("estimated_attitude", "estimated_rate")
    )
    est = subsystem_backend.AttitudeMEKF(
        q_in,
        w_in,
        p_in,
        1e-4,
        record=record_arg,
    )
    assert isinstance(est.quaternion, np.ndarray)
    assert isinstance(est.rate, np.ndarray)
    assert isinstance(est.covariance, np.ndarray)
    assert isinstance(est.acceleration_psd, np.ndarray)
    assert est.quaternion.shape == (4,)
    assert est.rate.shape == (3,)
    assert est.covariance.shape == (6, 6)
    assert est.acceleration_psd.shape == (3, 3)
    np.testing.assert_allclose(est.quaternion, [1.0, 0.0, 0.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(est.rate, [0.1, -0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(est.acceleration_psd, np.eye(3) * 1e-4, atol=1e-14)
    assert set(est.record) == {"estimated_attitude", "estimated_rate"}

    q, w, p, a = est.prepare()
    assert all(isinstance(arr, np.ndarray) for arr in (q, w, p, a))
    q[0] = 42.0
    p[0, 0] = 42.0
    np.testing.assert_allclose(est.quaternion, [1.0, 0.0, 0.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(est.covariance, p_in, atol=1e-14)

    mod = _estimator_module(subsystem_backend_name)
    q1, w1, p1 = mod.predict(q_in := np.array([1.0, 0.0, 0.0, 0.0]), [0.0, 0.0, 0.2], p_in, np.eye(3) * 1e-4, 0.5)
    assert isinstance(q1, np.ndarray) and q1.shape == (4,)
    assert isinstance(w1, np.ndarray) and w1.shape == (3,)
    assert isinstance(p1, np.ndarray) and p1.shape == (6, 6)

    q2, w2, p2 = mod.correct_gyro(
        q1, w1, p1, np.array([0.0, 0.0, 0.1]), np.eye(3), np.eye(3) * 1e-3
    )
    assert all(isinstance(arr, np.ndarray) for arr in (q2, w2, p2))
    assert q2.shape == (4,) and w2.shape == (3,) and p2.shape == (6, 6)

    q3, w3, p3 = mod.correct_star_tracker(
        q2, w2, p2, np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]), np.eye(3) * 1e-3
    )
    assert all(isinstance(arr, np.ndarray) for arr in (q3, w3, p3))
    assert q3.shape == (4,) and w3.shape == (3,) and p3.shape == (6, 6)


def test_step_binding_roundtrip(subsystem_backend_name: str) -> None:
    mod = _estimator_module(subsystem_backend_name)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    w0 = np.array([0.5, -0.3, 0.2])
    p0 = np.eye(6) * 0.1
    psd = np.eye(3) * 1e-6
    readings = np.zeros(3)

    if subsystem_backend_name == "cpp":
        sensor = mod.Sensor()
        sensor.code = 0
        sensor.noise_cholesky = np.eye(3) * 1e-3
        q1, w1, p1 = mod.step(q0, w0, p0, psd, 0.1, readings, [sensor])
    else:
        sensor_config = (
            np.array([0], dtype=np.int64),
            np.array([[1.0, 0.0, 0.0, 0.0]]),
            np.array([[1.0, 0.0, 0.0, 0.0]]),
            np.zeros((1, 6)),
            np.eye(6)[None, :, :] * 1e-3,
            np.array([[0.0, 0.0, 1.0]]),
            np.zeros((1, 2)),
            np.zeros((1, 3)),
            np.array([True]),
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([0], dtype=np.int64),
        )
        q1, w1, p1 = mod.step(q0, w0, p0, psd, 0.1, readings, sensor_config)

    assert isinstance(q1, np.ndarray) and q1.shape == (4,)
    assert isinstance(w1, np.ndarray) and w1.shape == (3,)
    assert isinstance(p1, np.ndarray) and p1.shape == (6, 6)
    assert np.linalg.norm(w1) < 0.05


def test_error_mapping_to_python_exceptions(
    subsystem_backend: Any, subsystem_backend_name: str
) -> None:
    mod = _estimator_module(subsystem_backend_name)
    ident_q = np.array([1.0, 0.0, 0.0, 0.0])
    zero_w = np.zeros(3)
    eye6 = np.eye(6)
    eye3 = np.eye(3)

    with pytest.raises(ValueError):
        subsystem_backend.AttitudeMEKF(np.zeros(4), zero_w, eye6, 1e-4)
    with pytest.raises(ValueError):
        subsystem_backend.AttitudeMEKF(ident_q, [np.nan, 0.0, 0.0], eye6, 1e-4)
    asym = eye6.copy()
    asym[0, 1] = 0.5
    with pytest.raises(ValueError):
        subsystem_backend.AttitudeMEKF(ident_q, zero_w, asym, 1e-4)
    with pytest.raises(ValueError):
        subsystem_backend.AttitudeMEKF(ident_q, zero_w, -eye6, 1e-4)
    invalid_record: Any = (
        {"invalid_field"} if subsystem_backend_name == "cpp" else ("invalid_field",)
    )
    with pytest.raises(ValueError):
        subsystem_backend.AttitudeMEKF(ident_q, zero_w, eye6, 1e-4, record=invalid_record)
    with pytest.raises((ValueError, TypeError)):
        subsystem_backend.AttitudeMEKF(np.ones(3), zero_w, eye6, 1e-4)
    with pytest.raises((ValueError, TypeError)):
        subsystem_backend.AttitudeMEKF(ident_q, zero_w, np.eye(7), 1e-4)

    with pytest.raises(ValueError):
        mod.predict(ident_q, zero_w, eye6, eye3, -0.1)

    h = np.zeros((3, 6))
    h[:, :3] = eye3
    with pytest.raises(ValueError):
        mod.correct(ident_q, zero_w, np.zeros((6, 6)), zero_w, h, np.zeros((3, 3)))
