"""Python binding behavior for native C++ and Rust AttitudeEKF."""

from importlib import import_module
from typing import Any

import numpy as np
import pytest

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def _estimator_module(backend_name: str) -> Any:
    if backend_name == "cpp":
        return import_module(
            "GenADCS.backends.cpp.hardware.ADCS_subsystem.estimator.attitude_estimator._attitude_estimator"
        )
    return import_module(
        f"GenADCS.backends.{backend_name}.hardware.ADCS_subsystem.estimator.attitude_estimator"
    )


def test_ekf_numpy_roundtrip_and_normalization(
    subsystem_backend: Any, subsystem_backend_name: str
) -> None:
    record: Any = {"estimated_rate"} if subsystem_backend_name == "cpp" else ("estimated_rate",)
    ekf = subsystem_backend.AttitudeEKF(
        [2.0, 0.0, 0.0, 0.0],
        (0.1, -0.2, 0.3),
        np.eye(7) * 0.25,
        1e-5,
        record=record,
    )

    assert isinstance(ekf.quaternion, np.ndarray)
    assert isinstance(ekf.rate, np.ndarray)
    assert isinstance(ekf.covariance, np.ndarray)
    assert isinstance(ekf.acceleration_psd, np.ndarray)
    assert ekf.quaternion.shape == (4,)
    assert ekf.rate.shape == (3,)
    assert ekf.covariance.shape == (7, 7)
    assert ekf.acceleration_psd.shape == (3, 3)
    np.testing.assert_allclose(ekf.quaternion, IDENTITY, atol=1e-15)
    np.testing.assert_allclose(ekf.rate, [0.1, -0.2, 0.3], atol=1e-15)
    np.testing.assert_allclose(ekf.covariance, np.eye(7) * 0.25, atol=1e-15)
    np.testing.assert_allclose(ekf.acceleration_psd, np.eye(3) * 1e-5, atol=1e-15)
    assert ekf.record == frozenset({"estimated_rate"})

    q, w, p, a = ekf.prepare()
    assert (q.shape, w.shape, p.shape, a.shape) == ((4,), (3,), (7, 7), (3, 3))
    q[0] = 99.0
    w[0] = 99.0
    p[0, 0] = 99.0
    a[0, 0] = 99.0
    np.testing.assert_allclose(ekf.quaternion, IDENTITY, atol=1e-15)
    np.testing.assert_allclose(ekf.rate, [0.1, -0.2, 0.3], atol=1e-15)
    np.testing.assert_allclose(ekf.covariance, np.eye(7) * 0.25, atol=1e-15)
    np.testing.assert_allclose(ekf.acceleration_psd, np.eye(3) * 1e-5, atol=1e-15)


def test_ekf_kernel_bindings_roundtrip_numpy_arrays(subsystem_backend_name: str) -> None:
    est = _estimator_module(subsystem_backend_name)
    q = IDENTITY.copy()
    w = np.array([0.2, -0.1, 0.05])
    p = np.eye(7) * 0.1
    psd = np.eye(3) * 1e-5

    q1, w1, p1 = est.ekf_predict(q, w, p, psd, 0.5)
    assert (q1.shape, w1.shape, p1.shape) == ((4,), (3,), (7, 7))
    np.testing.assert_allclose(np.linalg.norm(q1), 1.0, atol=1e-14)
    np.testing.assert_allclose(w1, w, atol=1e-14)

    h_gyro = np.zeros((3, 7))
    h_gyro[:, 4:] = np.eye(3)
    q2, w2, p2 = est.ekf_correct(q1, w1, p1, -w1, h_gyro, np.eye(3) * 1e-5)
    assert (q2.shape, w2.shape, p2.shape) == ((4,), (3,), (7, 7))
    assert np.linalg.norm(w2) < 1e-3

    q3, w3, p3 = est.ekf_correct_gyro(
        q1, w1, p1, np.zeros(3), np.eye(3), np.eye(3) * 1e-5
    )
    np.testing.assert_allclose(w3, w2, atol=1e-14)
    np.testing.assert_allclose(p3, p2, atol=1e-14)

    q4, w4, p4 = est.ekf_correct_star_tracker(
        q1, w1, p1, IDENTITY, IDENTITY, np.eye(3) * 1e-5
    )
    assert (q4.shape, w4.shape, p4.shape) == ((4,), (3,), (7, 7))
    assert np.linalg.norm(q4[1:]) < np.linalg.norm(q1[1:]) / 10.0


def test_ekf_error_mapping_to_python_exceptions(
    subsystem_backend: Any, subsystem_backend_name: str
) -> None:
    est = _estimator_module(subsystem_backend_name)
    w = np.zeros(3)
    p = np.eye(7) * 0.1
    psd = np.eye(3) * 1e-5

    with pytest.raises((ValueError, TypeError)):
        subsystem_backend.AttitudeEKF(IDENTITY, w, np.eye(6), psd)
    bad_p = np.eye(7) * 0.1
    bad_p[0, 0] = -1.0
    with pytest.raises(ValueError):
        subsystem_backend.AttitudeEKF(IDENTITY, w, bad_p, psd)
    with pytest.raises(ValueError):
        subsystem_backend.AttitudeEKF(np.zeros(4), w, p, psd)
    bad_record: Any = {"bad_field"} if subsystem_backend_name == "cpp" else ("bad_field",)
    with pytest.raises(ValueError):
        subsystem_backend.AttitudeEKF(IDENTITY, w, p, psd, record=bad_record)
    with pytest.raises(ValueError):
        est.ekf_predict(IDENTITY, w, p, psd, -0.1)
