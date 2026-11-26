"""Python binding behavior for native C++ and Rust AttitudeSequentialUKF."""

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


def test_ukf_numpy_roundtrip_and_hyperparameters(
    subsystem_backend: Any, subsystem_backend_name: str
) -> None:
    record: Any = (
        {"estimated_rate"} if subsystem_backend_name == "cpp" else ("estimated_rate",)
    )
    ukf = subsystem_backend.AttitudeSequentialUKF(
        [2.0, 0.0, 0.0, 0.0],
        (0.1, -0.2, 0.3),
        np.eye(6) * 0.25,
        1e-5,
        alpha=0.5,
        beta=2.0,
        kappa=0.0,
        record=record,
    )

    assert ukf.COVARIANCE_SIZE == 6
    assert ukf.alpha == pytest.approx(0.5)
    assert ukf.beta == pytest.approx(2.0)
    assert ukf.kappa == pytest.approx(0.0)
    np.testing.assert_allclose(ukf.ukf_params, [0.5, 2.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(ukf.quaternion, IDENTITY, atol=1e-15)
    np.testing.assert_allclose(ukf.rate, [0.1, -0.2, 0.3], atol=1e-15)
    np.testing.assert_allclose(ukf.covariance, np.eye(6) * 0.25, atol=1e-15)
    np.testing.assert_allclose(ukf.acceleration_psd, np.eye(3) * 1e-5, atol=1e-15)
    assert ukf.record == frozenset({"estimated_rate"})

    q, w, p, a = ukf.prepare()
    assert (q.shape, w.shape, p.shape, a.shape) == ((4,), (3,), (6, 6), (3, 3))
    q[0] = 99.0
    p[0, 0] = 99.0
    np.testing.assert_allclose(ukf.quaternion, IDENTITY, atol=1e-15)
    np.testing.assert_allclose(ukf.covariance, np.eye(6) * 0.25, atol=1e-15)


def test_ukf_kernel_bindings_roundtrip_numpy_arrays(subsystem_backend_name: str) -> None:
    est = _estimator_module(subsystem_backend_name)
    q = IDENTITY.copy()
    w = np.array([0.2, -0.1, 0.05])
    p = np.eye(6) * 0.1
    psd = np.eye(3) * 1e-5

    q1, w1, p1 = est.ukf_predict(q, w, p, psd, 0.5, 0.5, 2.0, 0.0)
    assert (q1.shape, w1.shape, p1.shape) == ((4,), (3,), (6, 6))
    np.testing.assert_allclose(np.linalg.norm(q1), 1.0, atol=1e-14)
    np.testing.assert_allclose(w1, w, atol=1e-14)
    assert np.linalg.eigvalsh(p1).min() >= -1e-12


def test_ukf_error_mapping_to_python_exceptions(subsystem_backend: Any) -> None:
    w = np.zeros(3)
    p = np.eye(6) * 0.1
    psd = np.eye(3) * 1e-5

    with pytest.raises((ValueError, TypeError)):
        subsystem_backend.AttitudeSequentialUKF(IDENTITY, w, np.eye(7), psd)
    with pytest.raises(ValueError):
        subsystem_backend.AttitudeSequentialUKF(IDENTITY, w, p, psd, alpha=0.0)
    with pytest.raises(ValueError):
        subsystem_backend.AttitudeSequentialUKF(IDENTITY, w, p, psd, kappa=-6.0)
