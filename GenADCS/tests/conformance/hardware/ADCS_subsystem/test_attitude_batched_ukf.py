"""Cross-backend conformance tests for AttitudeBatchedUKF."""

from importlib import import_module
from typing import Any, cast

import numpy as np
import pytest

from GenADCS.backends.python.environment import Environment
from GenADCS.backends.python.environment.ephemeris import EphemerisModel
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import (
    AutonomousStarTracker,
    Gyroscope,
    Sensor,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors.sensor_sampling import (
    prepare_sensors,
)

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def _backend_name(adcs_api: Any) -> str:
    return adcs_api.__name__.split(".")[2]


def _batched_ukf_kernels(adcs_api: Any) -> tuple[Any, Any, Any]:
    name = _backend_name(adcs_api)
    if name == "python":
        mod = import_module(
            "GenADCS.backends.python.hardware.ADCS_subsystem.estimator.attitude_estimator.attitude_batched_ukf"
        )
        return mod.predict, mod.step, None
    if name == "cpp":
        mod = import_module(
            "GenADCS.backends.cpp.hardware.ADCS_subsystem.estimator.attitude_estimator._attitude_estimator"
        )
        return mod.batched_ukf_predict, mod.batched_ukf_step, mod.Sensor
    mod = import_module(
        "GenADCS.backends.rust.hardware.ADCS_subsystem.estimator.attitude_estimator"
    )
    return mod.batched_ukf_predict, mod.batched_ukf_step, None


def _build_sensors(sensor_cls: Any) -> Any:
    gyro = Gyroscope(np.zeros(3), IDENTITY, IDENTITY, 10.0, np.eye(3) * 1e-6, np.zeros(3))
    tracker = AutonomousStarTracker(
        np.zeros(3), IDENTITY, IDENTITY, 10.0, np.eye(3) * 1e-6, np.zeros(3)
    )
    if sensor_cls is None:
        return prepare_sensors(
            cast(list[Sensor], [gyro, tracker]),
            Environment(ephemeris=cast(EphemerisModel, object())),
        )
    s_gyro = sensor_cls()
    s_gyro.code = 0
    s_gyro.offset = 0
    s_gyro.noise_cholesky = np.eye(3) * 1e-3
    s_tracker = sensor_cls()
    s_tracker.code = 2
    s_tracker.offset = 3
    s_tracker.noise_cholesky = np.eye(3) * 1e-3
    return [s_gyro, s_tracker]


def test_batched_ukf_constructor_and_prepare_conformance(adcs_api: Any) -> None:
    record: Any = (
        {"estimated_rate"} if _backend_name(adcs_api) == "cpp" else ("estimated_rate",)
    )
    ukf = adcs_api.AttitudeBatchedUKF(
        np.array([2.0, 0.0, 0.0, 0.0]),
        np.array([0.1, -0.2, 0.3]),
        np.eye(6) * 0.1,
        1e-5,
        alpha=0.5,
        beta=2.0,
        kappa=0.0,
        record=record,
    )
    q, w, p, a = ukf.prepare()
    assert (q.shape, w.shape, p.shape, a.shape) == ((4,), (3,), (6, 6), (3, 3))
    np.testing.assert_allclose(q, IDENTITY, atol=1e-14)
    np.testing.assert_allclose(w, [0.1, -0.2, 0.3], atol=1e-14)
    np.testing.assert_allclose(p, np.eye(6) * 0.1, atol=1e-14)
    np.testing.assert_allclose(a, np.eye(3) * 1e-5, atol=1e-14)
    np.testing.assert_allclose(ukf.ukf_params, [0.5, 2.0, 0.0], atol=1e-14)


def test_batched_ukf_invalid_inputs_raise_errors(adcs_api: Any) -> None:
    w = np.zeros(3)
    p = np.eye(6) * 0.1
    psd = np.eye(3) * 1e-5

    with pytest.raises((ValueError, TypeError)):
        adcs_api.AttitudeBatchedUKF(IDENTITY, w, np.eye(7), psd)
    bad_asym = np.eye(6) * 0.1
    bad_asym[0, 1] = 0.05
    with pytest.raises(ValueError):
        adcs_api.AttitudeBatchedUKF(IDENTITY, w, bad_asym, psd)
    bad_neg = np.eye(6) * 0.1
    bad_neg[0, 0] = -1.0
    with pytest.raises(ValueError):
        adcs_api.AttitudeBatchedUKF(IDENTITY, w, bad_neg, psd)


def test_batched_ukf_predict_and_step_conformance(adcs_api: Any) -> None:
    predict_fn, step_fn, sensor_cls = _batched_ukf_kernels(adcs_api)
    q0 = np.array([0.9984379, 0.04997396, -0.02498698, 0.00999479])
    q0 /= np.linalg.norm(q0)
    w0 = np.array([0.15, -0.08, 0.04])
    p0 = np.eye(6) * 0.05
    psd = np.diag([1e-5, 2e-5, 1.5e-5])

    q1, w1, p1 = predict_fn(q0, w0, p0, psd, 0.25)
    np.testing.assert_allclose(np.linalg.norm(q1), 1.0, atol=1e-14)
    np.testing.assert_allclose(w1, w0, atol=1e-14)
    np.testing.assert_allclose(p1, p1.T, atol=1e-13)
    assert np.linalg.eigvalsh(p1).min() >= -1e-12

    sensors = _build_sensors(sensor_cls)
    readings = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    q2, w2, p2 = step_fn(q1, w1, p1, psd, 0.1, readings, sensors)
    assert np.linalg.norm(w2) < np.linalg.norm(w1)
    assert np.linalg.norm(q2[1:]) < np.linalg.norm(q1[1:])
    assert np.linalg.eigvalsh(p2).min() >= -1e-12
