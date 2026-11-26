"""Cross-backend conformance tests for AttitudeEKF."""

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


def _ekf_kernels(adcs_api: Any) -> tuple[Any, Any, Any, Any]:
    name = _backend_name(adcs_api)
    if name == "python":
        mod = import_module(
            "GenADCS.backends.python.hardware.ADCS_subsystem.estimator.attitude_estimator.attitude_ekf"
        )
        return mod.predict, mod.correct, mod.step, None
    if name == "cpp":
        mod = import_module(
            "GenADCS.backends.cpp.hardware.ADCS_subsystem.estimator.attitude_estimator._attitude_estimator"
        )
        return mod.ekf_predict, mod.ekf_correct, mod.ekf_step, mod.Sensor
    mod = import_module(
        "GenADCS.backends.rust.hardware.ADCS_subsystem.estimator.attitude_estimator"
    )
    return mod.ekf_predict, mod.ekf_correct, mod.ekf_step, None


def _build_sensors(sensor_cls: Any) -> Any:
    gyro = Gyroscope(
        np.zeros(3), IDENTITY, IDENTITY, 10.0, np.eye(3) * 1e-6, np.zeros(3)
    )
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


def test_ekf_constructor_and_prepare_conformance(adcs_api: Any) -> None:
    record: Any = {"estimated_rate"} if _backend_name(adcs_api) == "cpp" else ("estimated_rate",)
    ekf = adcs_api.AttitudeEKF(
        np.array([2.0, 0.0, 0.0, 0.0]),
        np.array([0.1, -0.2, 0.3]),
        np.eye(7) * 0.1,
        1e-5,
        record=record,
    )
    q, w, p, a = ekf.prepare()
    assert q.shape == (4,)
    assert w.shape == (3,)
    assert p.shape == (7, 7)
    assert a.shape == (3, 3)
    np.testing.assert_allclose(q, IDENTITY, atol=1e-14)
    np.testing.assert_allclose(w, [0.1, -0.2, 0.3], atol=1e-14)
    np.testing.assert_allclose(p, np.eye(7) * 0.1, atol=1e-14)
    np.testing.assert_allclose(a, np.eye(3) * 1e-5, atol=1e-14)


def test_ekf_invalid_inputs_raise_errors(adcs_api: Any) -> None:
    w = np.zeros(3)
    p = np.eye(7) * 0.1
    psd = np.eye(3) * 1e-5

    with pytest.raises((ValueError, TypeError)):
        adcs_api.AttitudeEKF(IDENTITY, w, np.eye(6), psd)
    bad_asym = np.eye(7) * 0.1
    bad_asym[0, 1] = 0.05
    with pytest.raises(ValueError):
        adcs_api.AttitudeEKF(IDENTITY, w, bad_asym, psd)
    bad_neg = np.eye(7) * 0.1
    bad_neg[0, 0] = -1.0
    with pytest.raises(ValueError):
        adcs_api.AttitudeEKF(IDENTITY, w, bad_neg, psd)
    with pytest.raises(ValueError):
        adcs_api.AttitudeEKF(np.zeros(4), w, p, psd)
    with pytest.raises(ValueError):
        adcs_api.AttitudeEKF(IDENTITY, np.array([0.0, np.nan, 0.0]), p, psd)
    with pytest.raises(ValueError):
        adcs_api.AttitudeEKF(IDENTITY, w, p, -1e-4)
    bad_record: Any = {"unknown_field"} if _backend_name(adcs_api) == "cpp" else ("unknown_field",)
    with pytest.raises(ValueError):
        adcs_api.AttitudeEKF(IDENTITY, w, p, psd, record=bad_record)


def test_ekf_predict_correct_and_step_conformance(adcs_api: Any) -> None:
    predict_fn, correct_fn, step_fn, sensor_cls = _ekf_kernels(adcs_api)
    w = np.array([0.2, -0.1, 0.05])
    p = np.eye(7) * 0.1
    psd = np.eye(3) * 2e-5

    q1, w1, p1 = predict_fn(IDENTITY, w, p, psd, 0.5)
    assert (q1.shape, w1.shape, p1.shape) == ((4,), (3,), (7, 7))
    np.testing.assert_allclose(np.linalg.norm(q1), 1.0, atol=1e-14)
    np.testing.assert_allclose(p1, p1.T, atol=1e-13)
    np.testing.assert_allclose(p1[:4, :4] @ q1, np.zeros(4), atol=1e-12)
    assert np.linalg.eigvalsh(p1).min() >= -1e-12

    with pytest.raises(ValueError):
        predict_fn(IDENTITY, w, p, psd, -0.1)

    h = np.zeros((3, 7))
    h[:, 4:] = np.eye(3)
    q2, w2, p2 = correct_fn(q1, w1, p1, -w1, h, np.eye(3) * 1e-5)
    assert np.linalg.norm(w2) < 1e-3
    np.testing.assert_allclose(p2, p2.T, atol=1e-13)
    np.testing.assert_allclose(p2[:4, :4] @ q2, np.zeros(4), atol=1e-12)
    assert np.linalg.eigvalsh(p2).min() >= -1e-12

    sensors = _build_sensors(sensor_cls)
    readings = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    q3, w3, p3 = step_fn(q1, w1, p1, psd, 0.1, readings, sensors)
    assert np.linalg.norm(w3) < 0.01
    assert np.linalg.norm(q3[1:]) < 0.01
    np.testing.assert_allclose(p3, p3.T, atol=1e-13)
    assert np.linalg.eigvalsh(p3).min() >= -1e-12

    q4, w4, _ = step_fn(q3, w3, p3, np.zeros((3, 3)), 0.1, np.full(7, np.nan), sensors)
    np.testing.assert_allclose(w4, w3, atol=1e-14)
    assert np.all(np.isfinite(q4))
