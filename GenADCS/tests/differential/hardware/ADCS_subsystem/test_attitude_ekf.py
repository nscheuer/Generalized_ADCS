"""Cross-backend differential tests comparing Python, C++, and Rust AttitudeEKF."""

from importlib import import_module
from typing import Any, cast

import numpy as np

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


def _ekf_kernels(name: str) -> tuple[Any, Any, Any, Any]:
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


def test_ekf_constructor_prepare_matches_across_backends(adcs_backends: Any) -> None:
    q0 = np.array([1.5, 0.2, -0.1, 0.05])
    w0 = np.array([0.1, -0.2, 0.3])
    p0 = np.eye(7) * 0.1
    psd = np.diag([1e-5, 2e-5, 3e-5])

    prepared = {
        name: api.AttitudeEKF(q0, w0, p0, psd).prepare()
        for name, api in adcs_backends.items()
    }
    ref_q, ref_w, ref_p, ref_a = prepared["python"]
    for name in ("cpp", "rust"):
        q, w, p, a = prepared[name]
        np.testing.assert_allclose(q, ref_q, atol=1e-14, err_msg=name)
        np.testing.assert_allclose(w, ref_w, atol=1e-14, err_msg=name)
        np.testing.assert_allclose(p, ref_p, atol=1e-14, err_msg=name)
        np.testing.assert_allclose(a, ref_a, atol=1e-14, err_msg=name)


def test_ekf_predict_and_correct_match_step_by_step(adcs_backends: Any) -> None:
    q0 = np.array([0.9984379, 0.04997396, -0.02498698, 0.00999479])
    q0 = q0 / np.linalg.norm(q0)
    w0 = np.array([0.15, -0.08, 0.04])
    p0 = np.eye(7) * 0.05
    psd = np.diag([1e-5, 2e-5, 1.5e-5])

    states = {name: (q0.copy(), w0.copy(), p0.copy()) for name in adcs_backends}
    kernels = {name: _ekf_kernels(name) for name in adcs_backends}

    for step_idx in range(3):
        for name in adcs_backends:
            predict_fn = kernels[name][0]
            q, w, p = states[name]
            states[name] = predict_fn(q, w, p, psd, 0.25)

        ref_q, ref_w, ref_p = states["python"]
        for name in ("cpp", "rust"):
            q, w, p = states[name]
            np.testing.assert_allclose(q, ref_q, atol=1e-10, rtol=1e-6, err_msg=f"{name} step {step_idx}")
            np.testing.assert_allclose(w, ref_w, atol=1e-10, rtol=1e-6, err_msg=f"{name} step {step_idx}")
            np.testing.assert_allclose(p, ref_p, atol=1e-8, rtol=1e-6, err_msg=f"{name} step {step_idx}")

    h = np.zeros((3, 7))
    h[:, 4:] = np.eye(3)
    residual = np.array([-0.15, 0.08, -0.04])
    r = np.eye(3) * 1e-4
    corrected = {
        name: kernels[name][1](*states[name], residual, h, r)
        for name in adcs_backends
    }
    ref_q, ref_w, ref_p = corrected["python"]
    for name in ("cpp", "rust"):
        q, w, p = corrected[name]
        np.testing.assert_allclose(q, ref_q, atol=1e-10, rtol=1e-6, err_msg=name)
        np.testing.assert_allclose(w, ref_w, atol=1e-10, rtol=1e-6, err_msg=name)
        np.testing.assert_allclose(p, ref_p, atol=1e-8, rtol=1e-6, err_msg=name)


def test_ekf_step_trajectory_matches_across_backends(adcs_backends: Any) -> None:
    q0 = np.array([0.9984379, 0.04997396, -0.02498698, 0.00999479])
    q0 = q0 / np.linalg.norm(q0)
    w0 = np.array([0.3, -0.2, 0.1])
    p0 = np.eye(7) * 0.1
    psd = np.eye(3) * 1e-7

    kernels = {name: _ekf_kernels(name) for name in adcs_backends}
    sensors = {name: _build_sensors(kernels[name][3]) for name in adcs_backends}
    states = {name: (q0.copy(), w0.copy(), p0.copy()) for name in adcs_backends}

    trajectory_readings = [
        np.array([0.01, -0.01, 0.005, 1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.9999, 0.01, -0.01, 0.0]),
        np.full(7, np.nan),
    ]

    for tick, readings in enumerate(trajectory_readings):
        for name in adcs_backends:
            step_fn = kernels[name][2]
            q, w, p = states[name]
            states[name] = step_fn(q, w, p, psd, 0.1, readings, sensors[name])

        ref_q, ref_w, ref_p = states["python"]
        for name in ("cpp", "rust"):
            q, w, p = states[name]
            np.testing.assert_allclose(q, ref_q, atol=1e-8, rtol=1e-6, err_msg=f"{name} tick {tick}")
            np.testing.assert_allclose(w, ref_w, atol=1e-8, rtol=1e-6, err_msg=f"{name} tick {tick}")
            np.testing.assert_allclose(p, ref_p, atol=1e-8, rtol=1e-6, err_msg=f"{name} tick {tick}")
