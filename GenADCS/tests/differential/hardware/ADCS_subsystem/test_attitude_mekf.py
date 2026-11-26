"""Differential tests comparing Python, C++, and Rust AttitudeMEKF outputs."""

from importlib import import_module
from typing import Any, cast

import numpy as np

from GenADCS.backends.python.environment import Environment
from GenADCS.backends.python.environment.ephemeris import EphemerisModel
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import (
    AutonomousStarTracker,
    CoarseSunSensor,
    FineSunSensor,
    Gyroscope,
    Magnetometer,
    Sensor,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors.sensor_sampling import (
    prepare_sensors,
)


def _estimator_modules(adcs_backends: dict[str, Any]) -> dict[str, Any]:
    return {
        name: import_module(f"{api.__name__}.estimator.attitude_estimator")
        for name, api in adcs_backends.items()
    }


def test_prepare_and_predict_trajectory_match(adcs_backends: dict[str, Any]) -> None:
    mods = _estimator_modules(adcs_backends)
    q0 = np.array([1.5, -0.3, 0.2, 0.1])
    w0 = np.array([0.12, -0.08, 0.25])
    p0 = np.diag([0.1, 0.12, 0.08, 0.05, 0.06, 0.04])
    psd = np.diag([1e-4, 2e-4, 3e-4])

    prepared = {
        name: api.AttitudeMEKF(q0, w0, p0, psd).prepare()
        for name, api in adcs_backends.items()
    }
    ref_q, ref_w, ref_p, ref_a = prepared["python"]
    for name in ("cpp", "rust"):
        q, w, p, a = prepared[name]
        np.testing.assert_allclose(q, ref_q, atol=1e-13, err_msg=name)
        np.testing.assert_allclose(w, ref_w, atol=1e-13, err_msg=name)
        np.testing.assert_allclose(p, ref_p, atol=1e-13, err_msg=name)
        np.testing.assert_allclose(a, ref_a, atol=1e-13, err_msg=name)

    states = {name: (ref_q.copy(), ref_w.copy(), ref_p.copy()) for name in mods}
    for _ in range(5):
        for name, mod in mods.items():
            q, w, p = states[name]
            states[name] = mod.predict(q, w, p, psd, 0.2)
        py_q, py_w, py_p = states["python"]
        for name in ("cpp", "rust"):
            q, w, p = states[name]
            np.testing.assert_allclose(q, py_q, atol=1e-12, err_msg=name)
            np.testing.assert_allclose(w, py_w, atol=1e-12, err_msg=name)
            np.testing.assert_allclose(p, py_p, atol=1e-12, err_msg=name)


def test_correct_gyro_and_star_tracker_match(adcs_backends: dict[str, Any]) -> None:
    mods = _estimator_modules(adcs_backends)
    q = np.array([0.99, 0.1, -0.05, 0.08])
    q /= np.linalg.norm(q)
    w = np.array([0.05, -0.02, 0.03])
    p = np.eye(6) * 0.2

    h = np.array(
        [
            [1.0, 0.1, 0.0, 0.0, 0.2, 0.0],
            [0.0, 1.0, -0.1, 0.0, 0.0, 0.2],
            [0.1, 0.0, 1.0, 0.2, 0.0, 0.0],
        ]
    )
    residual = np.array([0.03, -0.04, 0.02])
    r = np.diag([1e-3, 2e-3, 1.5e-3])
    direct = {name: mod.correct(q, w, p, residual, h, r) for name, mod in mods.items()}
    for name in ("cpp", "rust"):
        for idx in range(3):
            np.testing.assert_allclose(direct[name][idx], direct["python"][idx], atol=1e-12, err_msg=name)

    mounting = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    gyro = {
        name: mod.correct_gyro(q, w, p, np.array([0.1, 0.2, -0.05]), mounting, r)
        for name, mod in mods.items()
    }
    for name in ("cpp", "rust"):
        for idx in range(3):
            np.testing.assert_allclose(gyro[name][idx], gyro["python"][idx], atol=1e-12, err_msg=name)

    mount_q = np.array([np.cos(0.15), 0.0, np.sin(0.15), 0.0])
    meas_q = np.array([np.cos(0.12), np.sin(0.02), np.sin(0.12), -0.01])
    meas_q /= np.linalg.norm(meas_q)
    tracker = {
        name: mod.correct_star_tracker(q, w, p, meas_q, mount_q, r)
        for name, mod in mods.items()
    }
    for name in ("cpp", "rust"):
        for idx in range(3):
            np.testing.assert_allclose(tracker[name][idx], tracker["python"][idx], atol=1e-12, err_msg=name)


def test_multistep_gyro_and_star_tracker_match(adcs_backends: dict[str, Any]) -> None:
    mods = _estimator_modules(adcs_backends)
    ori_g = np.array([np.cos(0.1), np.sin(0.1), 0.0, 0.0])
    mis_g = np.array([np.cos(0.01), 0.0, np.sin(0.01), 0.0])
    bias_g = np.array([0.01, -0.02, 0.005])
    chol_g = np.diag([1e-3, 1.5e-3, 2e-3])

    ori_t = np.array([np.cos(0.2), 0.0, 0.0, np.sin(0.2)])
    mis_t = np.array([1.0, 0.0, 0.0, 0.0])
    chol_t = np.diag([2e-3, 2e-3, 3e-3])

    gyro = Gyroscope(np.zeros(3), ori_g, mis_g, 10.0, chol_g @ chol_g.T, bias_g)
    tracker = AutonomousStarTracker(np.zeros(3), ori_t, mis_t, 10.0, chol_t @ chol_t.T, np.zeros(3))
    py_cfg = prepare_sensors(
        cast(list[Sensor], [gyro, tracker]),
        Environment(ephemeris=cast(EphemerisModel, object())),
    )

    cpp_gyro = mods["cpp"].Sensor()
    cpp_gyro.code = 0
    cpp_gyro.orientation = ori_g
    cpp_gyro.misalignment = mis_g
    cpp_gyro.bias = bias_g
    cpp_gyro.noise_cholesky = chol_g
    cpp_gyro.offset = 0

    cpp_tracker = mods["cpp"].Sensor()
    cpp_tracker.code = 2
    cpp_tracker.orientation = ori_t
    cpp_tracker.misalignment = mis_t
    cpp_tracker.noise_cholesky = chol_t
    cpp_tracker.offset = 3
    cpp_sensors = [cpp_gyro, cpp_tracker]

    q0 = np.array([0.98, 0.1, -0.1, 0.12])
    q0 /= np.linalg.norm(q0)
    w0 = np.array([0.0, 0.0, 0.0])
    p0 = np.eye(6) * 0.2
    psd = np.eye(3) * 1e-5
    ticks = [
        np.array([0.05, -0.03, 0.02, 1.0, 0.0, 0.0, 0.0]),
        np.array([0.04, -0.02, 0.01, np.nan, np.nan, np.nan, np.nan]),
        np.array([0.03, -0.01, 0.02, np.cos(0.05), 0.0, np.sin(0.05), 0.0]),
    ]

    states = {name: (q0.copy(), w0.copy(), p0.copy()) for name in mods}
    for readings in ticks:
        for name, mod in mods.items():
            q, w, p = states[name]
            cfg = cpp_sensors if name == "cpp" else py_cfg
            states[name] = mod.step(q, w, p, psd, 0.1, readings, cfg)
        py_q, py_w, py_p = states["python"]
        for name in ("cpp", "rust"):
            q, w, p = states[name]
            np.testing.assert_allclose(q, py_q, atol=1e-12, err_msg=name)
            np.testing.assert_allclose(w, py_w, atol=1e-12, err_msg=name)
            np.testing.assert_allclose(p, py_p, atol=1e-12, err_msg=name)


def test_direction_sensor_step_matches_python_and_rust(adcs_backends: dict[str, Any]) -> None:
    mods = _estimator_modules(adcs_backends)
    mag = Magnetometer(
        np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]),
        10.0, np.eye(3) * 1e-5, np.zeros(3),
    )
    css = CoarseSunSensor(
        np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]), 1.0, 10.0, 1e-6, 0.0, 1.5,
    )
    fss = FineSunSensor(
        np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]),
        np.eye(2) * 1e-5, np.zeros(2), np.array([1.0, 1.0]),
    )
    cfg = prepare_sensors(
        cast(list[Sensor], [mag, css, fss]),
        Environment(
            magnetic_field=cast(Any, object()),
            ephemeris=cast(EphemerisModel, object()),
        ),
    )
    refs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    q0 = np.array([np.cos(0.05), np.sin(0.05), 0.0, 0.0])
    w0 = np.zeros(3)
    p0 = np.eye(6) * 0.1
    psd = np.eye(3) * 1e-6
    readings = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    py_res = mods["python"].step(q0, w0, p0, psd, 0.1, readings, cfg, refs)
    rs_res = mods["rust"].step(q0, w0, p0, psd, 0.1, readings, cfg, refs)
    for idx in range(3):
        np.testing.assert_allclose(rs_res[idx], py_res[idx], atol=1e-12)
