"""Differential tests for ADCS goals across Python, C++, and Rust backends."""

from __future__ import annotations

from importlib import import_module
from typing import Any, NamedTuple

import numpy as np
import pytest

from GenADCS.backends.python.environment.ephemeris.de421 import DE421


class _GoalRuntime(NamedTuple):
    mu: float
    ephemeris_segments: np.ndarray
    ephemeris_times: np.ndarray
    ephemeris_data: np.ndarray


@pytest.fixture(scope="module")
def goal_backends() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("python", "cpp", "rust"):
        try:
            result[name] = import_module(
                f"GenADCS.backends.{name}.hardware.ADCS_subsystem.goals"
            )
        except ImportError as exc:
            pytest.fail(f"build {name} ADCS subsystem goals before differential tests: {exc}")
    return result


@pytest.fixture(scope="module")
def goal_runtime() -> _GoalRuntime:
    ephemeris = DE421()
    ephemeris.position(0, 2451545.0)
    return _GoalRuntime(
        mu=3.986004418e14,
        ephemeris_segments=np.ascontiguousarray(ephemeris._segments),
        ephemeris_times=np.ascontiguousarray(ephemeris._times),
        ephemeris_data=np.ascontiguousarray(ephemeris._data),
    )


@pytest.fixture(scope="module")
def orbital_state() -> np.ndarray:
    return np.array(
        [
            1.0, 0.0, 0.0, 0.0,
            0.01, -0.02, 0.03,
            6878137.0, 120000.0, -45000.0,
            -120.0, 7450.0, 1500.0,
        ],
        dtype=np.float64,
    )


def _assert_quat_close(q_a: np.ndarray, q_b: np.ndarray, atol: float = 1e-11) -> None:
    assert q_a.shape == (4,)
    assert q_b.shape == (4,)
    assert abs(np.linalg.norm(q_a) - 1.0) < 1e-12
    assert abs(np.linalg.norm(q_b) - 1.0) < 1e-12
    assert np.allclose(q_a, q_b, atol=atol) or np.allclose(q_a, -q_b, atol=atol)


def test_goal_exports_and_aliases(goal_backends: dict[str, Any]) -> None:
    expected = [
        ("NoGoal", "nogoal"),
        ("QuaternionGoal", "quaterniongoal"),
        ("VectorGoal", "vectorgoal"),
        ("TwoVectorGoal", "twovectorgoal"),
        ("AngularRateGoal", "angularrategoal"),
        ("NadirGoal", "nadirgoal"),
        ("ZenithGoal", "zenithgoal"),
        ("AlongTrackGoal", "alongtrackgoal"),
        ("CrossTrackGoal", "crosstrackgoal"),
        ("VelocityGoal", "velocitygoal"),
        ("CoordinateGoal", "coordinategoal"),
        ("SunGoal", "sungoal"),
        ("CoordinateFMCGoal", "coordinatefmcgoal"),
    ]
    for name, mod in goal_backends.items():
        for cls_name, alias in expected:
            assert hasattr(mod, cls_name), f"{name} missing {cls_name}"
            assert hasattr(mod, alias), f"{name} missing {alias}"
            assert getattr(mod, cls_name) is getattr(mod, alias)


def test_basic_goals_differential(
    goal_backends: dict[str, Any],
    goal_runtime: SimpleNamespace,
    orbital_state: np.ndarray,
) -> None:
    epoch_jd = 2460000.5
    py_mod = goal_backends["python"]

    for name, mod in goal_backends.items():
        if name == "python":
            continue

        # NoGoal
        q_ref, w_ref = py_mod.NoGoal().reference(orbital_state, epoch_jd, goal_runtime)
        q_cur, w_cur = mod.NoGoal().reference(orbital_state, epoch_jd, goal_runtime)
        assert q_ref is None and w_ref is None
        assert q_cur is None and w_cur is None

        # QuaternionGoal
        q_in = np.array([2.0, -1.0, 0.5, 0.25])
        w_in = np.array([0.01, -0.02, 0.03])
        q_ref, w_ref = py_mod.QuaternionGoal(q_in, w_in).reference(orbital_state, epoch_jd, goal_runtime)
        q_cur, w_cur = mod.QuaternionGoal(q_in, w_in)(orbital_state, epoch_jd, goal_runtime)
        _assert_quat_close(q_cur, q_ref)
        assert np.allclose(w_cur, w_ref, atol=1e-12)

        # AngularRateGoal
        q_ref, w_ref = py_mod.AngularRateGoal(w_in).reference(orbital_state, epoch_jd, goal_runtime)
        q_cur, w_cur = mod.AngularRateGoal(w_in).reference(orbital_state, epoch_jd, goal_runtime)
        assert q_ref is None and q_cur is None
        assert np.allclose(w_cur, w_ref, atol=1e-12)

        # VectorGoal
        q_ref, w_ref = py_mod.VectorGoal(
            (0.0, 1.0, 2.0), (0.0, 0.0, 1.0),
            inertial_roll=(1.0, 0.0, 0.0), body_roll=(0.0, 1.0, 0.0),
        ).reference(orbital_state, epoch_jd, goal_runtime)
        q_cur, w_cur = mod.VectorGoal(
            (0.0, 1.0, 2.0), (0.0, 0.0, 1.0),
            inertial_roll=(1.0, 0.0, 0.0), body_roll=(0.0, 1.0, 0.0),
        ).reference(orbital_state, epoch_jd, goal_runtime)
        _assert_quat_close(q_cur, q_ref)
        assert np.allclose(w_cur, w_ref, atol=1e-12)

        # TwoVectorGoal
        iv = [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        bv = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        q_ref, w_ref = py_mod.TwoVectorGoal(iv, bv).reference(orbital_state, epoch_jd, goal_runtime)
        q_cur, w_cur = mod.TwoVectorGoal(iv, bv).reference(orbital_state, epoch_jd, goal_runtime)
        _assert_quat_close(q_cur, q_ref)
        assert np.allclose(w_cur, w_ref, atol=1e-12)


@pytest.mark.parametrize(
    "goal_name",
    ["NadirGoal", "ZenithGoal", "AlongTrackGoal", "CrossTrackGoal", "VelocityGoal", "SunGoal"],
)
def test_orbit_goals_differential(
    goal_name: str,
    goal_backends: dict[str, Any],
    goal_runtime: SimpleNamespace,
    orbital_state: np.ndarray,
) -> None:
    epoch_jd = 2460000.5
    q_py, w_py = getattr(goal_backends["python"], goal_name)().reference(
        orbital_state, epoch_jd, goal_runtime
    )
    for name in ("cpp", "rust"):
        q_cur, w_cur = getattr(goal_backends[name], goal_name)().reference(
            orbital_state, epoch_jd, goal_runtime
        )
        _assert_quat_close(q_cur, q_py, atol=1e-10)
        assert np.allclose(w_cur, w_py, atol=1e-10)


def test_coordinate_goals_differential(
    goal_backends: dict[str, Any],
    goal_runtime: SimpleNamespace,
    orbital_state: np.ndarray,
) -> None:
    epoch_jd = 2460000.5
    py_mod = goal_backends["python"]

    q_coord_py, w_coord_py = py_mod.CoordinateGoal(47.3769, 8.5417, 0.4).reference(
        orbital_state, epoch_jd, goal_runtime
    )
    q_fmc_py, w_fmc_py = py_mod.CoordinateFMCGoal(47.3769, 8.5417, 0.4, 3).reference(
        orbital_state, epoch_jd, goal_runtime
    )

    for name in ("cpp", "rust"):
        mod = goal_backends[name]
        q_coord, w_coord = mod.CoordinateGoal(47.3769, 8.5417, 0.4).reference(
            orbital_state, epoch_jd, goal_runtime
        )
        _assert_quat_close(q_coord, q_coord_py, atol=1e-10)
        assert np.allclose(w_coord, w_coord_py, atol=1e-10)

        q_fmc, w_fmc = mod.CoordinateFMCGoal(47.3769, 8.5417, 0.4, 3).reference(
            orbital_state, epoch_jd, goal_runtime
        )
        _assert_quat_close(q_fmc, q_fmc_py, atol=1e-10)
        assert np.allclose(w_fmc, w_fmc_py, atol=1e-10)


def test_goal_validation_errors(goal_backends: dict[str, Any]) -> None:
    for name, mod in goal_backends.items():
        with pytest.raises((ValueError, TypeError)):
            mod.CoordinateFMCGoal(0.0, 0.0, 0.0, True)
        with pytest.raises(ValueError):
            mod.CoordinateFMCGoal(0.0, 0.0, 0.0, 11)
        with pytest.raises(ValueError):
            mod.CoordinateGoal(95.0, 0.0, 0.0)
        with pytest.raises(ValueError):
            mod.VectorGoal((1.0, 0.0, 0.0), inertial_roll=(2.0, 0.0, 0.0))
        with pytest.raises(ValueError):
            mod.QuaternionGoal((0.0, 0.0, 0.0, 0.0))
