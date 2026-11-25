"""Python calls, NumPy conversions, and exception mapping for frame bindings."""
from importlib import import_module
from typing import Any

import numpy as np
import pytest


@pytest.fixture(params=("cpp", "rust"), ids=("cpp", "rust"))
def frames(request: Any) -> Any:
    return import_module(f"GenADCS.backends.{request.param}.helpers.math.frames")


def context() -> Any:
    return {
        "jd_ut1": 2451545.0,
        "jd_tt": 2451545.0,
        "xp": np.float64(1e-6),
        "yp": np.float64(-2e-6),
    }


def test_frame_rotation_accepts_numpy_context_and_returns_matrix(frames: Any) -> Any:
    result = frames.frame_rotation("GCRF", "ECEF", **context())
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3) and result.dtype == np.float64
    np.testing.assert_allclose(result @ result.T, np.eye(3), atol=2e-12)


def test_convert_vector_numpy_input_and_inverse_roundtrip(frames: Any) -> Any:
    vector = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    converted = frames.convert_vector(vector, "GCRF", "ECEF", **context())
    assert converted.shape == (3,) and converted.dtype == np.float64
    restored = frames.convert_vector(converted, "ECEF", "GCRF", **context())
    np.testing.assert_allclose(restored, vector, atol=2e-6)


def test_optional_numpy_body_context_converts_through_native_api(frames: Any) -> Any:
    position = np.array([7e6, 0.0, 0.0], dtype=np.float32)
    velocity = np.array([0.0, 7500.0, 0.0], dtype=np.float32)
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    matrix = frames.frame_rotation(
        "GCRF", "LVLH", position_gcrf=position, velocity_gcrf=velocity
    )
    np.testing.assert_allclose(matrix, np.eye(3), atol=1e-14)
    body = frames.frame_rotation("GCRF", "BODY", q_body_from_gcrf=q)
    np.testing.assert_allclose(body, np.eye(3), atol=1e-14)


def test_bad_context_and_names_map_to_python_value_error(frames: Any) -> Any:
    with pytest.raises(ValueError):
        frames.frame_rotation("GCRF", "ECEF")
    with pytest.raises(ValueError):
        frames.frame_rotation("unknown", "GCRF")
    with pytest.raises(ValueError):
        frames.frame_rotation("GCRF", "LVLH")
    with pytest.raises(ValueError):
        frames.frame_rotation("GCRF", "BODY")


@pytest.mark.parametrize(
    "bad_vector", [np.array([1.0, 2.0]), np.array([np.inf, 0.0, 0.0])]
)
def test_bad_vector_maps_to_value_error(frames: Any, bad_vector: Any) -> Any:
    with pytest.raises(ValueError):
        frames.convert_vector(bad_vector, "GCRF", "GCRF")


@pytest.mark.parametrize(
    "position,expected",
    [
        (np.array([6378137.0, 0.0, 0.0]), [0.0, 0.0, 0.0]),
        (np.array([0.0, 0.0, 6356752.314245]), [90.0, 0.0, 0.0]),
        (np.array([0.0, 0.0, -6356752.314245]), [-90.0, 0.0, 0.0]),
    ],
)
def test_geodetic_numpy_conversion_and_output_shape(frames: Any, position: Any, expected: Any) -> Any:
    result = frames.ecef_to_geodetic(position)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,) and result.dtype == np.float64
    np.testing.assert_allclose(result, expected, atol=2e-6)
