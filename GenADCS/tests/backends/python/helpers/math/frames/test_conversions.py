import numpy as np
import pytest

from GenADCS.backends.python.helpers.math.frames import (
    convert_vector,
    ecef_to_gcrf_matrix,
    ecef_to_geodetic,
    ecef_to_teme_matrix,
    frame_rotation,
    gcrf_position_to_geodetic,
    gcrf_to_ecef_matrix,
    gcrf_to_lvlh_matrix,
    gcrf_to_teme_matrix,
    teme_to_ecef_matrix,
    teme_to_gcrf_matrix,
)

JD = 2451545.0


@pytest.mark.parametrize("frame", ["GCRF", "ECEF", "TEME", "LVLH", "BODY"])
def test_frame_rotation_is_orthonormal_with_required_context(frame):
    context = {
        "jd_ut1": JD,
        "jd_tt": JD,
        "position_gcrf": np.array([7e6, 0, 0.0]),
        "velocity_gcrf": np.array([0, 7500, 0.0]),
        "q_body_from_gcrf": [1, 0, 0, 0],
    }
    matrix = frame_rotation("GCRF", frame, **context)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=2e-12)
    assert np.linalg.det(matrix) == pytest.approx(1.0, abs=2e-12)


@pytest.mark.parametrize(
    "frame,kwargs",
    [
        ("ECEF", {}),
        ("TEME", {"jd_ut1": JD}),
        ("LVLH", {}),
        ("BODY", {}),
    ],
)
def test_required_frame_context_errors(frame, kwargs):
    with pytest.raises(ValueError):
        frame_rotation("GCRF", frame, **kwargs)


def test_frame_names_are_case_insensitive_and_same_frame_is_identity():
    np.testing.assert_array_equal(frame_rotation("gcrf", "GCRF"), np.eye(3))
    np.testing.assert_array_equal(frame_rotation("TEME", "teme"), np.eye(3))


def test_unsupported_frame_names_raise_value_error():
    with pytest.raises(ValueError, match="unsupported source frame"):
        frame_rotation("BAD", "GCRF")
    with pytest.raises(ValueError, match="unsupported target frame"):
        frame_rotation("GCRF", "BAD")


def test_frame_rotations_compose_and_inverse():
    kw = {"jd_ut1": JD + 0.25, "jd_tt": JD + 0.25, "xp": 1e-6, "yp": -2e-6}
    r_ge = frame_rotation("GCRF", "ECEF", **kw)
    r_et = frame_rotation("ECEF", "TEME", **kw)
    r_gt = frame_rotation("GCRF", "TEME", **kw)
    np.testing.assert_allclose(r_et @ r_ge, r_gt, atol=2e-12)
    np.testing.assert_allclose(frame_rotation("ECEF", "GCRF", **kw), r_ge.T, atol=2e-12)


def test_vector_conversion_agrees_with_rotation_and_roundtrips():
    context = {"jd_ut1": JD, "jd_tt": JD}
    vector = np.array([1.0, -2, 3])
    converted = convert_vector(vector, "GCRF", "ECEF", **context)
    np.testing.assert_allclose(
        converted, frame_rotation("GCRF", "ECEF", **context) @ vector
    )
    np.testing.assert_allclose(
        convert_vector(converted, "ECEF", "GCRF", **context), vector, atol=2e-12
    )


@pytest.mark.parametrize("vector", ([1, 2], [1, np.nan, 3], [np.inf, 0, 0]))
def test_convert_vector_rejects_nonfinite_or_wrong_shape(vector):
    with pytest.raises(ValueError, match="finite 3-vector"):
        convert_vector(vector, "GCRF", "GCRF")


def test_lvlh_axes_and_degenerate_orbits():
    matrix = gcrf_to_lvlh_matrix(np.array([7e6, 0, 0.0]), np.array([0, 7500, 0.0]))
    np.testing.assert_allclose(matrix, np.eye(3), atol=1e-15)
    with pytest.raises(ValueError, match="nonzero position"):
        gcrf_to_lvlh_matrix(np.array([0.0, 0, 0]), np.array([0.0, 1, 0]))
    with pytest.raises(ValueError, match="angular momentum"):
        gcrf_to_lvlh_matrix(np.array([1.0, 0, 0]), np.array([2.0, 0, 0]))
    with pytest.raises(ValueError, match="3-vectors"):
        gcrf_to_lvlh_matrix(np.array([1.0, 2]), np.array([1.0, 2, 3]))


def test_named_matrix_helpers_are_transposes_and_compositions():
    r_ge = gcrf_to_ecef_matrix(JD, JD)
    r_eg = ecef_to_gcrf_matrix(JD, JD)
    r_gt = gcrf_to_teme_matrix(JD)
    r_tg = teme_to_gcrf_matrix(JD)
    r_te = teme_to_ecef_matrix(JD)
    r_et = ecef_to_teme_matrix(JD)
    np.testing.assert_allclose(r_eg, r_ge.T, atol=1e-14)
    np.testing.assert_allclose(r_tg, r_gt.T, atol=1e-14)
    np.testing.assert_allclose(r_et, r_te.T, atol=1e-14)
    np.testing.assert_allclose(r_te @ r_gt, r_ge, atol=2e-12)


@pytest.mark.parametrize(
    "position,expected",
    [
        ([6378137.0, 0.0, 0.0], (0.0, 0.0, 0.0)),
        ([0.0, 6378137.0, 0.0], (0.0, 90.0, 0.0)),
        ([0.0, 0.0, 6356752.314245], (90.0, 0.0, 0.0)),
        ([0.0, 0.0, -6356752.314245], (-90.0, 0.0, 0.0)),
    ],
)
def test_ecef_to_geodetic_reference_points(position, expected):
    actual = ecef_to_geodetic(position)
    np.testing.assert_allclose(actual, expected, atol=2e-6)


def test_ecef_to_geodetic_altitude_and_longitude_quadrants():
    lat, lon, alt = ecef_to_geodetic(np.array([6379137.0, 0.0, 0.0]))
    assert lat == pytest.approx(0, abs=1e-10)
    assert lon == pytest.approx(0, abs=1e-10)
    assert alt == pytest.approx(1.0, abs=1e-8)
    assert ecef_to_geodetic(np.array([-6378137.0, 0, 0]))[1] == pytest.approx(180.0)
    assert ecef_to_geodetic(np.array([0.0, -6378137, 0]))[1] == pytest.approx(-90.0)


def test_gcrf_position_to_geodetic_is_ecef_rotation_then_conversion():
    position = np.array([6378137.0, 0, 0])
    result = gcrf_position_to_geodetic(position, JD, JD)
    expected = ecef_to_geodetic(gcrf_to_ecef_matrix(JD, JD) @ position)
    np.testing.assert_allclose(result, expected, atol=1e-9)


def test_jit_kernels_handle_noncontiguous_vectors():
    r = np.array([7e6, 8.0, 0.0, 9.0, 0.0, 10.0])[::2]
    v = np.array([0.0, 8.0, 7500.0, 9.0, 0.0, 10.0])[::2]
    assert r.shape == (3,) and v.shape == (3,)
    matrix = gcrf_to_lvlh_matrix(r, v)
    assert matrix.shape == (3, 3)
    assert np.all(np.isfinite(matrix))
