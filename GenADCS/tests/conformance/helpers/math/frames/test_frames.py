from typing import Any

import numpy as np
import pytest


def test_body_transform_and_covariance_contract(frames: Any) -> None:
    q = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    v = np.array([1.0, 0.0, 0.0])
    result = frames.inertial_to_body(v, q)
    np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=2e-14)
    np.testing.assert_allclose(frames.body_to_inertial(result, q), v, atol=2e-14)
    dcm = frames.inertial_to_body_matrix(q)
    np.testing.assert_allclose(dcm @ dcm.T, np.eye(3), atol=2e-14)
    cov = np.diag([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        frames.transform_covariance(cov, dcm), dcm @ cov @ dcm.T, atol=2e-14
    )


@pytest.mark.parametrize(
    "bad",
    [np.array([1.0, 2.0]), np.array([np.nan, 0.0, 0.0]), np.array([np.inf, 0.0, 0.0])],
)
def test_vector_argument_validation(frames: Any, bad: Any) -> None:
    with pytest.raises(ValueError):
        frames.convert_vector(bad, "GCRF", "GCRF")


def test_frame_rotation_context_and_inverse_contract(frames: Any) -> None:
    context = {"jd_ut1": 2451545.0, "jd_tt": 2451545.0, "xp": 1e-6, "yp": -2e-6}
    matrix = frames.frame_rotation("GCRF", "ECEF", **context)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=2e-12)
    np.testing.assert_allclose(
        frames.frame_rotation("ECEF", "GCRF", **context), matrix.T, atol=2e-12
    )
    np.testing.assert_array_equal(frames.frame_rotation("gcrf", "GCRF"), np.eye(3))
    with pytest.raises(ValueError):
        frames.frame_rotation("GCRF", "ECEF")
    with pytest.raises(ValueError):
        frames.frame_rotation("unsupported", "GCRF")


def test_lvlh_and_geodetic_public_contract(frames: Any) -> None:
    np.testing.assert_allclose(
        frames.gcrf_to_lvlh_matrix(
            np.array([7e6, 0.0, 0.0]), np.array([0.0, 7500.0, 0.0])
        ),
        np.eye(3),
        atol=1e-14,
    )
    for position, latitude in (
        (np.array([6378137.0, 0.0, 0.0]), 0.0),
        (np.array([0.0, 0.0, 6356752.314245]), 90.0),
        (np.array([0.0, 0.0, -6356752.314245]), -90.0),
    ):
        result = np.asarray(frames.ecef_to_geodetic(position))
        assert result.shape == (3,)
        assert result[0] == pytest.approx(latitude, abs=2e-6)
