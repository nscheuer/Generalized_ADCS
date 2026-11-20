"""C++ reference-frame functions matching ADCS_Python.helpers.math.frames."""

from ._frames import (
    body_to_inertial, body_to_inertial_matrix, convert_vector, ecef_to_geodetic,
    ecef_to_gcrf_matrix, ecef_to_teme_matrix, frame_rotation,
    gcrf_position_to_geodetic, gcrf_to_ecef_matrix, gcrf_to_lvlh_matrix,
    gcrf_to_teme_matrix, inertial_to_body, inertial_to_body_matrix,
    teme_to_ecef_matrix, teme_to_gcrf_matrix, transform_covariance,
    transform_vector,
)

__all__ = [
    "body_to_inertial", "body_to_inertial_matrix", "convert_vector",
    "ecef_to_geodetic", "ecef_to_gcrf_matrix", "ecef_to_teme_matrix",
    "frame_rotation", "gcrf_position_to_geodetic", "gcrf_to_ecef_matrix",
    "gcrf_to_lvlh_matrix", "gcrf_to_teme_matrix", "inertial_to_body",
    "inertial_to_body_matrix", "teme_to_ecef_matrix", "teme_to_gcrf_matrix",
    "transform_covariance", "transform_vector",
]
