import numpy as np
from numba import njit

from ..rotations.quaternion.conversions import _quaternion_to_dcm


@njit(cache=True)
def inertial_to_body_matrix(q_b_from_i: np.ndarray) -> np.ndarray:
    return _quaternion_to_dcm(q_b_from_i)


@njit(cache=True)
def body_to_inertial_matrix(q_b_from_i: np.ndarray) -> np.ndarray:
    return _quaternion_to_dcm(q_b_from_i).T


@njit(cache=True)
def inertial_to_body(vector_i: np.ndarray, q_b_from_i: np.ndarray) -> np.ndarray:
    return _quaternion_to_dcm(q_b_from_i) @ vector_i


@njit(cache=True)
def body_to_inertial(vector_b: np.ndarray, q_b_from_i: np.ndarray) -> np.ndarray:
    return _quaternion_to_dcm(q_b_from_i).T @ vector_b


@njit(cache=True)
def transform_vector(vector: np.ndarray, source_to_target_dcm: np.ndarray) -> np.ndarray:
    return source_to_target_dcm @ vector


@njit(cache=True)
def transform_covariance(
    covariance_source: np.ndarray, source_to_target_dcm: np.ndarray
) -> np.ndarray:
    return source_to_target_dcm @ covariance_source @ source_to_target_dcm.T
