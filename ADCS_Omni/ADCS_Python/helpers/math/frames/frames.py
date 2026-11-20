from numba import njit

from ..rotations.quaternion.conversions import _quaternion_to_dcm


@njit(cache=True)
def inertial_to_body_matrix(q_b_from_i):
    return _quaternion_to_dcm(q_b_from_i)


@njit(cache=True)
def body_to_inertial_matrix(q_b_from_i):
    return _quaternion_to_dcm(q_b_from_i).T


@njit(cache=True)
def inertial_to_body(vector_i, q_b_from_i):
    return _quaternion_to_dcm(q_b_from_i) @ vector_i


@njit(cache=True)
def body_to_inertial(vector_b, q_b_from_i):
    return _quaternion_to_dcm(q_b_from_i).T @ vector_b


@njit(cache=True)
def transform_vector(vector, source_to_target_dcm):
    return source_to_target_dcm @ vector


@njit(cache=True)
def transform_covariance(covariance_source, source_to_target_dcm):
    return source_to_target_dcm @ covariance_source @ source_to_target_dcm.T
