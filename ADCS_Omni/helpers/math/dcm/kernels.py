import numpy as np
from numba import njit


@njit(cache=True)
def axis_angle_to_dcm(axis, angle):
    axis_norm = np.sqrt(np.dot(axis, axis))
    if axis_norm == 0.0:
        raise ValueError("axis must be nonzero")
    x, y, z = axis / axis_norm
    cosine = np.cos(angle)
    sine = np.sin(angle)
    one_minus_cosine = 1.0 - cosine
    return np.array((
        (cosine + x * x * one_minus_cosine, x * y * one_minus_cosine - z * sine, x * z * one_minus_cosine + y * sine),
        (y * x * one_minus_cosine + z * sine, cosine + y * y * one_minus_cosine, y * z * one_minus_cosine - x * sine),
        (z * x * one_minus_cosine - y * sine, z * y * one_minus_cosine + x * sine, cosine + z * z * one_minus_cosine),
    ))


@njit(cache=True)
def rotate_vector(dcm_b_from_i, vector_i):
    return dcm_b_from_i @ vector_i


@njit(cache=True)
def inverse_dcm(dcm_b_from_i):
    return dcm_b_from_i.T


@njit(cache=True)
def compose_dcm(left, right):
    return left @ right


@njit(cache=True)
def relative_dcm(target, reference):
    return target @ reference.T


@njit(cache=True)
def dcm_orthogonality_error(matrix):
    residual = matrix.T @ matrix - np.eye(3)
    return np.sqrt(np.sum(residual * residual))


@njit(cache=True)
def normalize_dcm(matrix):
    first = np.array((matrix[0, 0], matrix[1, 0], matrix[2, 0]))
    first_norm = np.sqrt(np.dot(first, first))
    if first_norm == 0.0:
        raise ValueError("DCM has a degenerate first column")
    first = first / first_norm
    second_column = np.array((matrix[0, 1], matrix[1, 1], matrix[2, 1]))
    second = second_column - np.dot(first, second_column) * first
    second_norm = np.sqrt(np.dot(second, second))
    if second_norm == 0.0:
        raise ValueError("DCM has degenerate first and second columns")
    second = second / second_norm
    third = np.cross(first, second)
    return np.column_stack((first, second, third))
