import numpy as np
import pytest

from ADCS.helpers.math_helpers import (
    ddrotmatTvecdqdq,
    drotmatTvecdq,
    normalize,
    normed_vec_hess,
    normed_vec_jac,
    vec_norm_hess,
    vec_norm_jac,
)


EPS = 1e-6
RNG = np.random.default_rng(0)


def unit_quaternion() -> np.ndarray:
    quaternion = RNG.normal(size=4)
    quaternion = quaternion / np.linalg.norm(quaternion)
    return quaternion if quaternion[0] >= 0 else -quaternion


def rotation_hessian_fd(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    numeric = np.zeros((4, 4, 3))
    for index in range(4):
        delta = np.zeros(4)
        delta[index] = EPS
        plus = np.asarray(drotmatTvecdq(quaternion + delta, vector), dtype=float)
        minus = np.asarray(drotmatTvecdq(quaternion - delta, vector), dtype=float)
        numeric[index] = (plus - minus) / (2.0 * EPS)
    return numeric


@pytest.mark.parametrize("trial", range(15))
def test_ddrotmatTvecdqdq_has_expected_shape(trial):
    analytic = np.asarray(ddrotmatTvecdqdq(unit_quaternion(), RNG.normal(size=3)), dtype=float)
    assert analytic.shape == (4, 4, 3)


@pytest.mark.parametrize("trial", range(15))
def test_ddrotmatTvecdqdq_matches_finite_difference(trial):
    quaternion = unit_quaternion()
    vector = RNG.normal(size=3)
    analytic = np.asarray(ddrotmatTvecdqdq(quaternion, vector), dtype=float)
    numeric = rotation_hessian_fd(quaternion, vector)
    assert np.max(np.abs(analytic - numeric)) < 1e-5


@pytest.mark.parametrize("trial", range(15))
def test_ddrotmatTvecdqdq_is_symmetric_in_quaternion_indices(trial):
    analytic = np.asarray(ddrotmatTvecdqdq(unit_quaternion(), RNG.normal(size=3)), dtype=float)
    assert np.allclose(analytic, np.transpose(analytic, (1, 0, 2)), atol=1e-9)


@pytest.mark.parametrize("trial", range(15))
def test_normed_vec_hess_has_expected_shape(trial):
    vector = RNG.normal(size=3) * RNG.uniform(0.5, 3.0)
    hessian = np.asarray(normed_vec_hess(vector), dtype=float)
    assert hessian.shape == (3, 3, 3)


@pytest.mark.parametrize("trial", range(15))
def test_normed_vec_hess_is_symmetric_in_input_indices(trial):
    vector = RNG.normal(size=3) * RNG.uniform(0.5, 3.0)
    hessian = np.asarray(normed_vec_hess(vector), dtype=float)
    assert np.allclose(hessian, np.transpose(hessian, (1, 0, 2)), atol=1e-9)


@pytest.mark.parametrize("trial", range(15))
def test_normed_vec_hess_predicts_second_order_curvature(trial):
    vector = RNG.normal(size=3) * RNG.uniform(0.5, 3.0)
    jacobian = np.asarray(normed_vec_jac(vector), dtype=float)
    hessian = np.asarray(normed_vec_hess(vector), dtype=float)
    base = normalize(vector)
    step = 1e-3
    max_relative_error = 0.0

    for _ in range(8):
        delta = step * RNG.standard_normal(3)
        linear = jacobian.T @ delta
        quadratic = 0.5 * np.einsum("ijk,i,j->k", hessian, delta, delta)
        prediction = base + linear + quadratic
        truth = normalize(vector + delta)
        residual = np.linalg.norm(truth - prediction)
        max_relative_error = max(max_relative_error, residual / (np.linalg.norm(quadratic) + 1e-30))

    assert max_relative_error < 0.05


@pytest.mark.parametrize("trial", range(15))
def test_vec_norm_hess_has_expected_shape(trial):
    hessian = np.asarray(vec_norm_hess(RNG.normal(size=3) * RNG.uniform(0.3, 4.0)), dtype=float)
    assert hessian.shape == (3, 3)


@pytest.mark.parametrize("trial", range(15))
def test_vec_norm_hess_matches_closed_form(trial):
    vector = RNG.normal(size=3) * RNG.uniform(0.3, 4.0)
    hessian = np.asarray(vec_norm_hess(vector), dtype=float)
    norm = np.linalg.norm(vector)
    closed_form = np.eye(3) / norm - np.outer(vector, vector) / norm**3
    assert np.allclose(hessian, closed_form, atol=1e-9)


@pytest.mark.parametrize("trial", range(15))
def test_vec_norm_hess_matches_finite_difference(trial):
    vector = RNG.normal(size=3) * RNG.uniform(0.3, 4.0)
    analytic = np.asarray(vec_norm_hess(vector), dtype=float)
    numeric = np.zeros((3, 3))
    for index in range(3):
        delta = np.zeros(3)
        delta[index] = EPS
        plus = np.asarray(vec_norm_jac(vector + delta), dtype=float).reshape(3)
        minus = np.asarray(vec_norm_jac(vector - delta), dtype=float).reshape(3)
        numeric[:, index] = (plus - minus) / (2.0 * EPS)
    assert np.allclose(analytic, numeric, atol=1e-5)


@pytest.mark.parametrize("trial", range(15))
def test_vec_norm_hess_is_symmetric(trial):
    hessian = np.asarray(vec_norm_hess(RNG.normal(size=3) * RNG.uniform(0.3, 4.0)), dtype=float)
    assert np.allclose(hessian, hessian.T, atol=1e-12)
