"""TRIAD, QUEST and the q-method: exactness, optimality, singularities and covariances."""

from __future__ import annotations

import numpy as np
import pytest

from ADCS.estimators.attitude_determination import (
    attitude_error,
    q_method,
    quaternion_from_rotation_matrix,
    quest,
    triad,
    triad_covariance,
    wahba_covariance,
    wahba_loss,
)
from ADCS.helpers.math_helpers import rot_mat


def _random_quaternion(rng) -> np.ndarray:
    q = rng.standard_normal(4)
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


def _random_directions(rng, count: int) -> np.ndarray:
    r = rng.standard_normal((count, 3))
    return r / np.linalg.norm(r, axis=1)[:, None]


def _observe(quaternion, references, sigma=0.0, rng=None) -> np.ndarray:
    """Body-frame observations with isotropic angular noise ``sigma`` (rad)."""
    body = references @ rot_mat(quaternion)  # rows: rot_mat(q).T @ r_i
    if rng is not None and np.any(np.asarray(sigma) > 0):
        body = body + np.asarray(sigma).reshape(-1, 1) * rng.standard_normal(body.shape)
        body = body / np.linalg.norm(body, axis=1)[:, None]
    return body


def _nees(errors: np.ndarray, covariance: np.ndarray) -> float:
    solved = np.linalg.solve(covariance, errors.T).T
    return float(np.mean(np.sum(errors * solved, axis=1)))


# --- exactness ---------------------------------------------------------------

@pytest.mark.parametrize("quaternion", [
    np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]),
    np.array([0.5, 0.5, 0.5, 0.5]), np.array([0.9, -0.2, 0.3, -0.1]) / np.linalg.norm([0.9, -0.2, 0.3, -0.1]),
])
def test_quaternion_from_rotation_matrix_round_trips_every_branch(quaternion):
    recovered = quaternion_from_rotation_matrix(rot_mat(quaternion))
    np.testing.assert_allclose(rot_mat(recovered), rot_mat(quaternion), atol=1e-14)
    assert recovered[0] >= 0.0


def test_triad_recovers_the_attitude_from_two_noise_free_directions():
    rng = np.random.default_rng(1)
    for _ in range(25):
        q = _random_quaternion(rng)
        r = _random_directions(rng, 2)
        b = _observe(q, r)
        solution = triad(b[0], b[1], r[0], r[1])
        assert np.linalg.norm(attitude_error(solution.quaternion, q)) < 1e-12
        assert solution.loss < 1e-24


@pytest.mark.parametrize("count", [2, 3, 6])
@pytest.mark.parametrize("solver", [q_method, quest])
def test_optimal_solvers_recover_the_attitude_from_noise_free_directions(count, solver):
    rng = np.random.default_rng(count)
    for _ in range(20):
        q = _random_quaternion(rng)
        r = _random_directions(rng, count)
        solution = solver(_observe(q, r), r, 1e-3)
        assert np.linalg.norm(attitude_error(solution.quaternion, q)) < 1e-9
        assert solution.loss < 1e-12


def test_quest_matches_the_q_method_on_noisy_data():
    rng = np.random.default_rng(7)
    for _ in range(30):
        q = _random_quaternion(rng)
        r = _random_directions(rng, 4)
        sigma = rng.uniform(1e-3, 3e-2, size=4)
        b = _observe(q, r, sigma, rng)
        a, c = quest(b, r, sigma), q_method(b, r, sigma)
        assert np.linalg.norm(attitude_error(a.quaternion, c.quaternion)) < 1e-9
        assert abs(a.loss - c.loss) < 1e-12


def test_quest_loss_is_never_worse_than_triad():
    rng = np.random.default_rng(11)
    for _ in range(30):
        q = _random_quaternion(rng)
        r = _random_directions(rng, 2)
        b = _observe(q, r, 2e-2, rng)
        optimum = quest(b, r, np.array([2e-2, 2e-2]))
        direct = triad(b[0], b[1], r[0], r[1], sigma_1=2e-2, sigma_2=2e-2)
        assert optimum.loss <= direct.loss + 1e-15
        assert wahba_loss(direct.quaternion, b, r, 2e-2) == pytest.approx(direct.loss)


# --- singularities and degeneracies -------------------------------------------

@pytest.mark.parametrize("solver", [q_method, quest])
def test_180_degree_rotations_are_handled(solver):
    rng = np.random.default_rng(3)
    cases = [np.array([0.0, 1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])]
    for _ in range(5):  # 179.99 degrees about random axes
        axis = _random_directions(rng, 1)[0]
        half = np.deg2rad(179.99) / 2.0
        cases.append(np.concatenate([[np.cos(half)], np.sin(half) * axis]))
    for q in cases:
        r = _random_directions(rng, 3)
        solution = solver(_observe(q, r), r, 1e-3)
        assert np.linalg.norm(attitude_error(solution.quaternion, q)) < 1e-8


def test_parallel_directions_are_rejected():
    r = np.array([[1.0, 0.0, 0.0], [1.0, 1e-12, 0.0]])
    with pytest.raises(ValueError, match="non-parallel"):
        triad(r[0], r[1], r[0], r[1])
    with pytest.raises(ValueError, match="not observable"):
        wahba_covariance(r, 1e-3)
    with pytest.raises(ValueError, match="not observable"):
        quest(r, r, 1e-3)


def test_inputs_are_validated():
    r = np.eye(3)
    with pytest.raises(ValueError, match="at least two"):
        quest(r[:1], r[:1], 1e-3)
    with pytest.raises(ValueError, match="positive"):
        quest(r, r, 0.0)
    with pytest.raises(ValueError, match="zero vectors"):
        quest(np.vstack([r[0], np.zeros(3)]), r[:2], 1e-3)
    with pytest.raises(ValueError, match="same shape"):
        quest(r, r[:2], 1e-3)


# --- covariances --------------------------------------------------------------

def test_wahba_covariance_is_consistent_with_the_scatter_of_quest_solutions():
    rng = np.random.default_rng(5)
    q = _random_quaternion(rng)
    r = _random_directions(rng, 3)
    sigma = np.array([2e-3, 4e-3, 3e-3])
    errors = np.array([attitude_error(quest(_observe(q, r, sigma, rng), r, sigma).quaternion, q) for _ in range(3000)])
    covariance = wahba_covariance(_observe(q, r), sigma)
    assert 2.75 < _nees(errors, covariance) < 3.25   # 3 degrees of freedom
    empirical = errors.T @ errors / errors.shape[0]
    assert np.linalg.norm(empirical - covariance) / np.linalg.norm(covariance) < 0.15


def test_triad_covariance_is_consistent_and_never_better_than_optimal():
    rng = np.random.default_rng(9)
    q = _random_quaternion(rng)
    r = _random_directions(rng, 2)
    sigma = np.array([1e-3, 5e-3])
    b = _observe(q, r)
    covariance = triad_covariance(b[0], b[1], r[0], r[1], sigma[0], sigma[1])
    errors = []
    for _ in range(3000):
        noisy = _observe(q, r, sigma, rng)
        errors.append(attitude_error(triad(noisy[0], noisy[1], r[0], r[1]).quaternion, q))
    errors = np.array(errors)
    assert 2.7 < _nees(errors, covariance) < 3.3
    excess = covariance - wahba_covariance(b, sigma)  # TRIAD ignores the first direction's noise structure
    assert np.linalg.eigvalsh(excess)[0] > -1e-12
    assert triad(b[0], b[1], r[0], r[1], sigma_1=sigma[0], sigma_2=sigma[1]).covariance is not None
    assert triad(b[0], b[1], r[0], r[1]).covariance is None


def test_the_more_precise_direction_dominates_the_optimal_solution():
    rng = np.random.default_rng(13)
    q = _random_quaternion(rng)
    r = _random_directions(rng, 2)
    b = _observe(q, r)
    corrupted = b.copy()
    corrupted[1] = _observe(q, r, np.array([0.0, 0.3]), rng)[1]  # second direction badly off
    solution = quest(corrupted, r, np.array([1e-6, 0.3]))
    predicted_first = rot_mat(solution.quaternion).T @ r[0]
    assert np.arccos(np.clip(predicted_first @ b[0], -1.0, 1.0)) < 1e-4  # the precise direction is honoured
