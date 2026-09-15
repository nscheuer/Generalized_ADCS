"""Proposed tests for the square-root transformed / updated_linear / updated_unscented paths.

Run against the patched module with:  pytest -p patched_plugin suggested_tests/
"""
import numpy as np
import pytest

from ADCS.covariance import Covariance
from ADCS.helpers.cholesky_update import cholupdate, choldowndate

SEED = 20260915


def _spd(n, rng, cond=None):
    with np.errstate(all="ignore"):
        Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        d = np.exp(rng.normal(scale=0.7, size=n)) + 0.3 if cond is None else np.logspace(0, -np.log10(cond), n)
        return Q @ np.diag(d) @ Q.T


def _ut_weights(n, alpha=1.0, beta=2.0, kappa=0.0):
    scale = alpha**2 * (n + kappa)
    mean = np.full(2 * n + 1, 0.5 / scale)
    cov = mean.copy()
    mean[0] = (scale - n) / scale
    cov[0] = mean[0] + 1.0 - alpha**2 + beta
    return np.sqrt(scale), mean, cov


class _Guard(np.ndarray):
    """Raises on any matmul between two factor-derived arrays: proves P is never formed."""

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc is np.matmul and sum(isinstance(a, _Guard) for a in inputs) >= 2:
            raise AssertionError("square-root path formed a product of two factor-derived arrays")
        inputs = tuple(np.asarray(a) if isinstance(a, _Guard) else a for a in inputs)
        result = getattr(ufunc, method)(*inputs, **kwargs)
        return result.view(_Guard) if isinstance(result, np.ndarray) else result


def _guarded(covariance):
    covariance._data = covariance._data.view(_Guard)
    return covariance


@pytest.mark.parametrize("n", [1, 3, 6, 9, 15])
@pytest.mark.parametrize("policy", ["strict", "allow_indefinite"])
def test_sqrt_transformed_matches_dense_and_never_forms_p(n, policy):
    rng = np.random.default_rng(SEED + n)
    P = _spd(n, rng)
    for rows in (n, max(n - 2, 0), n + 3):
        J = rng.normal(size=(rows, n))
        sqrt = _guarded(Covariance(P, form="sqrt", psd_policy=policy)).transformed(J, coordinates="x")
        dense = Covariance(P, form="full", psd_policy=policy).transformed(J, coordinates="x")
        assert sqrt.form == "sqrt" and sqrt.coordinates == "x" and sqrt.psd_policy == policy
        assert sqrt.shape == (rows, rows)
        np.testing.assert_allclose(sqrt.as_matrix(), dense.as_matrix(), rtol=1e-12, atol=1e-13 * max(1.0, np.abs(P).max()))
        factor = sqrt.upper_factor()
        assert np.allclose(factor, np.triu(factor)) and np.all(np.diag(factor) >= 0.0)


@pytest.mark.parametrize("n", [1, 3, 6, 9, 15])
@pytest.mark.parametrize("joseph", [True, False])
def test_sqrt_updated_linear_matches_dense_joseph(n, joseph):
    rng = np.random.default_rng(SEED + 100 + n)
    P = _spd(n, rng, cond=1e6)
    for m in sorted({1, max(1, n // 2), n, n + 2}):
        H = rng.normal(size=(m, n))
        R = Covariance(_spd(m, rng) * 0.3)
        gain_s, post_s = _guarded(Covariance(P, form="sqrt")).updated_linear(H, R, joseph=joseph)
        gain_d, post_d = Covariance(P, form="full").updated_linear(H, R, joseph=True)
        assert post_s.form == "sqrt"
        np.testing.assert_allclose(gain_s, gain_d, rtol=1e-10, atol=1e-13)
        np.testing.assert_allclose(post_s.as_matrix(), post_d.as_matrix(), rtol=1e-10, atol=1e-13)
        assert np.linalg.eigvalsh(post_s.as_matrix()).min() >= -1e-14


@pytest.mark.parametrize("n", [3, 6, 9])
@pytest.mark.parametrize("alpha_kappa", [(1.0, 0.0), (0.5, None)])
def test_sqrt_updated_unscented_matches_dense(n, alpha_kappa):
    rng = np.random.default_rng(SEED + 200 + n)
    alpha, kappa = alpha_kappa
    kappa = 3.0 - n if kappa is None else kappa  # negative W0 for alpha=0.5
    P = _spd(n, rng)
    gamma, wm, wc = _ut_weights(n, alpha, 2.0, kappa)
    sqrt = Covariance(P, form="sqrt")
    points = np.vstack([np.zeros(n), sqrt.sigma_offsets(gamma)])
    m = max(1, n // 2)
    H = rng.normal(size=(m, n))
    ydev = points @ H.T + 0.05 * np.tanh(points @ H.T)
    ydev -= wm @ ydev
    xdev = points - wm @ points
    R = Covariance(_spd(m, rng) * 0.3)
    gain_d, post_d = Covariance(P, form="full").updated_unscented(xdev, ydev, wc, R)
    gain_s, post_s = _guarded(sqrt).updated_unscented(xdev, ydev, wc, R)
    assert post_s.form == "sqrt"
    np.testing.assert_allclose(gain_s, gain_d, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(post_s.as_matrix(), post_d.as_matrix(), rtol=1e-10, atol=1e-13)


def test_sqrt_paths_handle_rank_deficient_quaternion_covariance():
    """7x7 full-quaternion covariance with P q = 0 (rank 6): all three sqrt paths stay exact."""
    rng = np.random.default_rng(SEED + 300)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    xi = 0.5 * np.array([[-x, -y, -z], [w, -z, y], [z, w, -x], [-y, x, w]])
    J = np.zeros((7, 6))
    J[:3, :3] = np.eye(3)
    J[3:, 3:] = xi
    P7 = J @ _spd(6, rng) @ J.T
    null = np.r_[0.0, 0.0, 0.0, q]
    sqrt = Covariance(P7, form="sqrt", psd_policy="allow_indefinite")
    dense = Covariance(P7, form="full", psd_policy="allow_indefinite")
    H = rng.normal(size=(3, 7))
    R = Covariance(_spd(3, rng) * 0.2)
    for joseph in (True, False):
        _, post = sqrt.updated_linear(H, R, joseph=joseph)
        np.testing.assert_allclose(post.as_matrix(), dense.updated_linear(H, R)[1].as_matrix(), rtol=1e-10, atol=1e-13)
        assert np.linalg.norm(post.as_matrix() @ null) < 1e-13
    Jr = rng.normal(size=(7, 7))
    np.testing.assert_allclose(sqrt.transformed(Jr).as_matrix(), Jr @ P7 @ Jr.T, rtol=1e-10, atol=1e-13)
    gamma, wm, wc = _ut_weights(7)
    points = np.vstack([np.zeros(7), sqrt.sigma_offsets(gamma)])
    ydev = points @ H.T
    ydev -= wm @ ydev
    xdev = points - wm @ points
    _, post_s = sqrt.updated_unscented(xdev, ydev, wc, R)
    _, post_d = dense.updated_unscented(xdev, ydev, wc, R)
    np.testing.assert_allclose(post_s.as_matrix(), post_d.as_matrix(), rtol=1e-9, atol=1e-13)


def test_sqrt_paths_handle_structurally_zero_channel():
    """An exactly-zero variance channel (e.g. a perfectly known bias) must not NaN."""
    factor = np.diag([1.0, 0.5, 0.0])
    sqrt = Covariance.from_upper_factor(factor, psd_policy="allow_indefinite")
    dense = Covariance(factor.T @ factor, psd_policy="allow_indefinite")
    H = np.array([[1.0, 1.0, 0.0]])
    R = np.array([[0.1]])
    for joseph in (True, False):
        _, post = sqrt.updated_linear(H, R, joseph=joseph)
        np.testing.assert_allclose(post.as_matrix(), dense.updated_linear(H, R)[1].as_matrix(), atol=1e-14)
    xdev = np.vstack([np.zeros(3), np.sqrt(3) * factor, -np.sqrt(3) * factor])
    weights = np.r_[0.0, np.full(6, 1 / 6)]
    _, post = sqrt.updated_unscented(xdev, xdev @ H.T, weights, R)
    np.testing.assert_allclose(post.as_matrix(), dense.updated_unscented(xdev, xdev @ H.T, weights, R)[1].as_matrix(), atol=1e-14)


def test_sqrt_paths_zero_dimensions():
    empty = Covariance.zeros(0, form="sqrt")
    gain, post = empty.updated_linear(np.zeros((2, 0)), np.eye(2))
    assert gain.shape == (0, 2) and post.shape == (0, 0) and post.form == "sqrt"
    gain, post = empty.updated_unscented(np.zeros((1, 0)), np.zeros((1, 2)), np.ones(1), np.eye(2))
    assert gain.shape == (0, 2) and post.shape == (0, 0)
    assert empty.transformed(np.zeros((3, 0))).shape == (3, 3)
    prior = Covariance(np.eye(3) * 2.0, form="sqrt")
    gain, post = prior.updated_linear(np.zeros((0, 3)), np.zeros((0, 0)))
    assert gain.shape == (3, 0)
    np.testing.assert_allclose(post.as_matrix(), np.eye(3) * 2.0)
    assert prior.transformed(np.zeros((0, 3))).shape == (0, 0)


def test_indefinite_unscented_posterior_follows_psd_policy_in_sqrt_form():
    """When P - K Pyy K^T is genuinely indefinite the sqrt path defers to the dense policy semantics."""
    deviations = np.vstack([np.ones(3) * 3.0, np.eye(3), -np.eye(3)])
    weights = np.r_[-1.0, np.full(6, 0.5)]
    measurement = deviations @ np.array([[1.0], [0.5], [0.2]])
    for policy, expected in (("strict", np.linalg.LinAlgError), ("allow_indefinite", ValueError)):
        prior = Covariance(np.eye(3) * 0.5, form="sqrt", psd_policy=policy)
        with pytest.raises(expected):
            prior.updated_unscented(deviations, measurement, weights, np.array([[0.1]]))


def test_joseph_update_test_tolerance_is_absolute_for_analytically_zero_gain():
    """Regression guard for testing/test_helpers/test_covariance.py::test_linear_prediction_and_joseph_update:
    the second gain entry is analytically zero there, so the comparison needs an atol."""
    P = np.array([[2.0, 0.4], [0.4, 1.0]])
    F = np.array([[1.0, 0.2], [0.0, 1.0]])
    predicted = Covariance(P, form="sqrt").predicted_linear(F, np.diag([0.1, 0.2]))
    gain, _ = predicted.updated_linear(np.array([[1.0, -0.5]]), [[0.3]])
    assert abs(gain[1, 0]) < 1e-15


@pytest.mark.parametrize("fn", [cholupdate, choldowndate])
def test_rank_one_primitives_reject_wrong_length_vectors(fn):
    """Numba has no bounds checks: a short x reads past the buffer unless the wrapper validates it."""
    R = np.linalg.cholesky(_spd(5, np.random.default_rng(SEED))).T.copy()
    with pytest.raises((ValueError, IndexError)):
        fn(R, np.array([0.5, 0.5]))
    with pytest.raises((ValueError, IndexError, TypeError)):
        fn(R, np.zeros((1, 5)))
