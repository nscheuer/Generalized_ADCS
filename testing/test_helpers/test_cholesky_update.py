r"""Tests for the in-tree rank-1 Cholesky update/downdate.

These replace the external ``choldate`` C extension, which the SRUAKF depends
on for every sigma-point covariance update, so they are held to the defining
identity rather than to a reference implementation:

.. math::

    \tilde{R}^\top \tilde{R} = R^\top R \pm x x^\top

Coverage: the identity across sizes/conditioning, exact agreement with SciPy's
dense Cholesky, upper-triangularity and positive diagonal, in-place semantics,
caller-array preservation, idempotence (update then downdate), scaling,
zero/unit edge cases, non-positive-definite detection, ill-conditioned and
near-singular factors, dtype/layout robustness, and equivalence to the
sequential application of several rank-1 updates.
"""

import numpy as np
import pytest
import scipy.linalg

from ADCS.helpers.cholesky_update import cholupdate, choldowndate

SEED = 20260804


def _random_spd(n, rng, cond=None):
    """Random symmetric positive definite matrix, optionally conditioned.

    The ``errstate`` guard is for the platform BLAS, not for anything under
    test: on Apple Accelerate this ``matmul`` raises spurious divide-by-zero /
    overflow flags on well-conditioned inputs. Verified to fire in pure NumPy
    before ``ADCS.helpers.cholesky_update`` is imported at all. The identity
    assertions below still check finiteness explicitly, so a genuine NaN out
    of the code under test fails loudly.
    """
    with np.errstate(all="ignore"):
        Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        if cond is None:
            d = np.exp(rng.normal(scale=0.5, size=n)) + 0.5
        else:
            d = np.logspace(0, -np.log10(cond), n)
        return Q @ np.diag(d) @ Q.T


def _chol_upper(A):
    """Upper-triangular R with ``A = R.T @ R``, C-contiguous."""
    with np.errstate(all="ignore"):
        return np.linalg.cholesky(A).T.copy(order="C")


def _gram(R):
    """``R.T @ R`` without the platform BLAS's spurious status flags."""
    with np.errstate(all="ignore"):
        return R.T @ R


# ---------------------------------------------------------------------------
# The defining identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 12, 20])
def test_update_satisfies_identity(n):
    rng = np.random.default_rng(SEED + n)
    for _ in range(25):
        A = _random_spd(n, rng)
        R = _chol_upper(A)
        x = rng.normal(size=n)
        cholupdate(R, x)
        assert np.all(np.isfinite(R)), "update produced a non-finite factor"
        assert np.allclose(_gram(R), A + np.outer(x, x), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 12, 20])
def test_downdate_satisfies_identity(n):
    rng = np.random.default_rng(SEED + 100 + n)
    checked = 0
    for _ in range(50):
        A = _random_spd(n, rng)
        x = rng.normal(size=n) * 0.1
        target = A - np.outer(x, x)
        if np.min(np.linalg.eigvalsh(target)) < 1e-9:
            continue
        R = _chol_upper(A)
        choldowndate(R, x)
        assert np.all(np.isfinite(R)), (
            "downdate NaN'd on a positive-definite target "
            f"(min eig {np.min(np.linalg.eigvalsh(target)):.3e})")
        assert np.allclose(_gram(R), target, atol=1e-10, rtol=1e-10)
        checked += 1
    assert checked > 0, "no positive-definite downdate cases were exercised"


@pytest.mark.parametrize("n", [2, 5, 9])
def test_matches_scipy_dense_cholesky(n):
    """The updated factor equals a fresh factorization of the updated matrix."""
    rng = np.random.default_rng(SEED + 200 + n)
    for _ in range(20):
        A = _random_spd(n, rng)
        x = rng.normal(size=n)
        R = _chol_upper(A)
        cholupdate(R, x)
        R_ref = scipy.linalg.cholesky(A + np.outer(x, x), lower=False)
        assert np.allclose(R, R_ref, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 4, 7])
def test_result_stays_upper_triangular_with_positive_diagonal(n):
    rng = np.random.default_rng(SEED + 300 + n)
    for _ in range(20):
        A = _random_spd(n, rng)
        R = _chol_upper(A)
        x = rng.normal(size=n)
        cholupdate(R, x)
        assert np.allclose(np.tril(R, -1), 0.0), "lower triangle must stay zero"
        assert np.all(np.diag(R) > 0.0), "diagonal must stay positive"


def test_update_is_in_place_and_returns_none():
    rng = np.random.default_rng(SEED)
    A = _random_spd(4, rng)
    R = _chol_upper(A)
    before = R.copy()
    out = cholupdate(R, rng.normal(size=4))
    assert out is None, "cholupdate must modify in place, like choldate"
    assert not np.allclose(R, before), "R must actually change"


def test_caller_vector_is_not_clobbered():
    """choldate destroys x; this implementation must not."""
    rng = np.random.default_rng(SEED)
    for fn in (cholupdate, choldowndate):
        R = _chol_upper(_random_spd(5, rng) + 5 * np.eye(5))
        x = rng.normal(size=5) * 0.05
        x0 = x.copy()
        fn(R, x)
        assert np.array_equal(x, x0), f"{fn.__name__} modified the caller's x"


# ---------------------------------------------------------------------------
# Algebraic properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 6])
def test_update_then_downdate_is_identity(n):
    rng = np.random.default_rng(SEED + 400 + n)
    for _ in range(20):
        A = _random_spd(n, rng)
        R = _chol_upper(A)
        R0 = R.copy()
        x = rng.normal(size=n) * 0.3
        cholupdate(R, x)
        choldowndate(R, x)
        assert np.allclose(R, R0, atol=1e-9, rtol=1e-9)


def test_sequential_updates_match_batch_outer_product():
    """k successive rank-1 updates equal one rank-k update of the matrix."""
    rng = np.random.default_rng(SEED + 500)
    n, k = 6, 5
    A = _random_spd(n, rng)
    X = rng.normal(size=(k, n)) * 0.4
    R = _chol_upper(A)
    for row in X:
        cholupdate(R, row)
    assert np.allclose(_gram(R), A + X.T @ X, atol=1e-9, rtol=1e-9)


def test_update_is_order_independent():
    rng = np.random.default_rng(SEED + 600)
    n = 5
    A = _random_spd(n, rng)
    X = rng.normal(size=(4, n)) * 0.3
    Ra = _chol_upper(A)
    for row in X:
        cholupdate(Ra, row)
    Rb = _chol_upper(A)
    for row in X[::-1]:
        cholupdate(Rb, row)
    assert np.allclose(Ra, Rb, atol=1e-9, rtol=1e-9)


def test_scaling_the_update_vector_scales_the_outer_product():
    rng = np.random.default_rng(SEED + 700)
    A = _random_spd(4, rng)
    x = rng.normal(size=4)
    for alpha in (0.1, 1.0, 3.0, 50.0):
        R = _chol_upper(A)
        cholupdate(R, alpha * x)
        assert np.allclose(_gram(R), A + alpha**2 * np.outer(x, x),
                           atol=1e-8, rtol=1e-9)


def test_sign_of_update_vector_is_irrelevant():
    """x x^T == (-x)(-x)^T, so the factor must be identical."""
    rng = np.random.default_rng(SEED + 800)
    A = _random_spd(5, rng)
    x = rng.normal(size=5)
    Rp, Rm = _chol_upper(A), _chol_upper(A)
    cholupdate(Rp, x)
    cholupdate(Rm, -x)
    assert np.allclose(Rp, Rm, atol=1e-12)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_zero_vector_leaves_factor_unchanged():
    rng = np.random.default_rng(SEED + 900)
    for fn in (cholupdate, choldowndate):
        A = _random_spd(5, rng)
        R = _chol_upper(A)
        R0 = R.copy()
        fn(R, np.zeros(5))
        assert np.allclose(R, R0, atol=1e-13), f"{fn.__name__} moved on x=0"


def test_scalar_1x1_case():
    R = np.array([[3.0]])
    cholupdate(R, np.array([4.0]))
    assert np.isclose(R[0, 0], 5.0), "1x1 update is a Pythagorean sum"
    R = np.array([[5.0]])
    choldowndate(R, np.array([3.0]))
    assert np.isclose(R[0, 0], 4.0)


def test_identity_matrix_update():
    R = np.eye(3)
    x = np.array([1.0, 0.0, 0.0])
    cholupdate(R, x)
    assert np.allclose(_gram(R), np.eye(3) + np.outer(x, x), atol=1e-13)


# ---------------------------------------------------------------------------
# Failure detection -- the behaviour that differs from choldate
# ---------------------------------------------------------------------------

def test_non_positive_definite_downdate_yields_nan():
    """choldate silently returns a wrong factor here; we must produce NaN.

    ``A - x x^T`` is indefinite, so no real Cholesky factor exists. The SRUAKF
    already guards on ``np.any(np.isnan(...))``; that guard only fires if the
    downdate reports failure this way.
    """
    A = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])
    R = _chol_upper(A)
    choldowndate(R, np.array([10.0, 10.0, 10.0]))
    assert np.any(np.isnan(R)), "non-PD downdate must be detectable"


def test_downdate_exactly_to_singular_is_flagged():
    """Removing the whole matrix leaves a singular result, not a valid factor."""
    x = np.array([1.0, 2.0, -0.5])
    A = np.outer(x, x) + 1e-12 * np.eye(3)
    R = _chol_upper(A)
    choldowndate(R, x)
    assert np.any(np.isnan(R)) or np.min(np.diag(R)) < 1e-5


@pytest.mark.parametrize("scale", [1.5, 2.0, 10.0, 1e3])
def test_progressively_larger_downdates_eventually_fail_not_silently(scale):
    rng = np.random.default_rng(SEED + 1000)
    A = _random_spd(4, rng)
    x = rng.normal(size=4) * scale
    target = A - np.outer(x, x)
    R = _chol_upper(A)
    choldowndate(R, x)
    if np.min(np.linalg.eigvalsh(target)) > 1e-9:
        assert np.allclose(_gram(R), target, atol=1e-8, rtol=1e-8)
    else:
        assert np.any(np.isnan(R)), (
            "an indefinite downdate must NaN rather than return a wrong factor")


# ---------------------------------------------------------------------------
# Conditioning and robustness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cond", [1e2, 1e6, 1e10])
def test_ill_conditioned_matrices(cond):
    rng = np.random.default_rng(SEED + 1100)
    n = 6
    A = _random_spd(n, rng, cond=cond)
    R = _chol_upper(A)
    x = rng.normal(size=n) * np.sqrt(np.trace(A) / n) * 0.1
    cholupdate(R, x)
    target = A + np.outer(x, x)
    # Relative tolerance scaled by the conditioning of the problem.
    assert np.allclose(_gram(R), target, atol=1e-12 * cond, rtol=1e-8)


def test_very_large_and_very_small_magnitudes():
    for s in (1e-8, 1e-4, 1e4, 1e8):
        A = np.diag([s, s, s]).astype(float)
        R = _chol_upper(A)
        x = np.array([s**0.5, 0.0, 0.0]) * 0.5
        cholupdate(R, x)
        assert np.allclose(_gram(R), A + np.outer(x, x), rtol=1e-9)


def test_accepts_non_contiguous_and_integer_input_vectors():
    """SRUAKF slices rows out of 2-D arrays, so x is often a view."""
    rng = np.random.default_rng(SEED + 1200)
    A = _random_spd(4, rng)

    X = rng.normal(size=(4, 8))[:, ::2]        # non-contiguous view
    R = _chol_upper(A)
    cholupdate(R, X[0])
    assert np.allclose(_gram(R), A + np.outer(X[0], X[0]), atol=1e-9)

    R = _chol_upper(A)
    xi = np.array([1, 0, 2, 0])                # integer dtype
    cholupdate(R, xi)
    assert np.allclose(_gram(R), A + np.outer(xi, xi).astype(float), atol=1e-9)


def test_repeated_updates_stay_stable():
    """A long run of updates must not drift away from the identity."""
    rng = np.random.default_rng(SEED + 1300)
    n = 5
    A = np.eye(n)
    R = _chol_upper(A)
    acc = A.copy()
    for _ in range(300):
        x = rng.normal(size=n) * 0.1
        cholupdate(R, x)
        acc = acc + np.outer(x, x)
    assert np.allclose(_gram(R), acc, atol=1e-8, rtol=1e-9)
    assert np.all(np.isfinite(R))


# ---------------------------------------------------------------------------
# Agreement with choldate, where it is installed
# ---------------------------------------------------------------------------

def test_agrees_with_choldate_if_available():
    """Bit-level parity with the package this replaces, when it is present."""
    choldate = pytest.importorskip("choldate")
    rng = np.random.default_rng(SEED + 1400)
    worst = 0.0
    for _ in range(200):
        n = int(rng.integers(2, 9))
        A = _random_spd(n, rng)
        R0 = _chol_upper(A)
        x = rng.normal(size=n) * 0.4

        Rm, Rc = R0.copy(), R0.copy()
        cholupdate(Rm, x)
        choldate.cholupdate(Rc, x.copy())
        worst = max(worst, float(np.abs(Rm - Rc).max()))

        if np.min(np.linalg.eigvalsh(A - np.outer(x, x))) > 1e-8:
            Rm, Rc = R0.copy(), R0.copy()
            choldowndate(Rm, x)
            choldate.choldowndate(Rc, x.copy())
            worst = max(worst, float(np.abs(Rm - Rc).max()))
    assert worst < 1e-12, f"diverged from choldate by {worst:.3e}"
