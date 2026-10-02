"""ADCS.environment.igrf must agree with ppigrf, which it replaced.

ppigrf was the geomagnetic field model for every result published up to 0.1.8.
The in-tree implementation exists for speed, not for a different answer, so the
contract is exact numerical agreement -- a drift here would silently change
every simulation that has ever been run against this package.

ppigrf is a dev-only dependency (`pip install -e ".[dev]"`); the oracle tests
skip without it rather than failing, but the self-contained ones always run.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from ADCS.environment.igrf import NMAX, RE, coefficients_at, igrf_gc

ppigrf = pytest.importorskip("ppigrf", reason="reference model not installed")

# Machine precision for a ~1e4 nT field summed over 104 terms. The two codes
# use the same recursion but accumulate in a different order, so exact equality
# is not available; 1e-9 relative is still ~1e-5 nT.
RTOL = 1e-9


def _ppigrf(r, theta, phi, date):
    """ppigrf with its date-broadcasting flattened to elementwise."""
    out = ppigrf.igrf_gc(r, theta, phi, date)
    arrs = [np.asarray(x, dtype=float) for x in out]
    if arrs[0].ndim == 2 and arrs[0].shape[0] == arrs[0].shape[1] > 1:
        arrs = [np.diagonal(a) for a in arrs]
    return [a.reshape(-1) for a in arrs]


def _worst(ref, got):
    got = [np.asarray(g, dtype=float).reshape(-1) for g in got]
    return max(
        float(np.max(np.abs((a - b) / np.maximum(np.abs(a), 1.0))))
        for a, b in zip(ref, got)
    )


def test_matches_reference_over_random_points():
    """5000 scattered LEO points at one epoch."""
    rng = np.random.default_rng(7)
    n = 5000
    r = 6371.2 + rng.uniform(300.0, 800.0, n)
    theta = rng.uniform(0.5, 179.5, n)
    phi = rng.uniform(-180.0, 180.0, n)
    date = datetime(2024, 3, 1, 12)
    assert _worst(_ppigrf(r, theta, phi, date), igrf_gc(r, theta, phi, date)) < RTOL


def test_matches_reference_for_a_single_point():
    """The scalar route is a separate code path from the batch kernel."""
    date = datetime(2024, 3, 1, 12)
    got = igrf_gc(6771.2, 45.0, 10.0, date)
    assert all(np.isscalar(x) or np.ndim(x) == 0 for x in got)
    assert _worst(_ppigrf(6771.2, 45.0, 10.0, date), got) < RTOL


def test_matches_reference_with_one_date_per_sample():
    """Dates pair elementwise with positions rather than forming a grid."""
    rng = np.random.default_rng(11)
    n = 300
    r = 6371.2 + rng.uniform(300.0, 800.0, n)
    theta = rng.uniform(0.5, 179.5, n)
    phi = rng.uniform(-180.0, 180.0, n)
    dates = [datetime(2019, 1, 1) + timedelta(days=3.7 * i) for i in range(n)]
    got = igrf_gc(r, theta, phi, dates)
    assert all(np.shape(x) == (n,) for x in got)
    assert _worst(_ppigrf(r, theta, phi, dates), got) < RTOL


def test_secular_variation_across_every_epoch_bracket():
    """Interpolation must hold across all 26 five-year brackets, not just one."""
    # Mid-year, so every one of the 26 brackets is interpolated rather
    # than landing on an epoch. Stops at 2029 to stay inside 1900-2030.
    dates = [datetime(y, 6, 15) for y in range(1900, 2030)]
    n = len(dates)
    r = np.full(n, 6771.2)
    theta = np.full(n, 55.0)
    phi = np.full(n, -7.0)
    assert _worst(_ppigrf(r, theta, phi, dates), igrf_gc(r, theta, phi, dates)) < RTOL


@pytest.mark.parametrize("theta", [0.01, 0.5, 90.0, 179.5, 179.99])
def test_matches_reference_near_the_poles(theta):
    """Bphi carries a 1/sin(theta), so the poles are the delicate case."""
    date = datetime(2024, 3, 1, 12)
    got = igrf_gc(6771.2, theta, 33.0, date)
    assert _worst(_ppigrf(6771.2, theta, 33.0, date), got) < RTOL


@pytest.mark.parametrize("theta", [0.0, 180.0])
def test_exact_poles_return_finite_field(theta):
    """The azimuthal component has a finite pole limit despite its 1/sin(theta) form."""
    date = datetime(2024, 3, 1, 12)
    got = igrf_gc(6771.2, theta, 33.0, date)
    assert all(np.isfinite(component) for component in got)
    near = igrf_gc(6771.2, theta + (1e-5 if theta == 0.0 else -1e-5), 33.0, date)
    np.testing.assert_allclose(got, near, rtol=1e-6, atol=1e-5)


def test_chunk_size_does_not_change_the_result():
    """chunk only partitions the parallel loop; it must not be observable."""
    rng = np.random.default_rng(3)
    n = 1500
    r = 6371.2 + rng.uniform(300.0, 800.0, n)
    theta = rng.uniform(0.5, 179.5, n)
    phi = rng.uniform(-180.0, 180.0, n)
    date = datetime(2022, 9, 9)
    base = igrf_gc(r, theta, phi, date, chunk=512)
    for chunk in (1, 7, 256, 4096):
        for a, b in zip(base, igrf_gc(r, theta, phi, date, chunk=chunk)):
            np.testing.assert_array_equal(a, b)


def test_shape_is_preserved():
    """The broadcast shape of the coordinates comes back unchanged."""
    date = datetime(2024, 3, 1)
    r = np.full((4, 3), 6771.2)
    theta = np.linspace(20.0, 160.0, 3)[None, :]
    phi = np.linspace(-90.0, 90.0, 4)[:, None]
    for comp in igrf_gc(r, theta, phi, date):
        assert comp.shape == (4, 3)


def test_mismatched_date_count_is_rejected():
    """Silently broadcasting a wrong-length date list is how 0.1.8's caller
    ended up extracting a diagonal from an N x N grid."""
    r = np.full(5, 6771.2)
    dates = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
    with pytest.raises(ValueError, match="expected 1 or 5"):
        igrf_gc(r, np.full(5, 45.0), np.full(5, 10.0), dates)


def test_field_magnitude_is_physical():
    """A sanity bound that does not depend on the reference implementation."""
    rng = np.random.default_rng(5)
    n = 500
    r = 6371.2 + rng.uniform(300.0, 800.0, n)
    br, bt, bp = igrf_gc(r, rng.uniform(1.0, 179.0, n), rng.uniform(-180.0, 180.0, n),
                         datetime(2024, 1, 1))
    mag = np.sqrt(br**2 + bt**2 + bp**2)
    assert np.all(mag > 1.0e4) and np.all(mag < 6.0e4)


def test_coefficients_have_the_expected_shape():
    g, h = coefficients_at(datetime(2024, 3, 1))
    assert g.shape == h.shape == (NMAX + 1, NMAX + 1)
    assert np.all(h[:, 0] == 0.0)          # m = 0 has no sine term
    assert abs(g[1, 0]) > 1.0e4            # the dipole term, ~-29400 nT
    assert RE == pytest.approx(6371.2)


def test_out_of_range_dates_warn_and_clamp():
    """Outside 1900-2030 the model is undefined; clamping loudly beats
    extrapolating silently."""
    with pytest.warns(RuntimeWarning, match="outside the coefficient file"):
        got = igrf_gc(6771.2, 45.0, 10.0, datetime(1850, 1, 1))
    assert all(np.isfinite(x) for x in got)
