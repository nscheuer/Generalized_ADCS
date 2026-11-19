"""Offline checks for the time and geodetic inputs passed to NRLMSISE."""

from datetime import datetime, timezone
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ADCS.environment import NRLMSISE00


def test_nrlmsise_forwards_aligned_date_and_geodetic_values(monkeypatch):
    captured = {}

    def calculate(dates, lons, lats, alts, **kwargs):
        captured.update(dates=dates, lons=lons, lats=lats, alts=alts, kwargs=kwargs)
        return np.array([[[[[2.5e-12] + [0.0] * 10]]], [[[[3.5e-12] + [0.0] * 10]]]])

    monkeypatch.setitem(sys.modules, "pymsis", SimpleNamespace(calculate=calculate))
    model = NRLMSISE00(version=0, f107=120, f107a=115, ap=4)
    dates = [datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 1)]
    lla = np.array([[12.0, 34.0, 400.0], [-10.0, 80.0, 450.0]])
    density = model.density(np.array([[6800.0, 0.0, 0.0], [6800.0, 1.0, 0.0]]), dates, lla)

    np.testing.assert_allclose(density, [2.5e-12, 3.5e-12])
    np.testing.assert_allclose(captured["lons"], [34.0, 80.0])
    np.testing.assert_allclose(captured["lats"], [12.0, -10.0])
    np.testing.assert_allclose(captured["alts"], [400.0, 450.0])
    assert captured["kwargs"]["version"] == 0
    np.testing.assert_array_equal(captured["kwargs"]["f107s"], [120, 120])
    np.testing.assert_array_equal(captured["kwargs"]["f107as"], [115, 115])
    np.testing.assert_array_equal(captured["kwargs"]["aps"], np.full((2, 7), 4.0))


def test_nrlmsise_pinned_indices_with_real_pymsis():
    """A batch with pinned indices runs through the real backend.

    The mocked tests above check what is passed; this one checks that PyMSIS
    accepts it. Pinned indices need no space-weather download.
    """
    pymsis = pytest.importorskip("pymsis")
    model = NRLMSISE00(version=0, f107=150, f107a=150, ap=4)
    dates = [datetime(2024, 1, 1, h) for h in (0, 1, 2)]
    lla = np.array([[0.0, 0.0, 400.0], [45.0, 90.0, 450.0], [-60.0, 200.0, 500.0]])
    density = model.density(np.zeros((3, 3)) + [6800.0, 0.0, 0.0], dates, lla)

    expected = [
        pymsis.calculate(np.datetime64(d), lon, lat, alt, f107s=[150.0], f107as=[150.0],
                         aps=[[4.0] * 7], version=0).reshape(-1)[0]
        for d, (lat, lon, alt) in zip(dates, lla)
    ]
    np.testing.assert_allclose(density, expected, rtol=1e-12)


def test_nrlmsise_requires_date_for_orbit_density():
    with pytest.raises(ValueError, match="datetime_utc is required"):
        NRLMSISE00().density(np.array([6800.0, 0.0, 0.0]))


def test_nrlmsise_derives_geodetic_for_timezone_aware_date(monkeypatch):
    captured = {}

    def calculate(dates, lons, lats, alts, **kwargs):
        captured.update(dates=dates, lons=lons, lats=lats, alts=alts)
        return np.array([[[[[1e-12] + [0.0] * 10]]]])

    monkeypatch.setitem(sys.modules, "pymsis", SimpleNamespace(calculate=calculate))
    model = NRLMSISE00(version=0, f107=120, f107a=115, ap=4)
    density = model.density(
        np.array([6800.0, 0.0, 0.0]),
        datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    assert density == pytest.approx(1e-12)
    assert captured["dates"][0] == np.datetime64("2024-01-01T00:00:00")
    assert np.isfinite(captured["lats"][0])
    assert np.isfinite(captured["lons"][0])
    assert np.isfinite(captured["alts"][0])
