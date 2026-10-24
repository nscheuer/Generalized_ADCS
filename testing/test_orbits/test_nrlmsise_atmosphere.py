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
    assert captured["kwargs"] == {"version": 0, "f107s": 120, "f107as": 115, "aps": 4}


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
