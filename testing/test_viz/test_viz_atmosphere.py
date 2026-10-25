"""Precomputed atmosphere tables (Hillaire 2020 / Bruneton 2017 parameterisation)."""

import numpy as np
import pytest

from ADCS.viz.assets.atmosphere import (
    AtmosphereParams,
    _sample_transmittance,
    multiscattering_table,
    transmittance_table,
    transmittance_uv,
)
from ADCS.viz.assets.fetch import fetch

P = AtmosphereParams()


@pytest.fixture(scope="module")
def T():
    return transmittance_table(P)


def test_zenith_optical_depth_matches_the_analytic_column(T):
    # Exponential layers: tau = beta * H; ozone tent: tau = beta * half-width.
    tau_expected = (np.array(P.rayleigh_scattering) * P.rayleigh_scale_height
                    + np.array(P.mie_extinction) * P.mie_scale_height
                    + np.array(P.ozone_absorption) * P.ozone_half_width)
    zen = _sample_transmittance(P, T, np.array([P.bottom_radius]), np.array([1.0]))[0]
    np.testing.assert_allclose(-np.log(zen), tau_expected, rtol=2e-3)


def test_transmittance_is_physical(T):
    assert T.dtype == np.float32 and T.shape == (64, 256, 3)
    assert np.all((T > 0) & (T <= 1.0 + 1e-6))
    # Blue is attenuated more than red; the horizon more than the zenith.
    hor = _sample_transmittance(P, T, np.array([P.bottom_radius]), np.array([0.0]))[0]
    zen = _sample_transmittance(P, T, np.array([P.bottom_radius]), np.array([1.0]))[0]
    assert hor[2] < hor[1] < hor[0] < zen[0]
    # At the top of the atmosphere looking up there is nothing left to cross.
    top = _sample_transmittance(P, T, np.array([P.top_radius]), np.array([1.0]))[0]
    np.testing.assert_allclose(top, 1.0, atol=1e-5)


def test_texture_mapping_covers_the_unit_square():
    u, v = transmittance_uv(P, np.array([P.bottom_radius, P.top_radius]), np.array([1.0, 1.0]))
    W, H = P.transmittance_size
    assert 0.5 / W - 1e-12 <= u.min() and u.max() <= 1 - 0.5 / W + 1e-12
    np.testing.assert_allclose(v, [0.5 / H, 1 - 0.5 / H])


def test_multiple_scattering_is_physical(T):
    small = AtmosphereParams(multiscattering_size=(8, 4))
    M = multiscattering_table(small, T, directions=32, steps=16)
    assert M.shape == (4, 8, 3) and np.all(M >= 0)
    assert np.all(M[:, 0] == 0)            # Sun far below the horizon: no light to scatter
    assert np.all(M[0, -1] > M[0, 4])      # more light with the Sun overhead
    assert M[0, -1, 2] > M[0, -1, 0]       # sky light is bluer than red


def test_fetch_of_no_sources_fetches_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("ADCS_VIZ_CACHE", str(tmp_path))
    sources = {"x": {"url": "https://example.invalid/never", "bytes": 1, "sha256": None}}
    assert fetch([], sources=sources, log=lambda *_: None) == []
