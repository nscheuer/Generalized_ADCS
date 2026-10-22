r"""
Solar-activity-dependent atmospheric density (DensityModel.solar_level).

Quiet-Sun (solar minimum, the default) reproduces the legacy table behavior;
raising solar_level toward 1 (solar maximum) amplifies density by an altitude-
dependent factor (small low down, ~2 orders of magnitude near 1000 km).
"""
import numpy as np
import pytest

from ADCS.orbits.density_model import (
    DensityModel, SMAD_altrange, SMAD_rhovsalt, solar_amplification, f107_to_solar_level,
)


def test_default_is_quiet_and_backward_compatible():
    dm = DensityModel()
    assert dm.solar_level == 0.0
    # exact legacy table values at sample altitudes, scalar -> float
    for h, rho in zip(SMAD_altrange, SMAD_rhovsalt):
        got = dm.interpolate(float(h))
        assert isinstance(got, float)
        assert np.isclose(got, rho, rtol=1e-12)
    # solar_level=0 is a no-op vs default
    assert DensityModel(solar_level=0.0).interpolate(650.0) == dm.interpolate(650.0)


def test_interpolate_accepts_arrays_matching_scalar():
    dm = DensityModel()
    hs = np.array([-50.0, 0.0, 312.0, 650.0, 925.0, 1500.0])
    arr = dm.interpolate(hs)
    assert isinstance(arr, np.ndarray) and arr.shape == hs.shape
    assert np.allclose(arr, [dm.interpolate(float(h)) for h in hs], rtol=0, atol=0)


def test_solar_level_raises_density_monotonically():
    h = 650.0
    rhos = [DensityModel(solar_level=s).interpolate(h) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(rhos[i] < rhos[i + 1] for i in range(len(rhos) - 1))
    # quiet->max amplification at 650 km is sizeable (tens x)
    assert rhos[-1] / rhos[0] > 10.0


def test_solar_amplification_grows_with_altitude():
    a200 = solar_amplification(200.0, 1.0)
    a500 = solar_amplification(500.0, 1.0)
    a900 = solar_amplification(900.0, 1.0)
    assert 1.0 <= a200 < a500 < a900
    assert solar_amplification(650.0, 0.0) == 1.0  # level 0 -> factor 1
    # scaling is log-linear in level: amp(level=0.5) == sqrt(amp(level=1))
    assert np.isclose(solar_amplification(700.0, 0.5), np.sqrt(solar_amplification(700.0, 1.0)))


def test_from_f107_and_level_mapping():
    assert f107_to_solar_level(70.0) == 0.0
    assert f107_to_solar_level(250.0) == 1.0
    assert np.isclose(f107_to_solar_level(160.0), 0.5)
    assert f107_to_solar_level(30.0) == 0.0 and f107_to_solar_level(400.0) == 1.0  # clamped
    # from_f107 builds the matching model
    assert DensityModel.from_f107(250.0).interpolate(650.0) == DensityModel.for_solar_level(1.0).interpolate(650.0)


def test_above_and_below_table_are_handled():
    dm = DensityModel(solar_level=1.0)
    # below table clamps to surface value (x amplification at that low alt ~1)
    assert dm.interpolate(-100.0) > 0
    # above table keeps decaying (no spurious floor), still positive
    assert 0 < dm.interpolate(3000.0) < dm.interpolate(1000.0)


def test_negative_solar_level_rejected():
    with pytest.raises(ValueError):
        DensityModel(solar_level=-0.1)
