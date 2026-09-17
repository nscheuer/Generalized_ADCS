from __future__ import annotations

import numpy as np
import pytest

from ADCS.orbits.density_model import DensityModel


@pytest.mark.parametrize(
    ("altitudes", "densities"),
    [([0.0, 1.0], [1.0]), ([-1.0, 1.0], [1.0, 0.1]), ([0.0, 1.0], [1.0, 0.0])],
)
def test_density_model_rejects_invalid_tables(altitudes, densities):
    with pytest.raises(ValueError):
        DensityModel(np.array(altitudes), np.array(densities))


@pytest.mark.parametrize("altitude", [-100.0, 0.0, 25.0, 49.9, 50.0])
def test_density_model_clamps_below_its_table(altitude):
    model = DensityModel(np.array([50.0, 100.0]), np.array([2.0, 0.5]))
    assert model.interpolate(altitude) == pytest.approx(2.0)


@pytest.mark.parametrize("fraction", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
def test_density_model_is_log_linear_inside_a_table_interval(fraction):
    model = DensityModel(np.array([100.0, 200.0]), np.array([1.0e-6, 1.0e-8]))
    expected = 1.0e-6 * ((1.0e-2) ** fraction)
    assert model.interpolate(100.0 + 100.0 * fraction) == pytest.approx(expected)


@pytest.mark.parametrize("altitude", [250.0, 500.0, 1000.0])
def test_density_model_extrapolates_with_the_last_scale_height(altitude):
    model = DensityModel(np.array([100.0, 200.0]), np.array([1.0e-6, 1.0e-8]))
    assert model.interpolate(altitude) < model.interpolate(200.0)


def test_density_model_handles_a_non_decaying_final_interval_and_has_a_concise_repr():
    model = DensityModel(np.array([100.0, 200.0]), np.array([1.0e-8, 1.0e-6]))
    assert model.interpolate(500.0) == pytest.approx(1.0e-6)
    assert repr(model) == "DensityModel(n=2)"
