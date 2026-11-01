import numpy as np
import pytest

from ADCS.environment.atmosphere import SMADAtmosphere


def test_vector_density_matches_scalar_across_interpolation_regions():
    model = SMADAtmosphere()
    altitudes = np.array([-10.0, 0.0, 50.0, 400.0, 1000.0, 1200.0, 35000.0])
    vectorized = model.interpolate(altitudes)
    scalar = np.array([model.interpolate(altitude) for altitude in altitudes])
    np.testing.assert_allclose(vectorized, scalar, rtol=1e-14, atol=0.0)


def test_smad_density_vector_path_matches_scalar_calls():
    model = SMADAtmosphere()
    altitudes = np.array([-5.0, 50.0, 225.0, 1000.0, 20000.0])
    positions = np.column_stack((6378.137 + altitudes, np.zeros_like(altitudes), np.zeros_like(altitudes)))
    density = model.density(positions)
    expected = np.array([model.density(position) for position in positions])
    np.testing.assert_allclose(density, expected, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("rho_range", [[1e-9, 1e-12, 1e-12], [1e-9, 1e-12, 1e-11]])
def test_rejects_tables_that_do_not_decay_at_the_top(rho_range):
    """The scale height above the table would be infinite or negative."""
    with pytest.raises(ValueError, match="decrease"):
        SMADAtmosphere(altitude_range=[0.0, 500.0, 1000.0], rho_range=rho_range)
