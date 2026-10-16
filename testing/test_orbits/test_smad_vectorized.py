import numpy as np

from ADCS.environment.atmosphere import SMADAtmosphere
from ADCS.environment.atmosphere.density_model import DensityModel


def test_vector_density_matches_scalar_across_interpolation_regions():
    model = DensityModel()
    altitudes = np.array([-10.0, 0.0, 50.0, 400.0, 1000.0, 1200.0, 35000.0])
    vectorized = model.interpolate_many(altitudes)
    scalar = np.array([model.interpolate(altitude) for altitude in altitudes])
    np.testing.assert_allclose(vectorized, scalar, rtol=1e-14, atol=0.0)


def test_smad_density_vector_path_matches_scalar_calls():
    model = SMADAtmosphere()
    altitudes = np.array([-5.0, 50.0, 225.0, 1000.0, 20000.0])
    positions = np.column_stack((6378.137 + altitudes, np.zeros_like(altitudes), np.zeros_like(altitudes)))
    density = model.density(positions)
    expected = np.array([model.density(position) for position in positions])
    np.testing.assert_allclose(density, expected, rtol=1e-14, atol=0.0)
