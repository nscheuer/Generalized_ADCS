"""Fast VSOP2013 tests using a tiny synthetic Chebyshev table."""

import numpy as np

from ADCS.environment.ephemeris import VSOP2013
from ADCS.environment.ephemeris.vsop2013 import _chebyshev_xyz


def _write_tiny_kernel(path):
    starts = [1 + 6 * index for index in range(9)]
    with path.open("w", encoding="ascii") as stream:
        stream.write(" 2013\n")
        stream.write("1721006.5\n")
        stream.write("1721038.5\n")
        stream.write("32.0\n")
        stream.write("     1\n")
        stream.write(" 978\n")
        stream.write(" ".join(f"{value:04d}" for value in starts) + "\n")
        stream.write(" ".join("0001" for _ in range(9)) + "\n")
        stream.write(" ".join("0001" for _ in range(9)) + "\n")

        coefficients = np.zeros(978)
        coefficients[starts[2] - 1] = 1.0  # Earth-Moon barycenter, x = 1 AU
        coefficients[starts[3] - 1] = 2.0  # Mars, x = 2 AU
        stream.write(" 1721006.5 1721038.5\n")
        for row in range(163):
            values = coefficients[row * 6 : (row + 1) * 6]
            stream.write("  " + "".join(f"{value:20.16f}{0:4d}" for value in values) + "\n")


def test_vsop2013_returns_planet_positions_relative_to_earth(tmp_path):
    data_file = tmp_path / "VSOP2013.p1000"
    _write_tiny_kernel(data_file)
    model = VSOP2013(kernel_dir=tmp_path)
    epoch = (1721006.5 - 2451545.0) / 36525.0

    mars = model.position_eci("mars", epoch)
    sun = model.sun_position_eci(epoch)

    np.testing.assert_allclose(np.linalg.norm(mars), VSOP2013.AU_KM, rtol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(sun), VSOP2013.AU_KM, rtol=1e-12)


def test_vsop2013_supports_vector_epochs_and_earth_aliases(tmp_path):
    _write_tiny_kernel(tmp_path / "VSOP2013.p1000")
    model = VSOP2013(kernel_dir=tmp_path)
    epoch = (1721006.5 - 2451545.0) / 36525.0

    result = model.position_eci("mars", [epoch, epoch], observer="earth-moon barycenter")

    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result[0], result[1])
    assert "pluto" in model.available_bodies


def test_numba_chebyshev_evaluation_matches_numpy():
    rng = np.random.default_rng(4)
    count = 24
    coefficients = rng.normal(size=3 * count)
    for x in (-1.0, -0.73, 0.0, 0.41, 1.0):
        actual = _chebyshev_xyz(coefficients, 0, count, x)
        expected = np.array([
            np.polynomial.chebyshev.chebval(x, coefficients[axis * count : (axis + 1) * count])
            for axis in range(3)
        ])
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
