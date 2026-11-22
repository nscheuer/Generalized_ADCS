"""Reference checks for the standalone Numba NRLMSISE-00 implementation."""

import numpy as np
import pytest

from GenADCS.backends.python.environment.atmosphere import NRLMSISE00
from GenADCS.backends.python.environment.atmosphere.NRMLSISE00._nrlmsise00_core import msis_gtd7


def test_reference_c_case_and_effective_oxygen():
    ap_history = np.full(7, -1.0)
    ordinary = msis_gtd7(172, 29000.0, 400.0, 60.0, -70.0, 16.0,
                         150.0, 150.0, 4.0, ap_history, False)
    effective = msis_gtd7(172, 29000.0, 400.0, 60.0, -70.0, 16.0,
                          150.0, 150.0, 4.0, ap_history, True)
    assert ordinary.d[5] == pytest.approx(4.074713532757222e-12, rel=1e-12)
    assert ordinary.t[0] == pytest.approx(1250.53994356, rel=1e-10)
    assert effective.d[5] >= ordinary.d[5]
    assert msis_gtd7.nopython_signatures


def test_eci_density_and_corotation():
    model = NRLMSISE00()
    position = np.array([6778137.0, 0.0, 0.0])
    assert model.density(position, 2451545.0) == pytest.approx(
        4.99891e-12, rel=2e-5
    )
    assert np.allclose(model.velocity(position, 2451545.0),
                       [0.0, 494.2695449, 0.0], rtol=1e-9)
    assert model.density.nopython_signatures
    assert model.velocity.nopython_signatures


def test_storm_ap_history_and_invalid_length():
    model = NRLMSISE00()
    position = np.array([6778137.0, 0.0, 0.0])
    storm = model.density(position, 2451545.0, 150.0, 150.0, 10.0, True,
                          np.array([10., 12., 14., 16., 18., 20., 22.]))
    assert storm == pytest.approx(5.648493010892528e-12, rel=1e-12)
    with pytest.raises(ValueError, match="seven"):
        model.density(position, 2451545.0, 150.0, 150.0, 10.0, True,
                      np.array([10., 12.]))
