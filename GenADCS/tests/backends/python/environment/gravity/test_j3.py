import numpy as np

from GenADCS.backends.python.environment.gravity import J3


def test_j3_gravity_is_finite_and_has_north_south_asymmetry():
    model = J3()
    north = model.evaluate((7_000_000.0, 0.0, 1_000_000.0))
    south = model.evaluate((7_000_000.0, 0.0, -1_000_000.0))

    assert np.all(np.isfinite(north))
    assert not np.allclose(north, south)
