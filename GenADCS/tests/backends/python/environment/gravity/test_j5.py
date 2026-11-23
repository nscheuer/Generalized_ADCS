import numpy as np

from GenADCS.backends.python.environment.gravity import J5


def test_j5_gravity_is_finite_and_perturbative():
    point_mass = J5(degrees=())
    model = J5()
    position = np.array((7_000_000.0, 0.0, 1_000_000.0))
    delta = model.evaluate(position) - point_mass.evaluate(position)

    assert np.all(np.isfinite(model.evaluate(position)))
    assert 1e-7 < np.linalg.norm(delta) < 1e-2
