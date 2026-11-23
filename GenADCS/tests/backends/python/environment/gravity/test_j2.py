import numpy as np

from GenADCS.backends.python.environment.gravity import J2


def test_j2_gravity_is_realistic_and_central():
    model = J2()
    position = np.array((7_000_000.0, 0.0, 0.0))
    acceleration = model.evaluate(position)

    assert 7.5 < np.linalg.norm(acceleration) < 8.5
    assert acceleration[0] < 0.0
    np.testing.assert_allclose(acceleration[1], 0.0, atol=1e-14)


def test_j2_gravity_perturbation_is_small_and_symmetric():
    point_mass = J2(degrees=())
    j2 = J2()
    p = np.array((7_000_000.0, 0.0, 1_000_000.0))
    delta = j2.evaluate(p) - point_mass.evaluate(p)

    assert 1e-4 < np.linalg.norm(delta) < 2e-2
    np.testing.assert_allclose(j2.evaluate((0.0, 7_000_000.0, 1_000_000.0))[0], 0.0, atol=1e-14)
