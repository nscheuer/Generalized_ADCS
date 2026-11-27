import numpy as np

from GenADCS.backends.python.environment.gravity import J2


def test_j2_gravity_is_realistic_and_central() -> None:
    model = J2()
    position = np.array((7_000_000.0, 0.0, 0.0))
    acceleration = model.evaluate(position)

    assert 7.5 < np.linalg.norm(acceleration) < 8.5
    assert acceleration[0] < 0.0
    np.testing.assert_allclose(acceleration[1], 0.0, atol=1e-14)


def test_j2_gravity_perturbation_is_small_and_symmetric() -> None:
    point_mass = J2(degrees=())
    j2 = J2()
    p = np.array((7_000_000.0, 0.0, 1_000_000.0))
    delta = j2.evaluate(p) - point_mass.evaluate(p)

    assert 1e-4 < np.linalg.norm(delta) < 2e-2
    np.testing.assert_allclose(
        j2.evaluate((0.0, 7_000_000.0, 1_000_000.0))[0], 0.0, atol=1e-14
    )


def _j2_closed_form(p: np.ndarray, mu: float, radius: float, j2: float) -> np.ndarray:
    """Textbook central + J2 acceleration (J2 axis = inertial z)."""
    x, y, z = p
    r2 = float(p @ p)
    r = np.sqrt(r2)
    k = 1.5 * j2 * mu * radius**2 / r**5
    zz = 5.0 * z * z / r2
    return -mu * p / r**3 + k * np.array((x * (zz - 1.0), y * (zz - 1.0), z * (zz - 3.0)))


def _zonal_potential(p: np.ndarray, mu: float, radius: float, coefficients) -> float:
    """U = mu/r * (1 - sum_n Jn (R/r)^n Pn(z/r)), n = 2..6."""
    r = float(np.linalg.norm(p))
    s = p[2] / r
    total = 1.0
    for k, jn in enumerate(coefficients):
        n = k + 2
        total -= jn * (radius / r) ** n * np.polynomial.legendre.legval(s, [0.0] * n + [1.0])
    return mu / r * total


OFF_EQUATOR = (
    np.array((7_000_000.0, 0.0, 1_000_000.0)),
    np.array((4_000_000.0, 3_000_000.0, 4_500_000.0)),
    np.array((-2_500_000.0, 1_500_000.0, -6_000_000.0)),
)


def test_j2_matches_closed_form_off_the_equator() -> None:
    # Regression test: the latitude term once had the wrong sign, which only
    # shows away from the equator (z != 0).
    from GenADCS.backends.python.environment.gravity.j_gravity_model import (
        EARTH_MU_M3_S2,
        EARTH_REFERENCE_RADIUS_M,
        EARTH_ZONAL_COEFFICIENTS,
        gravity_acceleration_inertial_m_s2,
    )

    j2 = EARTH_ZONAL_COEFFICIENTS[0]
    coefficients = np.array((j2, 0.0, 0.0, 0.0, 0.0))
    for p in OFF_EQUATOR:
        a = gravity_acceleration_inertial_m_s2(
            p, 0.0, EARTH_MU_M3_S2, EARTH_REFERENCE_RADIUS_M, coefficients
        )
        np.testing.assert_allclose(
            a, _j2_closed_form(p, EARTH_MU_M3_S2, EARTH_REFERENCE_RADIUS_M, j2),
            rtol=0.0, atol=1e-12,
        )


def test_zonal_acceleration_is_the_gradient_of_the_potential() -> None:
    from GenADCS.backends.python.environment.gravity.j_gravity_model import (
        EARTH_MU_M3_S2,
        EARTH_REFERENCE_RADIUS_M,
        EARTH_ZONAL_COEFFICIENTS,
        gravity_acceleration_inertial_m_s2,
    )

    h = 1.0  # m, central differences
    for p in OFF_EQUATOR:
        gradient = np.array([
            (_zonal_potential(p + h * e, EARTH_MU_M3_S2, EARTH_REFERENCE_RADIUS_M,
                              EARTH_ZONAL_COEFFICIENTS)
             - _zonal_potential(p - h * e, EARTH_MU_M3_S2, EARTH_REFERENCE_RADIUS_M,
                                EARTH_ZONAL_COEFFICIENTS)) / (2.0 * h)
            for e in np.eye(3)
        ])
        a = gravity_acceleration_inertial_m_s2(
            p, 0.0, EARTH_MU_M3_S2, EARTH_REFERENCE_RADIUS_M,
            np.asarray(EARTH_ZONAL_COEFFICIENTS, dtype=float),
        )
        np.testing.assert_allclose(a, gradient, rtol=0.0, atol=1e-7)
