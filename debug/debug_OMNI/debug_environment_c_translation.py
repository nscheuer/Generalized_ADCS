"""Parity checks for native and adapted C++ environmental models."""
from __future__ import annotations

import numpy as np

from GenADCS.backends.cpp.environment import (
    BODY_EARTH, DE421 as CDE421, Earth as CEarth, Environment as CEnvironment,
    IGRF14 as CIGRF14, J2 as CJ2, JGravityModel as CJGravityModel,
    NRLMSISE00 as CNRLMSISE00, co_rotating_atmosphere_velocity as c_velocity,
)
from GenADCS.backends.python.environment import (
    DE421, Earth, Environment, IGRF14, J2, JGravityModel, NRLMSISE00,
)
from GenADCS.backends.python.environment.atmosphere import co_rotating_atmosphere_velocity


def close(actual, expected, label: str) -> None:
    if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
        raise AssertionError(f"{label} differs:\nC++: {actual}\nPython: {expected}")


def main() -> None:
    c_earth, py_earth = CEarth(), Earth()
    if (c_earth.name, c_earth.gravitational_parameter, c_earth.reference_radius) != (
            py_earth.name, py_earth.gravitational_parameter, py_earth.reference_radius):
        raise AssertionError("Earth constants differ")
    if CEnvironment().planetary_body.name != Environment().planetary_body.name:
        raise AssertionError("Environment defaults differ")

    position = np.array([6_900_000.0, -400_000.0, 1_200_000.0])
    coefficients = [1.08262668e-3, -2.5324105e-6, -1.6198976e-6, -2.2729608e-7, 5.406812e-7]
    for c_model, py_model, label in ((CJGravityModel(), JGravityModel(), "J2-J6 gravity"),
                                     (CJ2(), J2(), "J2 gravity")):
        close(c_model.evaluate(position, 2451545.0), py_model.evaluate(position, 2451545.0), label)
    close(CJGravityModel([2, 4], coefficients).evaluate(position),
          JGravityModel([2, 4], coefficients).evaluate(position), "selected gravity terms")
    if CJ2.degree != 2 or CJ2().degree != 2:
        raise AssertionError("J2 degree differs")

    close(c_velocity(position, 2451545.0), co_rotating_atmosphere_velocity(position, 2451545.0), "atmosphere velocity")
    close(CNRLMSISE00.density_geodetic(180, 43_200.0, 400.0, 45.0, 10.0),
          NRLMSISE00.density_geodetic(180, 43_200.0, 400.0, 45.0, 10.0), "NRLMSISE geodetic density")

    epoch = 2459000.5
    close(CIGRF14().field_inertial_tesla(position, epoch),
          IGRF14().field_inertial_tesla(position, epoch), "IGRF14 field")
    c_ephemeris, py_ephemeris = CDE421(), DE421()
    close(c_ephemeris.position(BODY_EARTH, epoch), py_ephemeris.position(BODY_EARTH, epoch), "DE421 Earth position")
    close(c_ephemeris.velocity_eci("sun", epoch), py_ephemeris.velocity_eci("sun", epoch), "DE421 Sun velocity")
    print("C++ environmental models match the Python reference for tested behavior.")


if __name__ == "__main__":
    main()
