"""Create an orbital state and print its Keplerian elements."""

import numpy as np

import ADCS


def main() -> None:
    orbit = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 1.0]),
    )
    semi_major_axis, eccentricity, inclination, raan, argp, true_anomaly = orbit.to_keplerian()

    print(f"Semi-major axis: {semi_major_axis:.2f} km")
    print(f"Eccentricity: {eccentricity:.6f}")
    print(f"Inclination: {inclination:.2f} deg")
    print(f"RAAN: {raan:.2f} deg")
    print(f"Argument of periapsis: {argp:.2f} deg")
    print(f"True anomaly: {true_anomaly:.2f} deg")


if __name__ == "__main__":
    main()
