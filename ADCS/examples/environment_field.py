"""Evaluate the default IGRF-14 magnetic field at one ECEF position."""

from datetime import datetime, timezone

import numpy as np

import ADCS


def main() -> None:
    environment = ADCS.Environment(magnetic_field=ADCS.IGRF14())
    position_ecef_km = np.array([6771.2, 0.0, 0.0])
    date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    field_ecef_tesla = environment.magnetic_field.field(position_ecef_km, date)

    print("Magnetic field at 400 km altitude [T]:", field_ecef_tesla)


if __name__ == "__main__":
    main()
