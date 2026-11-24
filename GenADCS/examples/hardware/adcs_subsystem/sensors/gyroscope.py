"""Attach a gyroscope to a Python ADCS subsystem and measure body rate.

Run from the repository root with
``venv/bin/python GenADCS/examples/hardware/adcs_subsystem/sensors/gyroscope.py``.
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.hardware.ADCS_subsystem import ADCSSubsystem
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors import Gyroscope


def main():
    geometry = SatelliteGeometry(
        mass=12.0,
        inertia=np.diag((0.13, 0.10, 0.05)),
        panels=[],
    )
    gyroscope = Gyroscope(
        position=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        misalignment=np.array([1.0, 0.0, 0.0, 0.0]),
        saturation_value=10.0,
        noise=np.diag((1e-8, 1e-8, 1e-8)),
        bias=np.array([1e-4, -2e-4, 1.5e-4]),
    )
    adcs = ADCSSubsystem(sensors=[gyroscope])
    satellite = Satellite(geometry=geometry, adcs=adcs)

    body_rate = np.array([0.01, -0.015, 0.02])
    measurement = gyroscope.noisy_biased_measurement(body_rate)
    print(f"Attached sensors: {len(satellite.adcs.sensors)}")
    print(f"True body rate [rad/s]: {body_rate}")
    print(f"Gyroscope reading [rad/s]: {measurement}")
    return satellite, measurement


if __name__ == "__main__":
    main()
