"""Check that drag opposes velocity and changes with satellite orientation."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

import ADCS
from ADCS.helpers.math_helpers import rot_mat
from ADCS.orbits.universal_constants import EarthConstants


def main() -> None:
    environment = ADCS.Environment(atmosphere=ADCS.SMADAtmosphere())
    radius = EarthConstants.R_e + 250.0
    os = ADCS.Orbital_State.from_position(
        J2000=0.22,
        R=np.array([radius, 0.0, 0.0]),
        V=np.array([0.0, np.sqrt(EarthConstants.mu_e / radius), 0.0]),
        environment=environment,
    )

    # A 4 m by 20 cm by 20 cm box: broadside area is 0.8 m²,
    # while the end-on area is 0.04 m². Centered faces isolate force direction.
    length, width, cd = 4.0, 0.2, 2.2
    faces = [
        ADCS.disturbances.GeometryFace(
            area=width**2,
            centroid=np.zeros(3),
            normal=np.array([sign, 0.0, 0.0]),
            CD=cd,
        )
        for sign in (-1.0, 1.0)
    ]
    faces.extend(
        ADCS.disturbances.GeometryFace(
            area=length * width,
            centroid=np.zeros(3),
            normal=normal,
            CD=cd,
        )
        for normal in (
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, -1.0]),
            np.array([0.0, 0.0, 1.0]),
        )
    )
    drag = ADCS.disturbances.Drag_Disturbance(
        ADCS.disturbances.GeometryConfig(faces)
    )
    satellite = ADCS.Satellite(
        mass=10.0,
        J_0=np.diag([0.12, 13.35, 13.35]),
        disturbances=[drag],
    )

    # Orbit velocity is +ECI Y. Identity presents a broad side; this 90°
    # rotation points the long body axis along velocity (end-on).
    attitudes = {
        "broadside": np.array([1.0, 0.0, 0.0, 0.0]),
        "end-on": np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]),
    }
    velocity_eci = os.V * 1000.0
    print(f"Orbital speed: {np.linalg.norm(velocity_eci):.1f} m/s")
    for name, quaternion in attitudes.items():
        state = ADCS.State.from_array([0.0, 0.0, 0.0, *quaternion])
        force_body = satellite.dist_forces(state, os)
        force_eci = rot_mat(quaternion) @ force_body
        projected_area = np.linalg.norm(force_eci) / (
            0.5 * os.rho * cd * np.linalg.norm(velocity_eci) ** 2
        )
        direction_check = np.dot(force_eci, velocity_eci)
        print(
            f"{name:9s}: F_ECI={force_eci} N, "
            f"projected area={projected_area:.3f} m², "
            f"F·V={direction_check:.3e} (must be negative)"
        )


if __name__ == "__main__":
    main()
