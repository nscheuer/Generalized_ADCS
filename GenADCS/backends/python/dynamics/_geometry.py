"""Convert SatelliteGeometry panels to arrays for compiled dynamics kernels.

Panel orientation is extrinsic XYZ (roll, pitch, yaw), applied as Rz @ Ry @ Rx
to the local +Z face normal. Panel position is its body-frame centroid.
"""

import numpy as np

from GenADCS.backends.python.hardware.satellite_geometry import SatelliteGeometry


def prepare_panels(
    geometry: SatelliteGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(geometry.panels)
    normals = np.empty((count, 3), dtype=np.float64)
    arms = np.empty((count, 3), dtype=np.float64)
    areas = np.empty(count, dtype=np.float64)
    coefficients = np.empty((count, 4), dtype=np.float64)
    center = np.asarray(geometry.center_of_mass, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("center_of_mass must be a finite 3-vector")
    for i, panel in enumerate(geometry.panels):
        angles = np.asarray(panel.orientation, dtype=np.float64)
        position = np.asarray(panel.position, dtype=np.float64)
        dimensions = np.asarray(panel.dimensions, dtype=np.float64)
        optical = np.asarray((panel.eta_s, panel.eta_d, panel.eta_a), dtype=np.float64)
        if angles.shape != (3,) or position.shape != (3,) or dimensions.shape != (2,):
            raise ValueError(
                "panel orientation, position, and dimensions must have lengths 3, 3, and 2"
            )
        if not (
            np.all(np.isfinite(angles))
            and np.all(np.isfinite(position))
            and np.all(np.isfinite(dimensions))
            and np.all(np.isfinite(optical))
            and np.isfinite(panel.cd)
        ):
            raise ValueError("panel values must be finite")
        if np.any(dimensions <= 0.0) or panel.cd < 0.0:
            raise ValueError("panel dimensions must be positive and cd nonnegative")
        if (
            np.any(optical < 0.0)
            or np.any(optical > 1.0)
            or not np.isclose(np.sum(optical), 1.0)
        ):
            raise ValueError(
                "panel optical coefficients must be in [0, 1] and sum to one"
            )
        roll, pitch, yaw = angles
        normals[i] = (
            np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll),
            np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll),
            np.cos(pitch) * np.cos(roll),
        )
        arms[i] = position - center
        areas[i] = dimensions[0] * dimensions[1]
        coefficients[i] = (panel.cd, *optical)
    return tuple(
        np.ascontiguousarray(array) for array in (normals, arms, areas, coefficients)
    )
