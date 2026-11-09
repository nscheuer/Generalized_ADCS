"""Select environment models and describe a 3U spacecraft's faces."""

import ADCS
import numpy as np

environment = ADCS.Environment(
    magnetic_field=ADCS.IGRF14(),
    atmosphere=ADCS.SMADAtmosphere(),
    ephemeris=ADCS.DE421(),
)

size = np.array([0.1, 0.3, 0.1])
faces = []
for axis in range(3):
    for sign in (-1.0, 1.0):
        normal = sign * np.eye(3)[axis]
        faces.append(ADCS.disturbances.GeometryFace(
            area=np.prod(np.delete(size, axis)),
            centroid=0.5 * size[axis] * normal,
            normal=normal,
            CD=2.2,
        ))
geometry = ADCS.disturbances.GeometryConfig(faces)
