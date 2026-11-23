Environment and Geometry
========================

Environment
-----------

.. code-block:: python
   :caption: environment_and_geometry.py

   import ADCS

   environment = ADCS.Environment(
       magnetic_field=ADCS.IGRF14(),
       atmosphere=ADCS.SMADAtmosphere(),
       ephemeris=ADCS.DE421(),
   )

Geometry
--------

.. code-block:: python

   import numpy as np

   # A 10 × 30 × 10 cm 3U body; body y is the long axis.
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
