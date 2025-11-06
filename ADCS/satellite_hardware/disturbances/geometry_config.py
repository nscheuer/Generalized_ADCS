__all__ = ["GeometryConfig"]

import numpy as np
from typing import List, Dict, Any

class GeometryConfig:
    r"""
    **Satellite Geometry Configuration**

    The :class:`GeometryConfig` class defines the geometric and aerodynamic
    properties of the satellite’s external surfaces.  
    This configuration is used by disturbance and actuator models such as
    :class:`~ADCS.satellite_hardware.disturbances.drag_disturbance.Drag_Disturbance`
    to compute forces and torques.

    Each entry in the geometry configuration represents one **surface element**
    (e.g., a panel or face) with attributes describing its physical and
    aerodynamic properties.

    **Structure of a Geometry Entry**

    Each geometry element is expected to contain the following ordered parameters:

    .. list-table::
       :header-rows: 1
       :widths: 10 40

       * - Key
         - Description
       * - ``index``
         - Unique integer identifier for the surface element.
       * - ``area``
         - Surface area :math:`A_i` [m²].
       * - ``centroid``
         - Centroid position vector :math:`\mathbf{r}_i` in body frame [m].
       * - ``normal``
         - Surface normal unit vector :math:`\mathbf{n}_i` in body frame.
       * - ``eta_s``
         - Specular reflection coefficient (0–1).
       * - ``eta_d``
         - Diffuse reflection coefficient (0–1).
       * - ``eta_a``
         - Absorptivity coefficient (0–1).
       * - ``CD``
         - Drag coefficient :math:`C_{D,i}` of the surface.

    Parameters
    ----------
    geometry : list[dict[str, any]]
        List of dictionaries defining geometry parameters for all satellite surfaces.
        Each entry must include the keys listed above.
    """

    def __init__(self, geometry: List[Dict[str, Any]]) -> None:
        r"""
        Initialize a :class:`GeometryConfig` instance.

        Parameters
        ----------
        geometry : list[dict[str, any]]
            List of dictionaries describing each surface element.
            See :ref:`Structure of a Geometry Entry <geometry_entry>` above.
        """
        self.params = geometry

    def add_geometry(self, geometry: List[Dict[str, Any]]) -> None:
        r"""
        Add new surface geometry elements to the configuration.

        This method appends additional surface definitions to the existing
        configuration.  
        Each input face must contain **8 elements** in the order:

        .. code-block:: python

            [index, area, centroid, normal, eta_s, eta_d, eta_a, CD]

        The new surface is stored internally as a dictionary:

        .. code-block:: python

            {
                "index": int,
                "area": float,
                "centroid": np.ndarray,
                "normal": np.ndarray,
                "eta_s": float,
                "eta_d": float,
                "eta_a": float,
                "cd": float
            }

        Parameters
        ----------
        geometry : list[list[any]]
            List of surface definitions, each containing exactly 8 elements
            (index, area, centroid, normal, eta_s, eta_d, eta_a, CD).

        Raises
        ------
        ValueError
            If any geometry element does not contain exactly 8 parameters.
        """
        for face in geometry:
            if len(face) != 8:
                raise ValueError(
                    "Each geometry element must have 8 values: "
                    "[index, area, centroid, normal, eta_s, eta_d, eta_a, CD]"
                )

            face_dict = {
                "index": face[0],
                "area": face[1],
                "centroid": np.array(face[2]),
                "normal": np.array(face[3]),
                "eta_s": face[4],
                "eta_d": face[5],
                "eta_a": face[6],
                "cd": face[7],
            }
            self.params.append(face_dict)
