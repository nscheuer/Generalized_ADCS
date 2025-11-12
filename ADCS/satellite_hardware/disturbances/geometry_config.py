__all__ = ["GeometryFace", "GeometryConfig"]

import numpy as np
from typing import List, Dict, Any

class GeometryFace:
    def __init__(self, area: float, centroid: np.ndarray, normal: np.ndarray, eta_s: float = 0, eta_d: float = 0, eta_a: float = 0, CD: float = 0):
        self.area = area
        self.centroid = centroid
        self.normal = normal
        self.eta_s = eta_s
        self.eta_d = eta_d
        self.eta_a = eta_a
        self.CD = CD

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeometryFace":
        """
        Create a GeometryFace from a dictionary.

        Accepts either 'CD' or 'CD' for drag coefficient, and defaults
        unspecified eta/CD values to 0.
        """
        return cls(
            area=data["area"],
            centroid=np.array(data["centroid"]),
            normal=np.array(data["normal"]),
            eta_s=data.get("eta_s", 0),
            eta_d=data.get("eta_d", 0),
            eta_a=data.get("eta_a", 0),
            CD=data.get("CD", data.get("CD", 0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this GeometryFace to a dictionary representation.
        """
        return {
            "area": self.area,
            "centroid": np.array(self.centroid),
            "normal": np.array(self.normal),
            "eta_s": self.eta_s,
            "eta_d": self.eta_d,
            "eta_a": self.eta_a,
            "CD": self.CD,
        }
    


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

    def __init__(self, geometry_faces: List[GeometryFace]) -> None:
        r"""
        Initialize a :class:`GeometryConfig` instance.

        Parameters
        ----------
        geometry : list[dict[str, any]]
            List of dictionaries describing each surface element.
            See :ref:`Structure of a Geometry Entry <geometry_entry>` above.
        """ 
        self.faces = geometry_faces
        self.params: List[Dict[str, Any]] = []
        for f in self.faces:
            # prefer 'CD' (lowercase) in params for back-compat
            cd_val = getattr(f, "CD", getattr(f, "CD", 0))
            self.params.append({
                "area": float(f.area),
                "centroid": np.array(f.centroid, dtype=float),
                "normal": np.array(f.normal, dtype=float),
                "eta_s": float(getattr(f, "eta_s", 0)),
                "eta_d": float(getattr(f, "eta_d", 0)),
                "eta_a": float(getattr(f, "eta_a", 0)),
                "CD": float(cd_val),
            })