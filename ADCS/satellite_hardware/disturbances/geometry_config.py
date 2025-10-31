import numpy as np
from typing import List, Dict, Any

class GeometryConfig():
    def __init__(self, geometry: List[Dict[str, Any]]) -> None:
        self.params = geometry

    def add_geometry(self, geometry: List[Dict[str, Any]]) -> None:
        for face in geometry:
            if len(face) != 8:
                raise ValueError("Each geometry element must have 7 values: [index, area, centroid, normal, eta_s, eta_d, eta_a, CD]")
            
            face_dict = {
                "index": face[0],
                "area": face[1],
                "centroid": np.array(face[2]),
                "normal": np.array(face[3]),
                "eta_s": face[4],
                "eta_d": face[5],
                "eta_a": face[6],
                "cd": face[7]
            }
            self.params.append(face_dict)