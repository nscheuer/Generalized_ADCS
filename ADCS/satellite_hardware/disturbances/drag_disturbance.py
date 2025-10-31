from __future__ import annotations 
__all__ = ["Drag_Disturbance"]

import numpy as np
from typing import TYPE_CHECKING
from ADCS.satellite_hardware.disturbances.disturbance import Disturbance
from ADCS.satellite_hardware.disturbances.geometry_config import GeometryConfig
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize

if TYPE_CHECKING:
    from ADCS.satellite_hardware.satellite.satellite import Satellite

class Drag_Disturbance(Disturbance):
    def __init__(self, config: GeometryConfig):
        self.config = config
        params = self.config.params

        self.numfaces = len(params)
        self.areas = np.array([p["area"] for p in params])
        self.centroids = np.vstack([p["centroid"] for p in params])
        self.normals = np.vstack([normalize(p["normal"]) for p in params])
        self.CDs = np.array([p["cd"] for p in params])

    def torque(self, sat: Satellite, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        vecs = os.get_state_vector(q0=q)

        V_B = vecs["v"]
        rho = vecs["rho"]

        cos_alpha = np.maximum(0, np.dot(self.normals, V_B))
        F = self.CDs*self.areas*cos_alpha
        cents = self.centroids - sat.COM
        ct = 0.5*rho
        return -ct*np.cross(F@cents, V_B)
    
    def torque_qjac(self, sat: Satellite, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        vecs = os.get_state_vector(q0=q)

        V_B = vecs["v"]
        rho = vecs["rho"]

        dv_body__dq = vecs["dv"]
        ddv_body__dqdq = vecs["ddv"]

        cos_alpha = np.maximum(0, np.dot(self.normals, V_B))
        F = self.CDs * self.areas * cos_alpha
        cents = self.centroids - sat.COM

        # Initialize derivative arrays
        dcos_alpha__dq = np.zeros_like(dv_body__dq @ self.normals.T)

        # Explicit conditional logic instead of inline boolean mask
        normals_term = dv_body__dq @ self.normals.T
        for i in range(len(cos_alpha)):
            if cos_alpha[i] > 0:
                dcos_alpha__dq[i] = normals_term[i]
            else:
                dcos_alpha__dq[i] = np.zeros_like(normals_term[i])

        dF__dq = dcos_alpha__dq * self.CDs * self.areas

        ct = 0.5 * rho
        return -ct * (np.cross(dF__dq @ cents, V_B) + np.cross(F @ cents, dv_body__dq)) * self.active

    def torque_qqhess(self, sat, vecs):
        V_B = vecs["v"]
        rho = vecs["rho"]
        dv_body__dq = vecs["dv"]
        ddv_body__dqdq = vecs["ddv"]

        cos_alpha = np.maximum(0, np.dot(self.normals, V_B))
        F = self.CDs * self.areas * cos_alpha
        cents = self.centroids - sat.COM

        dcos_alpha__dq = np.zeros_like(dv_body__dq @ self.normals.T)
        ddcos_alpha__dqdq = np.zeros_like(ddv_body__dqdq @ self.normals.T)

        normals_term_1 = dv_body__dq @ self.normals.T
        normals_term_2 = ddv_body__dqdq @ self.normals.T

        for i in range(len(cos_alpha)):
            if cos_alpha[i] > 0:
                dcos_alpha__dq[i] = normals_term_1[i]
                ddcos_alpha__dqdq[i] = normals_term_2[i]
            else:
                dcos_alpha__dq[i] = np.zeros_like(normals_term_1[i])
                ddcos_alpha__dqdq[i] = np.zeros_like(normals_term_2[i])

        dF__dq = dcos_alpha__dq * self.CDs * self.areas
        ddF__dqdq = ddcos_alpha__dqdq * self.CDs * self.areas
        ct = 0.5 * rho

        tmp = np.cross(np.expand_dims(dF__dq @ cents, 0), np.expand_dims(dv_body__dq, 1))

        return -ct * (
            np.cross(ddF__dqdq @ cents, V_B)
            + tmp
            + np.transpose(tmp, (1, 0, 2))
            + np.cross(F @ cents, ddv_body__dqdq)
        )


