from __future__ import annotations 
__all__ = ["SRP_Disturbance"]

import numpy as np
from typing import TYPE_CHECKING
from ADCS.satellite_hardware.disturbances.disturbance import Disturbance
from ADCS.satellite_hardware.disturbances.geometry_config import GeometryConfig
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize, normed_vec_jac, normed_vec_hess
from ADCS.orbits.universal_constants import EarthConstants


if TYPE_CHECKING:
    from ADCS.satellite_hardware.satellite.satellite import Satellite


class SRP_Disturbance(Disturbance):
    def __init__(self, config: GeometryConfig):
        self.config = config
        params = self.config.params

        self.numfaces = len(params)
        self.areas = np.array([p["area"] for p in params])
        self.centroids = np.vstack([p["centroid"] for p in params])
        self.normals = np.vstack([normalize(p["normal"]) for p in params])
        self.eta_s = np.array([p["eta_s"] for p in params])
        self.eta_d = np.array([p["eta_d"] for p in params])
        self.eta_a = np.array([p["eta_a"] for p in params])

    def torque(self, sat: Satellite, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        vecs = os.get_state_vector(q0=q)
        
        S_B = vecs["s"]
        R_B = vecs["r"]

        s_body = normalize(S_B - R_B)

        cos_gamma = np.maximum(0, np.dot(self.normals, s_body))
        proj_area = self.areas*cos_gamma
        cents = self.centroids - sat.COM
        m_s = proj_area*(self.eta_a + self.eta_d)
        t_s = m_s@np.cross(cents, s_body)
        m_n = proj_area*(2*self.eta_s*cos_gamma + (2/3)*self.eta_d)
        t_n = m_n@np.cross(cents, self.normals)

        if os.is_sunlit():
            return -(EarthConstants.solar_constant/EarthConstants.c)*(t_s + t_n)
        else:
            return np.zeros((3, 1))
        
    def torque_qjav(self, sat: Satellite, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        vecs = os.get_state_vector(q0=q)

        S_B = vecs["s"]
        R_B = vecs["r"]

        s_body = normalize(S_B - R_B)
        ds_body__dq = normed_vec_jac(S_B - R_B, vecs["ds"] - vecs["dr"])

        # --- Compute cos(gamma) and projection area ---
        cos_gamma = np.dot(self.normals, s_body)

        # Clamp negatives to zero (no contribution when light is from behind)
        for i in range(len(cos_gamma)):
            if cos_gamma[i] < 0:
                cos_gamma[i] = 0.0

        proj_area = self.areas * cos_gamma
        cents = self.centroids - sat.COM

        # --- Derivatives ---
        dcos_gamma__dq = np.zeros_like(ds_body__dq @ self.normals.T)
        for i in range(len(cos_gamma)):
            if cos_gamma[i] > 0:
                dcos_gamma__dq[i] = (ds_body__dq @ self.normals.T)[i]
            else:
                dcos_gamma__dq[i] = 0.0

        dproj_area__dq = self.areas * dcos_gamma__dq

        # --- Specular + diffuse absorbed component ---
        m_s = proj_area * (self.eta_a + self.eta_d)
        dm_s__dq = dproj_area__dq * (self.eta_a + self.eta_d)
        dt_s__dq = dm_s__dq @ np.cross(cents, s_body) + np.cross(m_s @ cents, ds_body__dq)

        # --- Specular + diffuse reflected component ---
        dm_n__dq = (
            dproj_area__dq * (2 * self.eta_s * cos_gamma + (2 / 3) * self.eta_d)
            + proj_area * (2 * self.eta_s * dcos_gamma__dq)
        )
        dt_n__dq = dm_n__dq @ np.cross(cents, self.normals)

        # --- Sunlight condition ---
        if os.is_sunlit():
            return -(EarthConstants.solar_constant / EarthConstants.c) * (dt_s__dq + dt_n__dq)
        else:
            return np.zeros((3, 1))


    def torque_qqhess(self, sat: Satellite, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        vecs = os.get_state_vector(q0=q)

        S_B = vecs["s"]
        R_B = vecs["r"]

        s_body = normalize(S_B - R_B)
        ds_body__dq = normed_vec_jac(S_B - R_B, vecs["ds"] - vecs["dr"])
        dds_body__dqdq = normed_vec_hess(S_B - R_B, vecs["ds"] - vecs["dr"], vecs["dds"] - vecs["ddr"])

        # --- Cosine of incidence angle ---
        cos_gamma = np.dot(self.normals, s_body)

        # Clamp negative cosines to zero
        for i in range(len(cos_gamma)):
            if cos_gamma[i] < 0.0:
                cos_gamma[i] = 0.0

        proj_area = self.areas * cos_gamma
        cents = self.centroids - sat.COM

        # --- Derivatives initialization ---
        dcos_gamma__dq = np.zeros_like(ds_body__dq @ self.normals.T)
        ddcos_gamma__dqdq = np.zeros_like(dds_body__dqdq @ self.normals.T)

        # Only compute derivatives where cos_gamma > 0
        normals_term_1 = ds_body__dq @ self.normals.T
        normals_term_2 = dds_body__dqdq @ self.normals.T
        for i in range(len(cos_gamma)):
            if cos_gamma[i] > 0.0:
                dcos_gamma__dq[i] = normals_term_1[i]
                ddcos_gamma__dqdq[i] = normals_term_2[i]

        # --- Projected area derivatives ---
        dproj_area__dq = self.areas * dcos_gamma__dq
        ddproj_area__dqdq = self.areas * ddcos_gamma__dqdq

        # --- Scalar terms (absorbed + diffuse) ---
        m_s = proj_area * (self.eta_a + self.eta_d)
        dm_s__dq = dproj_area__dq * (self.eta_a + self.eta_d)
        ddm_s__dqdq = ddproj_area__dqdq * (self.eta_a + self.eta_d)

        dt_s__dq = dm_s__dq @ np.cross(cents, s_body) + np.cross(m_s @ cents, ds_body__dq)
        tmp = np.cross(np.expand_dims(dm_s__dq @ cents, 0), np.expand_dims(ds_body__dq, 1))
        ddt_s__dqdq = (
            ddm_s__dqdq @ np.cross(cents, s_body)
            + tmp
            + np.transpose(tmp, (1, 0, 2))
            + np.cross(m_s @ cents, dds_body__dqdq)
        )

        # --- Specular + diffuse reflection components ---
        dm_n__dq = (
            dproj_area__dq * (2 * self.eta_s * cos_gamma + (2 / 3) * self.eta_d)
            + proj_area * (2 * self.eta_s * dcos_gamma__dq)
        )

        tmp2 = np.expand_dims(dproj_area__dq, 0) * np.expand_dims((2 * self.eta_s * dcos_gamma__dq), 1)

        ddm_n__dqdq = (
            ddproj_area__dqdq * (2 * self.eta_s * cos_gamma + (2 / 3) * self.eta_d)
            + tmp2
            + np.transpose(tmp2, (1, 0, 2))
            + proj_area * (2 * self.eta_s * ddcos_gamma__dqdq)
        )

        ddt_n__dqdq = ddm_n__dqdq @ np.cross(cents, self.normals)

        # --- Final torque Hessian ---
        if os.is_sunlit():
            return -(EarthConstants.solar_constant / EarthConstants.c) * (ddt_s__dqdq + ddt_n__dqdq)
        else:
            return np.zeros_like(ddt_s__dqdq)
