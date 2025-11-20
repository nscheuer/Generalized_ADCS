from __future__ import annotations 
__all__ = ["GG_Disturbance"]

import numpy as np
from typing import TYPE_CHECKING
from ADCS.satellite_hardware.disturbances.disturbance import Disturbance
from ADCS.satellite_hardware.disturbances.geometry_config import GeometryConfig
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize, normed_vec_jac, normed_vec_hess, norm, vec_norm_jac, vec_norm_hess
from ADCS.orbits.universal_constants import EarthConstants

if TYPE_CHECKING:
    from ADCS.satellite_hardware.satellite.satellite import Satellite

class GG_Disturbance(Disturbance):
    def __init__(self):
        super().__init__()

    def torque(self, sat: Satellite, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        vecs = os.get_state_vector(x=x)

        R_B = vecs["r"]
        r_body_hat = normalize(R_B)
        nadir_vec = -r_body_hat

        const_term = 3.0*EarthConstants.mu_e/(norm(R_B)**3.0)
        return const_term*np.cross(nadir_vec, nadir_vec@sat.J_0)
    
    def torque_qvac(self, sat: Satellite, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        vecs = os.get_state_vector(x=x)

        R_B = vecs["r"]
        r_body_hat = normalize(R_B)
        dr_body_hat__dq = normed_vec_jac(R_B, vecs["dr"])
        nadir_vec = -r_body_hat
        dnadir_vec__dq = -dr_body_hat__dq

        const_term = 3.0*EarthConstants.mu_e/(norm(R_B)**3.0)

        dc__dq = -9.0*EarthConstants.mu_e*vec_norm_jac(R_B, vecs["dr"])/(norm(R_B)**4.0)
        dv__dq = np.cross(dnadir_vec__dq, nadir_vec@sat.J_0) + np.cross(nadir_vec, dnadir_vec__dq@sat.J)
        vec_term = np.cross(nadir_vec, nadir_vec@sat.J_0)

        return np.outer(dc__dq, vec_term) + const_term*dv__dq

    def torque__qqhess(self, sat: Satellite, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        vecs = os.get_state_vector(x=x)

        R_B = vecs["r"]
        r_body_hat = normalize(R_B)
        dr_body_hat__dq = normed_vec_jac(R_B, vecs["dr"])
        ddr_body_hat__dq = normed_vec_hess(R_B, vecs["dr"], vecs["ddr"])
        nadir_vec = -r_body_hat
        dnadir_vec__dq = -dr_body_hat__dq
        ddnadir_vec__dqdq = -ddr_body_hat__dq

        const_term = 3.0*EarthConstants.mu_e/(norm(R_B)**3.0)
        tmp = np.cross(np.expand_dims(dnadir_vec__dq, 1), np.expand_dims(dnadir_vec__dq@sat.J_0, 0))

        dc__dq = -9.0*EarthConstants.mu_e*vec_norm_jac(R_B, vecs["dr"])/(norm(R_B)**4.0)
        ddc__dqdq = -9.0*EarthConstants.mu_e*vec_norm_hess(R_B, vecs["dr"], vecs["ddr"])/(norm(R_B)**4.0) - 4.0*np.outer(vec_norm_jac(R_B, vecs["dr"]), vec_norm_jac(R_B, vecs["dr"]))/(norm(R_B)**5.0)
        dv__dq = np.cross(dnadir_vec__dq, nadir_vec@sat.J_0) + np.cross(nadir_vec, dnadir_vec__dq@sat.J)
        ddv__dqdq = np.cross(ddnadir_vec__dqdq, nadir_vec@sat.J_0) + tmp + np.transpose(tmp, (1, 0, 2)) + np.cross(nadir_vec, ddnadir_vec__dqdq@sat.J_0)

        vec_term = np.cross(nadir_vec, nadir_vec@sat.J_0)
        tmp2 = np.multiply.outer(dc__dq, dv__dq)

        return np.multiply.outer(ddc__dqdq, vec_term) + tmp2 + np.transpose(tmp2, (1, 0, 2)) + const_term*ddv__dqdq