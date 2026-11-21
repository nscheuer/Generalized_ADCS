"""Projected area solar radiation pressure force and torque wrappers."""
import numpy as np
from .._geometry import prepare_panels
from .._dynamics import projected_area_srp_force, projected_area_srp_torque
from ADCS_Omni.ADCS_Rust.environment.ephemeris import BODY_SUN
from ADCS_Omni.ADCS_Rust.helpers.math.time import utc_jd_to_tdb_jd

AU_M = 149_597_870_700.0
SOLAR_PRESSURE_AT_1_AU_PA = 1361.0 / 299_792_458.0
SUN_RADIUS_M = 696_340_000.0


class _ProjectedAreaSRPBase:
    def __init__(self, geometry):
        self.normals_b, self.arms_b, self.areas, coefficients = prepare_panels(geometry)
        self.optical_coefficients = np.ascontiguousarray(coefficients[:, 1:4])

    @staticmethod
    def _illumination(position_i, epoch_jd, environment):
        if environment.ephemeris is None:
            raise ValueError("SRP requires environment.ephemeris")
        position_i = np.asarray(position_i, dtype=np.float64)
        sun_position_i = 1000.0 * environment.ephemeris.position(BODY_SUN, utc_jd_to_tdb_jd(epoch_jd))
        illumination, _, _ = environment.ephemeris.eclipse_state(
            position_i, sun_position_i, environment.planetary_body.reference_radius, SUN_RADIUS_M)
        return position_i, sun_position_i, illumination


class ProjectedAreaSRPForceModel(_ProjectedAreaSRPBase):
    def evaluate(self, position_i, q_b_from_i, epoch_jd, environment):
        p, sun, illumination = self._illumination(position_i, epoch_jd, environment)
        return projected_area_srp_force(p, sun, np.asarray(q_b_from_i, dtype=np.float64),
                                        self.normals_b, self.areas, self.optical_coefficients, illumination)


class ProjectedAreaSRPTorqueModel(_ProjectedAreaSRPBase):
    def evaluate(self, position_i, q_b_from_i, epoch_jd, environment):
        p, sun, illumination = self._illumination(position_i, epoch_jd, environment)
        return projected_area_srp_torque(p, sun, np.asarray(q_b_from_i, dtype=np.float64),
                                         self.normals_b, self.arms_b, self.areas,
                                         self.optical_coefficients, illumination)

__all__ = ["ProjectedAreaSRPForceModel", "ProjectedAreaSRPTorqueModel", "projected_area_srp_force", "projected_area_srp_torque"]
