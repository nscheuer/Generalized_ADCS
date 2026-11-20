"""Separate projected-area solar radiation pressure force and torque models."""

import numpy as np
from numba import njit

from ADCS_Omni.environment.ephemeris import BODY_SUN
from ADCS_Omni.helpers.math.frames.frames import body_to_inertial, inertial_to_body
from ADCS_Omni.helpers.math.time.conversions import utc_jd_to_tdb_jd
from .._geometry import prepare_panels

AU_M = 149_597_870_700.0
SOLAR_PRESSURE_AT_1_AU_PA = 1361.0 / 299_792_458.0
SUN_RADIUS_M = 696_340_000.0


@njit(cache=True)
def _panel_srp_force_b(position_i, sun_position_i, q_b_from_i, normals_b, areas,
                       optical_coefficients, illumination):
    sun_direction_i = sun_position_i - position_i
    distance = np.sqrt(np.dot(sun_direction_i, sun_direction_i))
    if distance == 0.0:
        raise ValueError("spacecraft and Sun positions coincide")
    sun_direction_b = inertial_to_body(sun_direction_i / distance, q_b_from_i)
    pressure = SOLAR_PRESSURE_AT_1_AU_PA * (AU_M / distance) ** 2 * illumination
    force_b = np.zeros(3)
    for i in range(len(areas)):
        cosine = np.dot(normals_b[i], sun_direction_b)
        if cosine > 0.0:
            specular, diffuse, absorptive = optical_coefficients[i]
            force_b -= pressure * areas[i] * cosine * (
                (absorptive + diffuse) * sun_direction_b
                + (2.0 * specular * cosine + (2.0 / 3.0) * diffuse) * normals_b[i]
            )
    return force_b


@njit(cache=True)
def projected_area_srp_force(position_i, sun_position_i, q_b_from_i, normals_b,
                             areas, optical_coefficients, illumination=1.0):
    """Return solar radiation pressure force in inertial coordinates [N]."""
    force_b = _panel_srp_force_b(
        position_i, sun_position_i, q_b_from_i, normals_b, areas,
        optical_coefficients, illumination,
    )
    return body_to_inertial(force_b, q_b_from_i)


@njit(cache=True)
def projected_area_srp_torque(position_i, sun_position_i, q_b_from_i, normals_b,
                              arms_b, areas, optical_coefficients, illumination=1.0):
    """Return solar radiation pressure torque about the center of mass in body coordinates [N m]."""
    sun_direction_i = sun_position_i - position_i
    distance = np.sqrt(np.dot(sun_direction_i, sun_direction_i))
    if distance == 0.0:
        raise ValueError("spacecraft and Sun positions coincide")
    sun_direction_b = inertial_to_body(sun_direction_i / distance, q_b_from_i)
    pressure = SOLAR_PRESSURE_AT_1_AU_PA * (AU_M / distance) ** 2 * illumination
    torque_b = np.zeros(3)
    for i in range(len(areas)):
        cosine = np.dot(normals_b[i], sun_direction_b)
        if cosine > 0.0:
            specular, diffuse, absorptive = optical_coefficients[i]
            force_b = -pressure * areas[i] * cosine * (
                (absorptive + diffuse) * sun_direction_b
                + (2.0 * specular * cosine + (2.0 / 3.0) * diffuse) * normals_b[i]
            )
            torque_b += np.cross(arms_b[i], force_b)
    return torque_b


class _ProjectedAreaSRPBase:
    def __init__(self, geometry):
        self.normals_b, self.arms_b, self.areas, coefficients = prepare_panels(geometry)
        self.optical_coefficients = np.ascontiguousarray(coefficients[:, 1:4])

    @staticmethod
    def _illumination(position_i, epoch_jd, environment):
        if environment.ephemeris is None:
            raise ValueError("SRP requires environment.ephemeris")
        position_i = np.asarray(position_i, dtype=np.float64)
        # Ephemeris.position returns body-centered km; calculations use metres.
        sun_position_i = 1000.0 * environment.ephemeris.position(
            BODY_SUN, utc_jd_to_tdb_jd(epoch_jd),
        )
        illumination, _, _ = environment.ephemeris.eclipse_state(
            position_i, sun_position_i,
            environment.planetary_body.reference_radius, SUN_RADIUS_M,
        )
        return position_i, sun_position_i, illumination


class ProjectedAreaSRPForceModel(_ProjectedAreaSRPBase):
    def evaluate(self, position_i, q_b_from_i, epoch_jd, environment):
        position_i, sun_position_i, illumination = self._illumination(position_i, epoch_jd, environment)
        return projected_area_srp_force(
            position_i, sun_position_i, np.asarray(q_b_from_i, dtype=np.float64),
            self.normals_b, self.areas, self.optical_coefficients, illumination,
        )


class ProjectedAreaSRPTorqueModel(_ProjectedAreaSRPBase):
    def evaluate(self, position_i, q_b_from_i, epoch_jd, environment):
        position_i, sun_position_i, illumination = self._illumination(position_i, epoch_jd, environment)
        return projected_area_srp_torque(
            position_i, sun_position_i, np.asarray(q_b_from_i, dtype=np.float64),
            self.normals_b, self.arms_b, self.areas, self.optical_coefficients,
            illumination,
        )
