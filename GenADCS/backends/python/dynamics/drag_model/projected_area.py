"""Separate projected-area drag force and torque models."""

import numpy as np
from numba import njit

from GenADCS.backends.python.helpers.math.frames.frames import body_to_inertial, inertial_to_body
from .._geometry import prepare_panels


@njit(cache=True)
def projected_area_drag_force(velocity_relative_i, q_b_from_i, density, normals_b, areas, drag_coefficients):
    """Return aerodynamic drag force in inertial coordinates [N]."""
    if len(normals_b) != len(areas) or len(drag_coefficients) != len(areas):
        raise ValueError("panel arrays must have matching row counts")
    velocity_b = inertial_to_body(velocity_relative_i, q_b_from_i)
    speed = np.sqrt(np.dot(velocity_b, velocity_b))
    force_b = np.zeros(3)
    if speed > 0.0 and density > 0.0:
        for i in range(len(areas)):
            incidence = np.dot(normals_b[i], velocity_b) / speed
            if incidence > 0.0:
                force_b -= 0.5 * density * speed * areas[i] * drag_coefficients[i] * incidence * velocity_b
    return body_to_inertial(force_b, q_b_from_i)


@njit(cache=True)
def projected_area_drag_torque(velocity_relative_i, q_b_from_i, density, normals_b, arms_b, areas, drag_coefficients):
    """Return aerodynamic drag torque about the center of mass in body coordinates [N m]."""
    if len(normals_b) != len(areas) or len(arms_b) != len(areas) or len(drag_coefficients) != len(areas):
        raise ValueError("panel arrays must have matching row counts")
    velocity_b = inertial_to_body(velocity_relative_i, q_b_from_i)
    speed = np.sqrt(np.dot(velocity_b, velocity_b))
    torque_b = np.zeros(3)
    if speed > 0.0 and density > 0.0:
        for i in range(len(areas)):
            incidence = np.dot(normals_b[i], velocity_b) / speed
            if incidence > 0.0:
                face_force_b = -0.5 * density * speed * areas[i] * drag_coefficients[i] * incidence * velocity_b
                torque_b += np.cross(arms_b[i], face_force_b)
    return torque_b


class _ProjectedAreaDragBase:
    def __init__(self, geometry):
        self.normals_b, self.arms_b, self.areas, coefficients = prepare_panels(geometry)
        self.drag_coefficients = np.ascontiguousarray(coefficients[:, 0])

    @staticmethod
    def _flow(position_i, velocity_i, epoch_jd, environment):
        if environment.atmosphere is None:
            raise ValueError("drag requires environment.atmosphere")
        position_i = np.asarray(position_i, dtype=np.float64)
        density = environment.atmosphere.density(position_i, epoch_jd)
        atmosphere_velocity = environment.atmosphere.velocity(position_i, epoch_jd)
        return np.asarray(velocity_i, dtype=np.float64) - atmosphere_velocity, density


class ProjectedAreaDragForceModel(_ProjectedAreaDragBase):
    def evaluate(self, position_i, velocity_i, q_b_from_i, epoch_jd, environment):
        relative_velocity, density = self._flow(position_i, velocity_i, epoch_jd, environment)
        return projected_area_drag_force(
            relative_velocity, np.asarray(q_b_from_i, dtype=np.float64), density,
            self.normals_b, self.areas, self.drag_coefficients,
        )


class ProjectedAreaDragTorqueModel(_ProjectedAreaDragBase):
    def evaluate(self, position_i, velocity_i, q_b_from_i, epoch_jd, environment):
        relative_velocity, density = self._flow(position_i, velocity_i, epoch_jd, environment)
        return projected_area_drag_torque(
            relative_velocity, np.asarray(q_b_from_i, dtype=np.float64), density,
            self.normals_b, self.arms_b, self.areas, self.drag_coefficients,
        )
