"""Projected area drag force and torque wrappers."""
import numpy as np
from .._geometry import prepare_panels
from .._dynamics import projected_area_drag_force, projected_area_drag_torque


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
        return projected_area_drag_force(relative_velocity, np.asarray(q_b_from_i, dtype=np.float64), density,
                                         self.normals_b, self.areas, self.drag_coefficients)


class ProjectedAreaDragTorqueModel(_ProjectedAreaDragBase):
    def evaluate(self, position_i, velocity_i, q_b_from_i, epoch_jd, environment):
        relative_velocity, density = self._flow(position_i, velocity_i, epoch_jd, environment)
        return projected_area_drag_torque(relative_velocity, np.asarray(q_b_from_i, dtype=np.float64), density,
                                          self.normals_b, self.arms_b, self.areas, self.drag_coefficients)

__all__ = ["ProjectedAreaDragForceModel", "ProjectedAreaDragTorqueModel", "projected_area_drag_force", "projected_area_drag_torque"]
