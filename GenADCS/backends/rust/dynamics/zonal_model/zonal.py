"""Selectable central plus J2–J6 gravity force model."""
import numpy as np
from .._dynamics import zonal_gravity_force
from GenADCS.backends.rust.environment.gravity import EARTH_ZONAL_COEFFICIENTS


class ZonalForceModel:
    def __init__(self, geometry, environment, degrees=(2, 3, 4, 5, 6)):
        degrees = tuple(sorted(set(degrees)))
        if any(degree not in (2, 3, 4, 5, 6) for degree in degrees):
            raise ValueError("degrees must be selected from 2 through 6")
        self.mass = float(geometry.mass)
        if not np.isfinite(self.mass) or self.mass <= 0.0:
            raise ValueError("geometry.mass must be finite and positive")
        body = environment.planetary_body
        self.gravitational_parameter = float(body.gravitational_parameter)
        self.reference_radius = float(body.reference_radius)
        if self.gravitational_parameter <= 0.0 or self.reference_radius <= 0.0:
            raise ValueError("environment body mu and reference radius must be positive")
        configured = environment.j_gravity
        source = configured.zonal_coefficients if configured is not None else EARTH_ZONAL_COEFFICIENTS
        if len(source) < 5:
            raise ValueError("gravity coefficients must provide J2 through J6")
        coefficients = np.zeros(5, dtype=np.float64)
        for degree in degrees:
            coefficients[degree - 2] = source[degree - 2]
        self.degrees = degrees
        self.zonal_coefficients = np.ascontiguousarray(coefficients)

    def evaluate(self, position_i, epoch_jd=0.0):
        position_i = np.ascontiguousarray(position_i, dtype=np.float64)
        if position_i.shape != (3,) or not np.all(np.isfinite(position_i)):
            raise ValueError("position_i must be a finite inertial 3-vector")
        return zonal_gravity_force(position_i, self.mass, float(epoch_jd), self.gravitational_parameter,
                                   self.reference_radius, self.zonal_coefficients)

__all__ = ["ZonalForceModel", "zonal_gravity_force"]
