"""Fixed body-frame residual magnetic dipole."""

import numpy as np
from numba import njit

from ADCS_Omni.ADCS_Python.helpers.math.frames.frames import inertial_to_body


@njit(cache=True)
def static_residual_dipole_torque(dipole_b, magnetic_field_i, q_b_from_i):
    """Return m × B in body frame [N m], with m [A m²] and B [T]."""
    return np.cross(dipole_b, inertial_to_body(magnetic_field_i, q_b_from_i))


class StaticResidualDipoleModel:
    def __init__(self, dipole_b):
        self.dipole_b = np.ascontiguousarray(dipole_b, dtype=np.float64)
        if self.dipole_b.shape != (3,) or not np.all(np.isfinite(self.dipole_b)):
            raise ValueError("dipole_b must be a finite body-frame 3-vector")

    def evaluate(self, position_i, q_b_from_i, epoch_jd, environment):
        if environment.magnetic_field is None:
            raise ValueError("residual dipole requires environment.magnetic_field")
        field_i = environment.magnetic_field.field_inertial_tesla(
            np.asarray(position_i, dtype=np.float64), epoch_jd,
        )
        return static_residual_dipole_torque(
            self.dipole_b, field_i, np.asarray(q_b_from_i, dtype=np.float64),
        )
