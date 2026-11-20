import numpy as np
from numba import njit


@njit(cache=True)
def atmospheric_density(position_inertial_m, epoch_jd):
    """Density [kg/m³] for an ECI position [m] at UTC Julian date."""
    raise NotImplementedError


@njit(cache=True)
def co_rotating_atmosphere_velocity(position_inertial_m, epoch_jd, angular_velocity_inertial_rad_s=None):
    """Rigidly co-rotating atmospheric velocity in ECI coordinates [m/s].

    The optional angular-velocity vector allows callers to supply their own
    Earth orientation. By default the mean sidereal rate is used along ECI z.
    """
    if angular_velocity_inertial_rad_s is None:
        omega = np.array((0.0, 0.0, 7.2921150e-5))
    else:
        omega = angular_velocity_inertial_rad_s
    return np.cross(omega, position_inertial_m)


class AtmosphericModel:
    density = staticmethod(atmospheric_density)
    velocity = staticmethod(co_rotating_atmosphere_velocity)
