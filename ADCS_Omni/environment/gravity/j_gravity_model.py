from numba import njit


@njit(cache=True)
def gravity_acceleration_inertial_m_s2(
    position_inertial_m,
    epoch_jd,
    gravitational_parameter_m3_s2,
    reference_radius_m,
    zonal_coefficients,
):
    raise NotImplementedError


class JGravityModel:
    def __init__(self, degrees=(2, 3, 4, 5, 6)):
        self.degrees = tuple(degrees)
        if any(degree not in (2, 3, 4, 5, 6) for degree in self.degrees):
            raise ValueError("J-gravity degrees must be selected from 2 through 6")

    acceleration_inertial_m_s2 = staticmethod(gravity_acceleration_inertial_m_s2)
    acceleration = staticmethod(gravity_acceleration_inertial_m_s2)
