from numba import njit


@njit(cache=True)
def magnetic_field_inertial_tesla(position_inertial_m, epoch_jd):
    raise NotImplementedError


class MagneticFieldModel:
    field_inertial_tesla = staticmethod(magnetic_field_inertial_tesla)
    field = staticmethod(magnetic_field_inertial_tesla)
