import numpy as np
from numba import njit


@njit(cache=True)
def rigid_body_angular_acceleration(angular_velocity_b, torque_b, inertia_b):
    angular_momentum = inertia_b @ angular_velocity_b
    return np.linalg.solve(
        inertia_b, torque_b - np.cross(angular_velocity_b, angular_momentum)
    )


@njit(cache=True)
def wheel_coupled_angular_acceleration(
    angular_velocity_b, wheel_momentum, wheel_axes_b, torque_b, bus_inertia_b
):
    total_momentum = (
        bus_inertia_b @ angular_velocity_b + wheel_axes_b.T @ wheel_momentum
    )
    return np.linalg.solve(
        bus_inertia_b, torque_b - np.cross(angular_velocity_b, total_momentum)
    )


@njit(cache=True)
def wheel_momentum_derivative(wheel_torque, angular_acceleration_b, wheel_axes_b):
    return wheel_torque - wheel_axes_b @ angular_acceleration_b
