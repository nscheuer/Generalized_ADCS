import numpy as np
from numba import njit


@njit(cache=True)
def rigid_body_angular_acceleration(
    angular_velocity_b: np.ndarray, torque_b: np.ndarray, inertia_b: np.ndarray
) -> np.ndarray:
    angular_momentum = inertia_b @ angular_velocity_b
    return np.linalg.solve(
        inertia_b, torque_b - np.cross(angular_velocity_b, angular_momentum)
    )


@njit(cache=True)
def wheel_coupled_angular_acceleration(
    angular_velocity_b: np.ndarray,
    wheel_momentum: np.ndarray,
    wheel_axes_b: np.ndarray,
    torque_b: np.ndarray,
    bus_inertia_b: np.ndarray,
) -> np.ndarray:
    total_momentum = (
        bus_inertia_b @ angular_velocity_b + wheel_axes_b.T @ wheel_momentum
    )
    return np.linalg.solve(
        bus_inertia_b, torque_b - np.cross(angular_velocity_b, total_momentum)
    )


@njit(cache=True)
def wheel_momentum_derivative(
    wheel_torque: np.ndarray,
    angular_acceleration_b: np.ndarray,
    wheel_axes_b: np.ndarray,
) -> np.ndarray:
    return wheel_torque - wheel_axes_b @ angular_acceleration_b
