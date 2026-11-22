"""Numerical parity checks for the Eigen/pybind helper translation."""
from __future__ import annotations

import numpy as np

from GenADCS.backends.cpp.helpers import math as cpp
from GenADCS.backends.python.helpers import math as py
from GenADCS.backends.python.helpers.math import time as py_time


def close(actual, expected, label: str, atol: float = 1e-12) -> None:
    if not np.allclose(actual, expected, rtol=1e-12, atol=atol):
        raise AssertionError(f"{label} differs:\nC++: {actual}\nPython: {expected}")


def main() -> None:
    a = np.array([0.3, -0.4, 0.8])
    b = np.array([-0.2, 0.9, 0.1])
    close(cpp.cross(a, b), py.cross(a, b), "cross")
    close(cpp.normalize(a), py.normalize(a), "normalize")
    close(cpp.skew(a), py.skew(a), "skew")

    q = np.array([0.8, 0.2, -0.3, 0.45])
    q /= np.linalg.norm(q)
    omega = np.array([0.02, -0.03, 0.04])
    close(cpp.quaternion_derivative(q, omega), py.quaternion_derivative(q, omega), "quaternion derivative")
    close(cpp.quaternion_rate_matrix(q), py.quaternion_rate_matrix(q), "quaternion rate matrix")
    close(cpp.quaternion_normalization_jacobian(q), py.quaternion_normalization_jacobian(q), "normalization Jacobian")
    close(cpp.rotated_vector_quaternion_jacobian(q, a), py.rotated_vector_quaternion_jacobian(q, a), "rotated-vector Jacobian")

    inertia = np.diag([2.0, 3.0, 4.0])
    torque = np.array([0.1, -0.2, 0.3])
    close(cpp.rigid_body_angular_acceleration(omega, torque, inertia), py.rigid_body_angular_acceleration(omega, torque, inertia), "rigid-body dynamics")
    axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    wheel_momentum = np.array([0.01, -0.02])
    close(cpp.wheel_coupled_angular_acceleration(omega, wheel_momentum, axes, torque, inertia), py.wheel_coupled_angular_acceleration(omega, wheel_momentum, axes, torque, inertia), "wheel dynamics")
    close(cpp.wheel_momentum_derivative(wheel_momentum, omega, axes), py.wheel_momentum_derivative(wheel_momentum, omega, axes), "wheel momentum derivative")

    close(cpp.Quaternion(q).to_dcm(), py.Quaternion(q).to_dcm(), "quaternion DCM")
    close(cpp.DCM(cpp.Quaternion(q).to_dcm()).to_quaternion().components, py.DCM(py.Quaternion(q).to_dcm()).to_quaternion().components, "DCM quaternion")
    mrp = cpp.Modified_Rodriguez_Parameters.from_quaternion_array(q)
    mrp_py = py.Modified_Rodriguez_Parameters.from_quaternion_array(q)
    close(mrp.components, mrp_py.components, "MRP conversion")
    rod = cpp.Rodriguez_Parameters.from_quaternion_array(q)
    rod_py = py.Rodriguez_Parameters.from_quaternion_array(q)
    close(rod.components, rod_py.components, "Rodrigues conversion")
    close(cpp.slerp(q, -q, 0.4), py.slerp(q, -q, 0.4), "slerp")

    jd = cpp.calendar_to_jd(2024, 2, 29, 12, 34, 56.25)
    close(jd, py_time.calendar_to_jd(2024, 2, 29, 12, 34, 56.25), "calendar to JD")
    if cpp.jd_to_calendar(jd) != py_time.jd_to_calendar(jd):
        raise AssertionError("JD to calendar differs")
    for name in ("utc_jd_to_tai_jd", "tai_jd_to_tt_jd", "utc_jd_to_tt_jd", "utc_jd_to_tdb_jd"):
        close(getattr(cpp, name)(jd), getattr(py_time, name)(jd), name)
    print("C++ helper translation matches the Python reference for tested operations.")


if __name__ == "__main__":
    main()
