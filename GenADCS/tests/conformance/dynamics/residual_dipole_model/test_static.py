import numpy as np


def test_residual_dipole_torque_matches_body_frame_cross_product(dynamics_api):
    dipole = np.array([1.0, 0.0, 0.0])
    field_inertial = np.array([0.0, 0.0, 2e-5])
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])

    torque = dynamics_api.static_residual_dipole_torque(
        dipole, field_inertial, quaternion
    )

    np.testing.assert_allclose(torque, np.cross(dipole, field_inertial), atol=1e-20)


def test_residual_dipole_torque_vanishes_for_zero_dipole(dynamics_api):
    torque = dynamics_api.static_residual_dipole_torque(
        np.zeros(3),
        np.array([2e-5, -1e-5, 3e-5]),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(torque, np.zeros(3))
