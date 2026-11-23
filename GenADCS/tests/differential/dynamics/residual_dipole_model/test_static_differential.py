import numpy as np


def test_residual_dipole_torque_matches_across_backends(dynamics_backends):
    cases = (
        (
            np.array([1.0, -0.4, 0.2]),
            np.array([2e-5, -1e-5, 3e-5]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        ),
        (
            np.array([-0.3, 0.7, 0.1]),
            np.array([-2e-5, 4e-5, 1e-5]),
            np.array([0.9659258262890683, 0.0, 0.0, 0.25881904510252074]),
        ),
        (
            np.zeros(3),
            np.array([2e-5, 1e-5, -3e-5]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        ),
    )

    for dipole, field, quaternion in cases:
        results = {
            name: backend.static_residual_dipole_torque(dipole, field, quaternion)
            for name, backend in dynamics_backends.items()
        }
        reference = results["python"]
        for torque in results.values():
            np.testing.assert_allclose(torque, reference, rtol=2e-12, atol=1e-20)
