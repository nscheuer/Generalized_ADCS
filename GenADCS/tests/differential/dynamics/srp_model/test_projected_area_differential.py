import numpy as np


def test_srp_force_and_torque_match_across_backends(dynamics_backends):
    cases = (
        (
            np.zeros(3), np.array([149_597_870_700.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.array([[0.0, 0.5, 0.0], [0.0, 0.0, -0.3]]),
            np.array([2.0, 1.2]),
            np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]]),
            1.0,
        ),
        (
            np.array([1e7, -2e7, 3e7]),
            np.array([1.2e11, 5.0e10, -3.0e10]),
            np.array([0.9659258262890683, 0.0, 0.25881904510252074, 0.0]),
            np.array([[0.0, 0.0, 1.0]]),
            np.array([[0.3, -0.1, 0.5]]),
            np.array([3.5]),
            np.array([[0.4, 0.2, 0.4]]),
            0.37,
        ),
    )

    for position, sun, quaternion, normals, arms, areas, optical, illumination in cases:
        results = {}
        for name, backend in dynamics_backends.items():
            results[name] = (
                backend.projected_area_srp_force(
                    position, sun, quaternion, normals, areas, optical, illumination
                ),
                backend.projected_area_srp_torque(
                    position, sun, quaternion, normals, arms, areas, optical, illumination
                ),
            )

        reference = results["python"]
        for force, torque in results.values():
            np.testing.assert_allclose(force, reference[0], rtol=2e-12, atol=1e-18)
            np.testing.assert_allclose(torque, reference[1], rtol=2e-12, atol=1e-18)
