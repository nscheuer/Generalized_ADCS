import numpy as np


def test_zonal_force_matches_across_backends(dynamics_backends):
    coefficients = np.array(
        [
            1.08262668e-3,
            -2.5324105e-6,
            -1.6198976e-6,
            -2.2729608e-7,
            5.406812e-7,
        ]
    )
    cases = (
        (np.array([7e6, 0.0, 0.0]), 10.0, 0.0, 3.986004418e14, 6_378_137.0),
        (np.array([-2e6, 3e6, 6e6]), 125.0, 2451545.0, 3.986004418e14, 6_378_137.0),
        (np.array([8e6, -1e6, -2e6]), 1.0, 2460000.5, 4.282837e13, 3_389_500.0),
    )

    for position, mass, epoch, mu, reference_radius in cases:
        results = {
            name: backend.zonal_gravity_force(
                position, mass, epoch, mu, reference_radius, coefficients
            )
            for name, backend in dynamics_backends.items()
        }
        reference = results["python"]
        for force in results.values():
            np.testing.assert_allclose(force, reference, rtol=2e-12, atol=1e-13)
