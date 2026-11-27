from typing import Any


def test_device_scalar_fields_match(adcs_backends: Any) -> None:
    wheels = {
        name: api.ReactionWheel(
            (1, 2, 3), (0, 0, 1), 0.02, 0.1, 2.0, misalignment=(0.1, 0.2, 0.3)
        )
        for name, api in adcs_backends.items()
    }
    torquers = {
        name: api.Magnetorquer((3, 2, 1), (1, 0, 0), 0.25, misalignment=(0.2, 0.1, 0.0))
        for name, api in adcs_backends.items()
    }
    for name, wheel in wheels.items():
        assert (
            wheel.rotational_inertia,
            wheel.max_torque,
            wheel.max_momentum,
            wheel.misalignment,
        ) == (0.02, 0.1, 2.0, (0.1, 0.2, 0.3)), name
    for name, torquer in torquers.items():
        assert (torquer.max_dipole, torquer.misalignment) == (0.25, (0.2, 0.1, 0.0)), (
            name
        )


def test_device_noise_and_bias_match(adcs_backends: Any) -> None:
    for name, api in adcs_backends.items():
        wheel = api.ReactionWheel(
            (0, 0, 0), (0, 0, 1), 0.1, 0.2, 1.0, noise=0.02, bias=-0.01
        )
        torquer = api.Magnetorquer(
            (0, 0, 0), (0, 0, 1), 0.2, noise=0.03, bias=0.01
        )
        assert (wheel.noise, wheel.bias) == (0.02, -0.01), name
        assert (torquer.noise, torquer.bias) == (0.03, 0.01), name
