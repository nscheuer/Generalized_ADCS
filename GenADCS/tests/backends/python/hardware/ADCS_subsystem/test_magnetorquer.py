"""Constructor, noise/bias, and field tests for Magnetorquer."""

import pytest

from GenADCS.backends.python.hardware.ADCS_subsystem import Magnetorquer


def test_required_fields_and_defaults() -> None:
    mtq = Magnetorquer((1.0, 2.0, 3.0), (0.0, 0.0, 1.0), 0.25)
    assert mtq.position == (1.0, 2.0, 3.0)
    assert mtq.orientation == (0.0, 0.0, 1.0)
    assert mtq.max_dipole == 0.25
    assert mtq.misalignment == (0.0, 0.0, 0.0)
    assert mtq.noise == 0.0
    assert mtq.bias == 0.0


def test_noise_and_bias_validation() -> None:
    mtq = Magnetorquer((0, 0, 0), (0, 1, 0), 0.3, noise=0.02, bias=0.01)
    assert mtq.noise == 0.02
    assert mtq.bias == 0.01
    with pytest.raises(ValueError):
        Magnetorquer((0, 0, 0), (0, 1, 0), 0.3, noise=-0.01)
    with pytest.raises(ValueError):
        Magnetorquer((0, 0, 0), (0, 1, 0), 0.3, bias=float("inf"))


def test_fields_are_mutable_instance_is_slotted_and_missing_arguments_fail() -> None:
    mtq = Magnetorquer((0, 0, 0), (1, 0, 0), 0.2)
    mtq.position = (1, 0, 0)
    mtq.max_dipole = 0.35
    mtq.misalignment = (0.01, -0.02, 0.03)
    mtq.noise = 0.04
    mtq.bias = -0.05
    assert mtq.position == (1, 0, 0)
    assert mtq.max_dipole == 0.35
    assert mtq.misalignment == (0.01, -0.02, 0.03)
    assert (mtq.noise, mtq.bias) == (0.04, -0.05)
    with pytest.raises(AttributeError):
        mtq.extra = True
    with pytest.raises(TypeError):
        Magnetorquer((0, 0, 0), (1, 0, 0))


def test_constructor_preserves_supplied_mutable_values_without_coercion() -> None:
    position = [1, 2, 3]
    mtq = Magnetorquer(position, [0, 0, 1], 0.2)
    assert mtq.position is position
    position[0] = 8
    assert mtq.position == [8, 2, 3]
