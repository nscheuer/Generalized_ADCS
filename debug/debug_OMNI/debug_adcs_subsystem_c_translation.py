"""Parity checks for the C++ ADCS subsystem translation."""
from __future__ import annotations

from ADCS_Omni.ADCS_C.hardware.ADCS_subsystem import ADCSSubsystem as CSubsystem
from ADCS_Omni.ADCS_C.hardware.ADCS_subsystem import Magnetorquer as CMagnetorquer
from ADCS_Omni.ADCS_C.hardware.ADCS_subsystem import ReactionWheel as CReactionWheel
from ADCS_Omni.ADCS_C.hardware.ADCS_subsystem.effects import apply_effects as c_apply_effects
from ADCS_Omni.ADCS_Python.hardware.ADCS_subsystem import ADCSSubsystem, Magnetorquer, ReactionWheel
from ADCS_Omni.ADCS_Python.hardware.ADCS_subsystem.effects import apply_effects
from ADCS_Omni.ADCS_Python.bias import FixedBias
from ADCS_Omni.ADCS_Python.noise import GaussianWhiteNoise


class AddEffect:
    def apply(self, value, dt, rng=None):
        return value + dt


def main() -> None:
    effects = [GaussianWhiteNoise(0.1), FixedBias(0.2), AddEffect()]
    py_wheel = ReactionWheel((1.0, 2.0, 3.0), (0.0, 0.0, 1.0), 0.04, 0.2, 5.0, effects=effects)
    c_wheel = CReactionWheel((1.0, 2.0, 3.0), (0.0, 0.0, 1.0), 0.04, 0.2, 5.0, effects=effects)
    for field in ("position", "orientation", "rotational_inertia", "max_torque", "max_momentum", "misalignment"):
        if getattr(c_wheel, field) != getattr(py_wheel, field):
            raise AssertionError(f"ReactionWheel.{field} differs")
    if c_wheel.noises != py_wheel.noises or c_wheel.biases != py_wheel.biases:
        raise AssertionError("ReactionWheel effect filtering differs")
    c_wheel.effects = [effects[-1]]
    if c_wheel.apply_errors(1.5, 0.25) != apply_effects(1.5, [effects[-1]], 0.25):
        raise AssertionError("ReactionWheel effect application differs")
    if c_apply_effects(1.5, [effects[-1]], 0.25) != 1.75:
        raise AssertionError("apply_effects differs")

    py_magnetorquer = Magnetorquer((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.8)
    c_magnetorquer = CMagnetorquer((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.8)
    if (c_magnetorquer.position, c_magnetorquer.orientation, c_magnetorquer.max_dipole,
            c_magnetorquer.misalignment, c_magnetorquer.effects) != (
            py_magnetorquer.position, py_magnetorquer.orientation, py_magnetorquer.max_dipole,
            py_magnetorquer.misalignment, py_magnetorquer.effects):
        raise AssertionError("Magnetorquer fields differ")

    py_subsystem = ADCSSubsystem([py_wheel], [py_magnetorquer])
    c_subsystem = CSubsystem([c_wheel], [c_magnetorquer])
    if len(c_subsystem.reaction_wheels) != len(py_subsystem.reaction_wheels):
        raise AssertionError("ADCSSubsystem reaction wheel contents differ")
    if len(c_subsystem.magnetorquers) != len(py_subsystem.magnetorquers):
        raise AssertionError("ADCSSubsystem magnetorquer contents differ")
    c_subsystem.reaction_wheels.append(CReactionWheel((0, 0, 0), (1, 0, 0), 1.0, 1.0, 1.0))
    if len(c_subsystem.reaction_wheels) != 2:
        raise AssertionError("ADCSSubsystem lists are not mutable")
    print("C++ ADCS subsystem translation matches the Python reference for tested behavior.")


if __name__ == "__main__":
    main()
