"""Exercise the ADCS Omni Python/C++ state conversion layer.

Run from the repository root after building ADCS_C (see its README).
"""

import numpy as np

from GenADCS.backends.cpp.state import (
    AttitudeState as CAttitudeState,
    InternalState as CInternalState,
    OrbitalState as COrbitalState,
    State as CState,
    WheelMomentumState as CWheelMomentumState,
)
from GenADCS.backends.python.state import (
    AttitudeState as PythonAttitudeState,
    OrbitalState as PythonOrbitalState,
    State as PythonState,
    WheelMomentumState as PythonWheelMomentumState,
)


def main():
    epoch = 2451545.0
    python_state = PythonState(
        PythonAttitudeState([2.0, 0.0, 0.0, 0.0], [0.01, 0.02, 0.03]),
        PythonOrbitalState.from_cartesian([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0], epoch),
        PythonWheelMomentumState(wheel_momentum=[0.1, -0.2]),
    )
    c_state = CState.from_python(python_state)
    np.testing.assert_array_equal(c_state.to_record(), python_state.to_record())
    np.testing.assert_array_equal(c_state.to_python().to_record(), python_state.to_record())
    np.testing.assert_array_equal(CState.from_record(python_state.to_record(), epoch).to_record(), python_state.to_record())

    attitude = CAttitudeState.from_python(python_state.attitude)
    orbit = COrbitalState.from_python(python_state.orbit)
    internal = CInternalState.from_python(python_state.internal)
    wheels = CWheelMomentumState.from_python(python_state.internal)
    np.testing.assert_array_equal(attitude.to_python().quaternion_array, python_state.attitude.quaternion_array)
    np.testing.assert_array_equal(orbit.to_python().position, python_state.orbit.position)
    np.testing.assert_array_equal(internal.to_python().wheel_momentum, python_state.internal.wheel_momentum)
    np.testing.assert_array_equal(wheels.to_python().wheel_momentum, python_state.internal.wheel_momentum)

    class FakeWheel:
        def __init__(self, momentum):
            self.h = momentum

    class FakeSubsystem:
        reaction_wheels = [FakeWheel(0.1), FakeWheel(-0.2)]

    np.testing.assert_array_equal(
        CWheelMomentumState(FakeSubsystem()).wheel_momentum,
        python_state.internal.wheel_momentum,
    )

    class FakeEphemeris:
        calls = 0

        def position(self, body, julian_date):
            self.calls += 1
            return np.array([body, julian_date, 0.0])

    ephemeris = FakeEphemeris()
    c_orbit = COrbitalState.from_cartesian([1, 2, 3], [4, 5, 6], epoch, ephemeris)
    np.testing.assert_array_equal(c_orbit.sun_position, [0, epoch, 0])
    np.testing.assert_array_equal(c_orbit.sun_position, [0, epoch, 0])
    assert ephemeris.calls == 1
    assert c_orbit.to_python().ephemeris is ephemeris

    keplerian = PythonOrbitalState.from_keplerian(7000, 0.01, 0.1, 0.2, 0.3, 0.4, epoch)
    assert COrbitalState.from_python(keplerian).to_python()._keplerian_elements == keplerian._keplerian_elements

    try:
        c_state.to_record(wheel_count=3)
    except ValueError:
        pass
    else:
        raise AssertionError("wheel count validation did not run")
    print("ADCS Omni C++ state translation: OK")


if __name__ == "__main__":
    main()
