import numpy as np
import pytest
from numpy.typing import ArrayLike

from GenADCS.backends.python.state import (
    AttitudeState,
    InternalState,
    OrbitalState,
    State,
)


def make_state(momentum: ArrayLike = (0.1, -0.2)) -> State:
    return State(
        AttitudeState([2, 0, 0, 0], [0.01, 0.02, 0.03]),
        OrbitalState.from_cartesian([7e6, 0, 0], [0, 7.5e3, 0], 2451545.0),
        InternalState(momentum),
    )


def test_state_default_internal_is_independent() -> None:
    base = State(
        AttitudeState([1, 0, 0, 0], [0, 0, 0]), OrbitalState([0, 0, 0], [0, 0, 0], 1)
    )
    other = State(
        AttitudeState([1, 0, 0, 0], [0, 0, 0]), OrbitalState([0, 0, 0], [0, 0, 0], 1)
    )
    base.internal.wheel_momentum = np.array([3.0])
    assert other.internal.wheel_momentum.size == 0


def test_to_record_schema_and_values() -> None:
    record = make_state().to_record()
    assert record.shape == (1,)
    assert record.dtype.names == (
        "quaternion",
        "rates",
        "position",
        "velocity",
        "momentum",
    )
    assert record.dtype["quaternion"].subdtype[0] == np.dtype(np.float64)
    np.testing.assert_array_equal(record["quaternion"][0], [1, 0, 0, 0])
    np.testing.assert_array_equal(record["rates"][0], [0.01, 0.02, 0.03])
    np.testing.assert_array_equal(record["position"][0], [7e6, 0, 0])
    np.testing.assert_array_equal(record["velocity"][0], [0, 7.5e3, 0])
    np.testing.assert_array_equal(record["momentum"][0], [0.1, -0.2])


@pytest.mark.parametrize("count", [0, 1, 3, -1])
def test_to_record_rejects_wheel_count_mismatch(count: int) -> None:
    with pytest.raises(ValueError, match="expected .* reaction wheel momenta"):
        make_state().to_record(count)


def test_to_record_requires_cartesian_orbit() -> None:
    state = State(
        AttitudeState([1, 0, 0, 0], [0, 0, 0]),
        OrbitalState.from_keplerian(7000, 0, 0, 0, 0, 0, 2451545),
    )
    with pytest.raises(NotImplementedError):
        state.to_record()


def test_from_record_roundtrip_and_owns_data() -> None:
    original = make_state()
    record = original.to_record()
    rebuilt = State.from_record(record, original.orbit.julian_date)
    np.testing.assert_array_equal(
        rebuilt.attitude.quaternion_array, original.attitude.quaternion_array
    )
    np.testing.assert_array_equal(rebuilt.attitude.rates, original.attitude.rates)
    np.testing.assert_array_equal(rebuilt.orbit.position, original.orbit.position)
    np.testing.assert_array_equal(rebuilt.orbit.velocity, original.orbit.velocity)
    np.testing.assert_array_equal(
        rebuilt.internal.wheel_momentum, original.internal.wheel_momentum
    )
    record["momentum"][0, 0] = 9
    assert rebuilt.internal.wheel_momentum[0] == 0.1


def test_from_record_accepts_numpy_record_scalar_and_propagates_ephemeris() -> None:
    record = make_state().to_record()[0]
    marker = object()
    rebuilt = State.from_record(record, 123.0, marker)
    assert rebuilt.orbit.julian_date == 123.0
    assert rebuilt.orbit.ephemeris is marker


def test_state_slots_and_component_assignment_are_direct() -> None:
    state = make_state()
    replacement = InternalState([8])
    state.internal = replacement
    assert state.internal is replacement
    with pytest.raises(AttributeError):
        state.extra = 1
