import numpy as np
import pytest

from GenADCS.backends.python.state.attitude_state import AttitudeState


def test_rates_are_finite_contiguous_float64_and_detached() -> None:
    source = np.array([1, 2, 3], dtype=np.float32)
    state = AttitudeState([2, 0, 0, 0], source)
    rates = np.asarray(state.rates)
    assert rates.dtype == np.float64
    assert rates.flags.c_contiguous
    np.testing.assert_array_equal(rates, [1, 2, 3])
    source[0] = 99
    assert np.asarray(state.rates)[0] == 1


@pytest.mark.parametrize("rates", [[1, 2], [[1, 2, 3]], [0, np.inf, 0], [0, np.nan, 0]])
def test_rejects_invalid_rates(rates: object) -> None:
    with pytest.raises(ValueError, match="rates must be a finite 3-vector"):
        AttitudeState([1, 0, 0, 0], rates)


def test_quaternion_is_normalized_without_changing_orientation() -> None:
    state = AttitudeState([2, 0, 0, 0], [0, 0, 0])
    np.testing.assert_array_equal(state.quaternion_array, [1, 0, 0, 0])
    assert state.orientation == [2, 0, 0, 0]


def test_orientation_object_and_late_validation() -> None:
    class Orientation:
        def __init__(self) -> None:
            self.hamilton_components = [0, 0, 0, 5]

    state = AttitudeState(Orientation(), [0, 0, 0])
    np.testing.assert_array_equal(state.quaternion_array, [0, 0, 0, 1])
    state.orientation = [0, 0, 0, 0]
    with pytest.raises(ValueError, match="nonzero"):
        _ = state.quaternion_array


@pytest.mark.parametrize(
    "orientation", [[1, 0, 0], [1, 0, np.nan, 0], [np.inf, 0, 0, 0]]
)
def test_rejects_invalid_orientation_on_access(orientation: object) -> None:
    state = AttitudeState(orientation, [0, 0, 0])
    with pytest.raises(ValueError, match="finite Hamilton quaternion"):
        _ = state.quaternion_array


def test_quaternion_access_does_not_mutate_state() -> None:
    state = AttitudeState(np.array([2.0, 0, 0, 0]), [0, 0, 0])
    first = state.quaternion_array
    first[0] = 42
    np.testing.assert_array_equal(state.quaternion_array, [1, 0, 0, 0])


def test_accepts_negative_scaled_quaternion_and_returns_contiguous_unit_copy() -> None:
    state = AttitudeState([-3, 0, 0, 0], [0, 0, 0])
    q = state.quaternion_array
    np.testing.assert_array_equal(q, [-1, 0, 0, 0])
    assert q.dtype == np.float64 and q.flags.c_contiguous
    assert not np.shares_memory(q, np.asarray(state.orientation))


def test_slots_reject_unknown_attributes() -> None:
    with pytest.raises(AttributeError):
        AttitudeState([1, 0, 0, 0], [0, 0, 0]).extra = 1  # pyright: ignore[reportAttributeAccessIssue]
