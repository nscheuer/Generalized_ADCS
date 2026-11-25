import numpy as np
import pytest


def test_quaternion_normalization_and_rate_values(state_backend):
    state = state_backend.AttitudeState(
        [2, 0, 0, 0], np.array([1, 2, 3], dtype=np.float32)
    )
    np.testing.assert_array_equal(state.quaternion_array, [1, 0, 0, 0])
    np.testing.assert_array_equal(state.rates, [1, 2, 3])
    assert np.asarray(state.quaternion_array).dtype == np.float64


def test_hamilton_orientation_and_valid_reassignment(state_backend):
    class Orientation:
        hamilton_components = [0, 0, 0, -4]

    state = state_backend.AttitudeState(Orientation(), [0, 0, 0])
    np.testing.assert_array_equal(state.quaternion_array, [0, 0, 0, -1])
    state.orientation = [0, 2, 0, 0]
    state.rates = [4, 5, 6]
    np.testing.assert_array_equal(state.quaternion_array, [0, 1, 0, 0])
    np.testing.assert_array_equal(state.rates, [4, 5, 6])


@pytest.mark.parametrize("rates", ([1, 2], [0, np.inf, 0], [np.nan, 0, 0]))
def test_bad_rates_rejected(state_backend, rates):
    with pytest.raises(ValueError):
        state_backend.AttitudeState([1, 0, 0, 0], rates)


@pytest.mark.parametrize("orientation", ([0, 0, 0, 0], [1, 2, 3], [np.inf, 0, 0, 0]))
def test_bad_orientation_rejected_by_public_use(state_backend, orientation):
    with pytest.raises(ValueError):
        _ = state_backend.AttitudeState(orientation, [0, 0, 0]).quaternion_array
