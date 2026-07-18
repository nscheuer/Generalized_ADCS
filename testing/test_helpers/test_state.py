import numpy as np
import pytest

from ADCS import EstimatedState, State
from ADCS.satellite_hardware.satellite import Satellite


def test_state_roundtrip_owns_input_arrays():
    source = np.arange(9.0)
    state = State.from_array(source)
    source[:] = -1.0

    np.testing.assert_array_equal(state.as_array(), np.arange(9.0))
    assert state.w.shape == (3,)
    assert state.q.shape == (4,)
    assert state.h.shape == (2,)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"w": [1, 2], "q": [1, 0, 0, 0]}, "w must have shape"),
        ({"w": [1, 2, 3], "q": [1, 0, 0]}, "q must have shape"),
        ({"w": [1, 2, 3], "q": [1, 0, 0, 0], "h": [[0]]}, "h must be one-dimensional"),
    ],
)
def test_state_rejects_invalid_component_shapes(kwargs, message):
    with pytest.raises(ValueError, match=message):
        State(**kwargs)


def test_state_has_no_array_or_indexing_facade():
    state = State(w=np.zeros(3), q=np.array([1, 0, 0, 0]))
    with pytest.raises(TypeError):
        _ = state[0]
    with pytest.raises(TypeError):
        np.asarray(state, dtype=float)


def test_satellite_rejects_raw_state_vectors():
    satellite = Satellite()
    with pytest.raises(TypeError, match="x must be a State"):
        satellite.dynamics_core(np.array([0, 0, 0, 1, 0, 0, 0]), np.empty(0), None)


def test_state_normalized_does_not_mutate_original():
    state = State(w=np.zeros(3), q=np.array([2, 0, 0, 0]), h=[0.1])
    normalized = state.normalized()
    np.testing.assert_array_equal(state.q, [2, 0, 0, 0])
    np.testing.assert_array_equal(normalized.q, [1, 0, 0, 0])


def test_state_stack_requires_matching_widths():
    states = [
        State(w=np.zeros(3), q=[1, 0, 0, 0], h=[1]),
        State(w=np.ones(3), q=[1, 0, 0, 0], h=[2]),
    ]
    assert State.stack(states).shape == (2, 8)
    with pytest.raises(ValueError, match="same number"):
        State.stack([states[0], State(w=np.zeros(3), q=[1, 0, 0, 0])])


@pytest.mark.parametrize("full_covariance", [False, True])
def test_estimated_state_augmented_roundtrip_supports_covariance_modes(full_covariance):
    value = np.arange(13.0)
    cov_size = value.size if full_covariance else value.size - 1
    state = EstimatedState.from_estimator_array(
        value,
        n_rw=2,
        n_act_bias=1,
        n_sens_bias=2,
        n_dist_param=1,
        cov=np.eye(cov_size),
    )

    np.testing.assert_array_equal(state.as_array(), value[:9])
    np.testing.assert_array_equal(state.as_estimator_array(), value)
    assert state.uses_reduced_quaternion_covariance is not full_covariance
    assert isinstance(state, State)


def test_estimated_state_rejects_wrong_covariance_dimension():
    with pytest.raises(ValueError, match="reduced- or full-quaternion"):
        EstimatedState(w=np.zeros(3), q=[1, 0, 0, 0], cov=np.eye(2))


def test_state_dict_roundtrip_preserves_subclass_and_data():
    state = EstimatedState(
        w=[1, 2, 3],
        q=[1, 0, 0, 0],
        h=[4],
        act_bias=[5],
        sens_bias=[6],
        dist_param=[7],
    )
    rebuilt = EstimatedState.from_dict(state.to_dict())
    np.testing.assert_array_equal(rebuilt.as_estimator_array(), state.as_estimator_array())
    np.testing.assert_array_equal(rebuilt.cov, state.cov)
