import numpy as np
import pytest

from ADCS import EstimatorState, State
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


def test_state_assignment_validates_component_shapes_and_copies_values():
    state = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])
    source = np.array([0.1, 0.2])

    state.h = source
    source[:] = -1.0

    np.testing.assert_array_equal(state.h, [0.1, 0.2])
    with pytest.raises(ValueError, match="q must have shape"):
        state.q = [1.0, 0.0, 0.0]
    np.testing.assert_array_equal(state.q, [1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="w must be one-dimensional"):
        state.w = [[0.0, 0.0, 0.0]]


def test_state_has_no_array_or_indexing_facade():
    state = State(w=np.zeros(3), q=np.array([1, 0, 0, 0]))
    with pytest.raises(TypeError):
        _ = state[0]
    with pytest.raises(TypeError):
        np.asarray(state, dtype=float)


def test_state_equality_uses_array_values():
    state = State(w=[1.0, 2.0, 3.0], q=[1.0, 0.0, 0.0, 0.0], h=[0.1, 0.2])
    same = State(
        w=np.array([1.0, 2.0, 3.0]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        h=np.array([0.1, 0.2]),
    )
    different_value = State(w=[1.0, 2.0, 4.0], q=[1.0, 0.0, 0.0, 0.0], h=[0.1, 0.2])
    different_shape = State(w=[1.0, 2.0, 3.0], q=[1.0, 0.0, 0.0, 0.0], h=[0.1])

    assert state == same
    assert state != different_value
    assert state != different_shape
    assert state != object()


def test_state_and_estimated_state_are_not_equal():
    state = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])
    estimated = EstimatorState(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])

    assert state != estimated
    assert estimated != state


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


def test_state_stack_rejects_non_state_inputs_with_helpful_typeerror():
    with pytest.raises(TypeError, match="State.stack expects State objects"):
        State.stack([np.zeros(7)])


@pytest.mark.parametrize("full_covariance", [False, True])
def test_estimated_state_augmented_roundtrip_supports_covariance_modes(full_covariance):
    value = np.arange(13.0)
    cov_size = value.size if full_covariance else value.size - 1
    state = EstimatorState.from_estimator_array(
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
        EstimatorState(w=np.zeros(3), q=[1, 0, 0, 0], cov=np.eye(2))


def test_estimated_state_assignment_validates_estimator_blocks():
    state = EstimatorState(
        w=np.zeros(3),
        q=[1.0, 0.0, 0.0, 0.0],
        h=[0.1],
        act_bias=[0.2],
        sens_bias=[0.3],
        dist_param=[0.4],
    )
    source = np.array([0.5])

    state.act_bias = source
    source[:] = -1.0

    np.testing.assert_array_equal(state.act_bias, [0.5])
    with pytest.raises(ValueError, match="sens_bias must be one-dimensional"):
        state.sens_bias = [[0.1]]
    np.testing.assert_array_equal(state.sens_bias, [0.3])


def test_estimated_state_assignment_validates_covariance_shapes():
    state = EstimatorState(
        w=np.zeros(3),
        q=[1.0, 0.0, 0.0, 0.0],
        h=[0.1],
        act_bias=[0.2],
    )

    state.cov = np.eye(8)
    state.int_cov = np.eye(8) * 2.0

    np.testing.assert_array_equal(state.cov, np.eye(8))
    np.testing.assert_array_equal(state.int_cov, np.eye(8) * 2.0)
    with pytest.raises(ValueError, match="cov must be square"):
        state.cov = np.ones((8, 7))
    with pytest.raises(ValueError, match="int_cov must match cov shape"):
        state.int_cov = np.eye(9)
    with pytest.raises(ValueError, match="h assignment would make cov shape"):
        state.h = [0.1, 0.2]
    np.testing.assert_array_equal(state.h, [0.1])


def test_estimated_state_equality_includes_bias_and_covariance_blocks():
    kwargs = dict(
        w=[1.0, 2.0, 3.0],
        q=[1.0, 0.0, 0.0, 0.0],
        h=[0.1],
        act_bias=[0.2],
        sens_bias=[0.3, 0.4],
        dist_param=[0.5],
        cov=np.eye(11),
        int_cov=np.eye(11) * 2.0,
    )
    state = EstimatorState(**kwargs)
    same = EstimatorState(**kwargs)
    different_bias = EstimatorState(**{**kwargs, "act_bias": [0.25]})
    different_cov = EstimatorState(**{**kwargs, "cov": np.eye(11) * 3.0})
    different_int_cov = EstimatorState(**{**kwargs, "int_cov": np.eye(11) * 4.0})

    assert state == same
    assert state != different_bias
    assert state != different_cov
    assert state != different_int_cov
    assert state != object()


def test_state_dict_roundtrip_preserves_subclass_and_data():
    state = EstimatorState(
        w=[1, 2, 3],
        q=[1, 0, 0, 0],
        h=[4],
        act_bias=[5],
        sens_bias=[6],
        dist_param=[7],
    )
    rebuilt = EstimatorState.from_dict(state.to_dict())
    np.testing.assert_array_equal(rebuilt.as_estimator_array(), state.as_estimator_array())
    np.testing.assert_array_equal(rebuilt.cov, state.cov)
