import numpy as np
import pytest

from ADCS.covariance import Covariance
from ADCS.state import EstimatorState, State


def _unit(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def test_interpolate_slerp_matches_exact_geodesic():
    theta = 0.3
    a = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0], h=[0.1])
    b = State(w=[1.0, 1.0, 1.0], q=[np.cos(theta / 2), np.sin(theta / 2), 0.0, 0.0], h=[0.3])

    mid = a.interpolate(b, 0.5)

    expected_q = np.array([np.cos(theta / 4), np.sin(theta / 4), 0.0, 0.0])
    assert np.allclose(mid.q, expected_q, atol=1e-12)
    assert np.allclose(mid.w, [0.5, 0.5, 0.5])
    assert np.allclose(mid.h, [0.2])


@pytest.mark.parametrize("method", ["slerp", "nlerp"])
def test_interpolate_corrects_antipodal_quaternion_sign(method):
    theta = 0.3
    q1 = np.array([np.cos(theta / 2), np.sin(theta / 2), 0.0, 0.0])
    a = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])
    b = State(w=np.zeros(3), q=-q1)  # same rotation, opposite representation

    mid = a.interpolate(b, 0.5, method=method)

    expected_q = np.array([np.cos(theta / 4), np.sin(theta / 4), 0.0, 0.0])
    # Sign-blind lerp would pass near the origin and normalize garbage instead.
    assert min(np.linalg.norm(mid.q - expected_q), np.linalg.norm(mid.q + expected_q)) < 1e-4
    assert np.isclose(np.linalg.norm(mid.q), 1.0)


def test_interpolate_endpoints_and_validation():
    a = State(w=[1.0, 2.0, 3.0], q=_unit([0.9, 0.2, 0.3, 0.1]), h=[0.1])
    b = State(w=[4.0, 5.0, 6.0], q=_unit([0.5, -0.5, 0.5, 0.5]), h=[0.7])

    assert np.allclose(a.interpolate(b, 0.0).as_array(), a.as_array())
    assert np.allclose(a.interpolate(b, 1.0).as_array(), b.as_array())
    with pytest.raises(TypeError):
        a.interpolate(np.zeros(8), 0.5)
    with pytest.raises(ValueError):
        a.interpolate(State(w=b.w, q=b.q, h=[0.1, 0.2]), 0.5)
    with pytest.raises(ValueError):
        a.interpolate(b, 0.5, method="squad")


def test_subtract_small_angle_matches_rotation_vector():
    rot_vec = np.array([1.0e-3, -2.0e-3, 0.5e-3])
    dq = np.concatenate(([1.0], rot_vec / 2.0))
    ref = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])
    cur = State(w=np.zeros(3), q=_unit(dq))

    err = cur.subtract(ref)

    assert err.shape == (6,)
    assert np.allclose(err[3:6], rot_vec, rtol=1e-6)


def test_subtract_add_error_roundtrip_state():
    rng = np.random.default_rng(7)
    for _ in range(10):
        s1 = State(w=rng.normal(size=3), q=_unit(rng.normal(size=4)), h=rng.normal(size=2))
        s2 = State(w=rng.normal(size=3), q=_unit(rng.normal(size=4)), h=rng.normal(size=2))

        recovered = s2.add_error(s1.subtract(s2))

        assert np.allclose(recovered.w, s1.w)
        assert np.allclose(recovered.h, s1.h)
        # q and -q are the same rotation.
        assert (
            min(
                np.linalg.norm(recovered.q - s1.q),
                np.linalg.norm(recovered.q + s1.q),
            )
            < 1e-12
        )


def test_subtract_is_sign_invariant_in_representation():
    q = _unit([0.9, 0.2, 0.3, 0.1])
    ref = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])

    err_pos = State(w=np.zeros(3), q=q).subtract(ref)
    err_neg = State(w=np.zeros(3), q=-q).subtract(ref)

    assert np.allclose(err_pos, err_neg)


def test_subtract_add_error_roundtrip_estimated_state():
    rng = np.random.default_rng(11)
    kwargs = dict(h=rng.normal(size=1), act_bias=rng.normal(size=2), sens_bias=rng.normal(size=3), dist_param=rng.normal(size=1))
    e1 = EstimatorState(w=rng.normal(size=3), q=_unit(rng.normal(size=4)), **kwargs)
    kwargs2 = dict(h=rng.normal(size=1), act_bias=rng.normal(size=2), sens_bias=rng.normal(size=3), dist_param=rng.normal(size=1))
    e2 = EstimatorState(w=rng.normal(size=3), q=_unit(rng.normal(size=4)), **kwargs2)

    err = e1.subtract(e2)
    assert err.shape == (e1.augmented_size - 1,)

    recovered = e2.add_error(err)
    assert np.allclose(recovered.w, e1.w)
    assert np.allclose(recovered.h, e1.h)
    assert np.allclose(recovered.act_bias, e1.act_bias)
    assert np.allclose(recovered.sens_bias, e1.sens_bias)
    assert np.allclose(recovered.dist_param, e1.dist_param)
    assert (
        min(np.linalg.norm(recovered.q - e1.q), np.linalg.norm(recovered.q + e1.q))
        < 1e-12
    )


def test_estimated_interpolate_blends_blocks_and_covariance():
    cov1 = np.eye(9) * 2.0
    cov2 = np.eye(9) * 4.0
    e1 = EstimatorState(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0], h=[0.0], sens_bias=[0.0, 0.0], cov=cov1)
    e2 = EstimatorState(w=[2.0, 2.0, 2.0], q=[1.0, 0.0, 0.0, 0.0], h=[1.0], sens_bias=[2.0, 4.0], cov=cov2)

    mid = e1.interpolate(e2, 0.5)

    assert np.allclose(mid.w, [1.0, 1.0, 1.0])
    assert np.allclose(mid.h, [0.5])
    assert np.allclose(mid.sens_bias, [1.0, 2.0])
    assert np.allclose(mid.cov, np.eye(9) * 3.0)


def test_estimated_interpolate_rejects_mismatched_layouts():
    e1 = EstimatorState(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0], sens_bias=[0.0])
    e2 = EstimatorState(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0], sens_bias=[0.0, 0.0])
    with pytest.raises(ValueError):
        e1.interpolate(e2, 0.5)
    with pytest.raises(TypeError):
        e1.interpolate(State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0]), 0.5)


def test_add_error_rejects_out_of_range_attitude_block():
    s = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])
    delta = np.zeros(6)
    delta[3] = 2.5  # |δθ| > 2 has no unit-quaternion pre-image
    with pytest.raises(ValueError):
        s.add_error(delta)


@pytest.mark.parametrize(
    "mode",
    ["quaternion_vector", "rotation_vector", "mrp", "two_mrp", "cayley"],
)
@pytest.mark.parametrize("order", ["right", "left"])
def test_plus_minus_roundtrip_for_supported_attitude_conventions(mode, order):
    state = State(
        w=[0.1, -0.2, 0.3],
        q=_unit([0.8, -0.1, 0.3, 0.2]),
        h=[0.4, -0.5],
    )
    delta = np.array([0.01, -0.02, 0.03, 0.08, -0.04, 0.02, 0.05, -0.06])

    perturbed = state.plus(delta, quaternion_mode=mode, quaternion_order=order)
    recovered = perturbed.minus(state, quaternion_mode=mode, quaternion_order=order)

    assert np.allclose(recovered, delta, atol=1e-12)


def test_quaternion_delta_can_be_applied_directly_on_either_side():
    state = State(w=np.zeros(3), q=_unit([0.8, 0.1, -0.2, 0.3]))
    dq = State.quaternion_delta_from_vector([0.1, -0.05, 0.02], mode="rotation_vector")

    right = state.with_quaternion_delta(dq, order="right")
    left = state.with_quaternion_delta(dq, order="left")

    assert not np.allclose(right.q, left.q)
    assert np.allclose(
        right.minus(state, quaternion_mode="rotation_vector", quaternion_order="right")[3:6],
        [0.1, -0.05, 0.02],
    )
    assert np.allclose(
        left.minus(state, quaternion_mode="rotation_vector", quaternion_order="left")[3:6],
        [0.1, -0.05, 0.02],
    )


def test_full_quaternion_additive_mode_roundtrips_and_linearizes_normalization():
    reference = State(w=[0.1, 0.2, 0.3], q=_unit([0.8, 0.1, -0.2, 0.3]), h=[0.4])
    target = State(w=[0.3, 0.1, 0.4], q=-_unit([0.7, -0.2, 0.1, 0.4]), h=[0.8])

    delta = target.minus(reference, quaternion_mode="full_quaternion")
    recovered = reference.plus(delta, quaternion_mode="full_quaternion")

    assert delta.shape == (reference.full_size,)
    assert recovered.is_close(target, atol=1e-12)
    epsilon = 1e-7
    finite_difference = np.empty((reference.full_size, reference.full_size))
    for column in range(reference.full_size):
        offset = np.zeros(reference.full_size)
        offset[column] = epsilon
        plus = reference.plus(offset, quaternion_mode="full_quaternion").as_array()
        minus = reference.plus(-offset, quaternion_mode="full_quaternion").as_array()
        finite_difference[:, column] = (plus - minus) / (2.0 * epsilon)
    assert np.allclose(
        reference.tangent_map(quaternion_mode="full_quaternion"),
        finite_difference,
        atol=1e-9,
    )


@pytest.mark.parametrize(
    "mode",
    ["quaternion_vector", "rotation_vector", "mrp", "two_mrp", "cayley"],
)
@pytest.mark.parametrize("order", ["right", "left"])
def test_tangent_map_matches_finite_difference_and_pseudoinverse(mode, order):
    state = State(
        w=[0.1, -0.2, 0.3],
        q=_unit([0.8, -0.1, 0.3, 0.2]),
        h=[0.4, -0.5],
    )
    epsilon = 1e-7
    finite_difference = np.empty((state.full_size, state.tangent_size))
    for column in range(state.tangent_size):
        offset = np.zeros(state.tangent_size)
        offset[column] = epsilon
        plus = state.plus(offset, quaternion_mode=mode, quaternion_order=order).as_array()
        minus = state.plus(-offset, quaternion_mode=mode, quaternion_order=order).as_array()
        finite_difference[:, column] = (plus - minus) / (2.0 * epsilon)

    tangent = state.tangent_map(quaternion_mode=mode, quaternion_order=order)
    tangent_pinv = state.tangent_pinv(quaternion_mode=mode, quaternion_order=order)

    assert np.allclose(tangent, finite_difference, atol=1e-9)
    assert np.allclose(tangent_pinv @ tangent, np.eye(state.tangent_size), atol=1e-12)


@pytest.mark.parametrize(
    "mode",
    ["quaternion_vector", "rotation_vector", "mrp", "two_mrp", "cayley"],
)
@pytest.mark.parametrize("order", ["right", "left"])
def test_retraction_jacobian_matches_change_of_tangent_point(mode, order):
    state = EstimatorState(
        w=[0.1, -0.2, 0.3],
        q=_unit([0.8, -0.1, 0.3, 0.2]),
        h=[0.4],
        sens_bias=[0.01, -0.02],
    )
    delta = np.linspace(-0.02, 0.03, state.tangent_size)
    updated = state.plus(delta, quaternion_mode=mode, quaternion_order=order)
    epsilon = 1.0e-7
    numerical = np.empty((state.tangent_size, state.tangent_size))
    for column in range(state.tangent_size):
        offset = np.zeros(state.tangent_size)
        offset[column] = epsilon
        plus = state.plus(
            delta + offset, quaternion_mode=mode, quaternion_order=order
        ).minus(updated, quaternion_mode=mode, quaternion_order=order, shortest=False)
        minus = state.plus(
            delta - offset, quaternion_mode=mode, quaternion_order=order
        ).minus(updated, quaternion_mode=mode, quaternion_order=order, shortest=False)
        numerical[:, column] = (plus - minus) / (2.0 * epsilon)

    np.testing.assert_allclose(
        state.retraction_jacobian(
            delta, quaternion_mode=mode, quaternion_order=order
        ),
        numerical,
        atol=2.0e-9,
    )


@pytest.mark.parametrize("form", ["full", "sqrt"])
def test_retraction_transport_preserves_covariance_representation(form):
    state = State(w=np.zeros(3), q=_unit([0.9, 0.2, -0.1, 0.3]), h=[0.2])
    delta = np.linspace(-0.02, 0.03, state.tangent_size)
    covariance = Covariance.identity(
        state.tangent_size,
        scale=0.1,
        form=form,
        coordinates="state_tangent",
    )
    reset = state.retraction_jacobian(delta, quaternion_mode="rotation_vector")

    transported = state.transport_covariance(
        covariance,
        delta,
        quaternion_mode="rotation_vector",
    )

    assert transported.form == form
    assert transported.coordinates == "state_tangent"
    np.testing.assert_allclose(
        transported.as_matrix(),
        reset @ covariance.as_matrix() @ reset.T,
        atol=1.0e-14,
    )


def test_right_error_reset_has_expected_small_angle_sign():
    state = State(w=np.zeros(3), q=_unit([0.8, 0.1, -0.2, 0.3]))
    delta = np.zeros(state.tangent_size)
    attitude = np.array([1.0e-4, -2.0e-4, 0.5e-4])
    delta[state.slice("attitude", coordinates="tangent")] = attitude
    cross = np.array(
        [
            [0.0, -attitude[2], attitude[1]],
            [attitude[2], 0.0, -attitude[0]],
            [-attitude[1], attitude[0], 0.0],
        ]
    )

    reset = state.retraction_jacobian(
        delta,
        quaternion_mode="rotation_vector",
        quaternion_order="right",
    )

    np.testing.assert_allclose(reset[3:6, 3:6], np.eye(3) - 0.5 * cross, atol=2.0e-8)


def test_state_mean_handles_quaternion_sign_and_linear_blocks():
    angle = 0.4
    first = State(w=[0.0, 0.0, 0.0], q=[1.0, 0.0, 0.0, 0.0], h=[0.0])
    second = State(
        w=[2.0, 4.0, 6.0],
        q=-np.array([np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0]),
        h=[2.0],
    )

    mean = State.mean([first, second], [0.25, 0.75], quaternion_mode="rotation_vector")

    expected_q = np.array([np.cos(0.75 * angle / 2.0), 0.0, np.sin(0.75 * angle / 2.0), 0.0])
    assert mean.is_close(State(w=[1.5, 3.0, 4.5], q=expected_q, h=[1.5]), atol=1e-12)


def test_estimator_state_geometry_extends_augmented_blocks_and_covariance():
    reduced_size = 6 + 1 + 2 + 1 + 2
    covariance = np.diag(np.linspace(1.0, 2.0, reduced_size))
    state = EstimatorState(
        w=[0.1, -0.2, 0.3],
        q=_unit([0.8, -0.1, 0.3, 0.2]),
        h=[0.4],
        act_bias=[0.01, 0.02],
        sens_bias=[0.03],
        dist_param=[0.04, 0.05],
        cov=covariance,
    )
    delta = np.linspace(-0.03, 0.04, state.tangent_size)

    perturbed = state.plus(delta, quaternion_mode="two_mrp", quaternion_order="left")
    assert np.allclose(
        perturbed.minus(state, quaternion_mode="two_mrp", quaternion_order="left"),
        delta,
        atol=1e-12,
    )
    full_covariance = state.covariance_to_full()
    assert full_covariance.shape == (state.full_size, state.full_size)
    assert np.allclose(state.covariance_to_reduced(full_covariance), covariance, atol=1e-12)
    assert np.allclose(state.normalization_jacobian()[7:, 7:], np.eye(state.full_size - 7))


def test_estimator_mean_uses_explicit_covariance_policy():
    first = EstimatorState(
        w=np.zeros(3),
        q=[1.0, 0.0, 0.0, 0.0],
        sens_bias=[0.0],
        cov=np.eye(7),
    )
    second = EstimatorState(
        w=np.ones(3),
        q=[1.0, 0.0, 0.0, 0.0],
        sens_bias=[2.0],
        cov=3.0 * np.eye(7),
    )

    mean = EstimatorState.mean([first, second], [0.25, 0.75], covariance="weighted")

    assert np.allclose(mean.w, 0.75)
    assert np.allclose(mean.sens_bias, [1.5])
    assert np.allclose(mean.cov, 2.5 * np.eye(7))
