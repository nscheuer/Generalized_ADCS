import numpy as np
import pytest

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
