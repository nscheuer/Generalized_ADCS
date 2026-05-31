import numpy as np

from testing.test_estimators.ukf.helpers import make_baseline_sensors, make_satellites, make_ukf


def test_add_to_state_applies_vector_update_in_error_state_mode():
    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    ukf = make_ukf(est_sat, quat_as_vec=False)
    state = ukf.x_hat.val.copy()
    add = np.array([1.0e-3, -2.0e-3, 3.0e-3, 5.0e-2, -4.0e-2, 3.0e-2])

    out = ukf.add_to_state(state, add)

    assert np.allclose(out[:3], state[:3] + add[:3])
    assert np.isclose(np.linalg.norm(out[3:7]), 1.0)
    assert not np.allclose(out[3:7], state[3:7])


def test_add_to_state_applies_batch_update_in_error_state_mode():
    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    ukf = make_ukf(est_sat, quat_as_vec=False)
    state = ukf.x_hat.val.copy()
    adds = np.array(
        [
            [1.0e-3, 0.0, 0.0, 5.0e-2, 0.0, 0.0],
            [0.0, -2.0e-3, 0.0, 0.0, -4.0e-2, 0.0],
        ]
    )

    out = ukf.add_to_state(state, adds)

    assert out.shape == (2, state.size)
    assert np.allclose(np.linalg.norm(out[:, 3:7], axis=1), 1.0)


def test_add_to_state_renormalizes_full_quaternion_mode():
    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    ukf = make_ukf(est_sat, quat_as_vec=True)
    state = ukf.x_hat.val.copy()
    add = np.zeros(state.size)
    add[3:7] = np.array([0.1, -0.05, 0.07, 0.02])

    out = ukf.add_to_state(state, add)

    assert np.isclose(np.linalg.norm(out[3:7]), 1.0)


def test_reunite_states_reconstructs_reduced_state():
    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    ukf = make_ukf(est_sat, quat_as_vec=False)
    dynstate = np.array([1.0e-3, -2.0e-3, 3.0e-3, 0.97, 0.1, -0.1, 0.18])
    dynstate[3:7] = dynstate[3:7] / np.linalg.norm(dynstate[3:7])
    rest = np.array([5.0e-4, -6.0e-4])
    quatref = np.array([1.0, 0.0, 0.0, 0.0])

    out = ukf.reunite_states(dynstate, rest, quatref)

    assert out.shape == (8,)
    assert np.allclose(out[:3], dynstate[:3])
    assert np.allclose(out[-2:], rest)


def test_reunite_states_is_identity_in_full_quaternion_mode():
    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    ukf = make_ukf(est_sat, quat_as_vec=True)
    dynstate = ukf.x_hat.val[:7].copy()
    rest = np.array([1.0e-3, -2.0e-3])

    out = ukf.reunite_states(dynstate, rest, ukf.x_hat.val[3:7])

    assert np.allclose(out, np.concatenate([dynstate, rest]))


def test_new_post_state_returns_full_state_with_normalized_quaternion():
    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    ukf = make_ukf(est_sat, quat_as_vec=False)
    post_dyn = np.array([2.0e-3, -1.0e-3, 5.0e-4, 0.95, 0.2, -0.1, 0.22])
    post_dyn[3:7] = post_dyn[3:7] / np.linalg.norm(post_dyn[3:7])
    rest = np.array([7.0e-4])
    int_err = np.array([1.0e-4, -2.0e-4, 3.0e-4, 1.0e-2, -2.0e-2, 5.0e-3, 4.0e-4])
    quatref = np.array([1.0, 0.0, 0.0, 0.0])

    post_state, full_state = ukf.new_post_state(rest, post_dyn, int_err, quatref)

    assert post_state.shape == (7,)
    assert full_state.shape == (8,)
    assert np.isclose(np.linalg.norm(full_state[3:7]), 1.0)
