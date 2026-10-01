"""Fast integration contracts for the next estimator implementations."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from ADCS.covariance import Covariance
from ADCS.estimators.process_model import propagate_state
from ADCS.estimators.process_noise import discretize_process_noise
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import (
    EarthHorizonSensor,
    Gyro,
    MTM,
    StarTracker,
    StarTrackerQuaternion,
    SunPair,
    SunSensor,
)
from ADCS.state import EstimatorState, State


@pytest.fixture(scope="module")
def orbital_state() -> Orbital_State:
    result = Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]),
        B=np.array([2.0e-5, -1.0e-5, 3.0e-5]),
        S=np.array([1.5e8, 1.0e7, -2.0e7]),
        rho=0.0,
        fast=True,
    )
    result._sunlit = True
    return result


def _estimate(*, wheel_momentum=()) -> EstimatorState:
    quaternion = np.array([0.9, 0.2, -0.3, 0.1])
    quaternion /= np.linalg.norm(quaternion)
    return EstimatorState(
        w=np.array([0.02, -0.015, 0.01]),
        q=quaternion,
        h=wheel_momentum,
    )


class _RecordingSatellite:
    state_len = 7

    def __init__(self):
        self.mid_orbital_state = None
        self.quat_as_vec = None

    def noiseless_rk4(
        self,
        state,
        control,
        dt,
        orbital_state_start,
        orbital_state_end,
        *,
        verbose,
        mid_orbital_state,
        quat_as_vec,
        give_err_est,
    ):
        self.mid_orbital_state = mid_orbital_state
        self.quat_as_vec = quat_as_vec
        return state.copy()


def _numerical_discrete_transition(
    satellite: EstimatedSatellite,
    state: EstimatorState,
    orbital_state: Orbital_State,
    dt: float,
) -> np.ndarray:
    control = np.zeros(satellite.control_len)

    def propagate(candidate: EstimatorState) -> EstimatorState:
        return propagate_state(
            candidate,
            satellite,
            control,
            dt,
            orbital_state,
            orbital_state,
            midpoint_orbital_state=orbital_state,
        )

    nominal = propagate(state)
    epsilon = 1.0e-7
    transition = np.empty((state.tangent_size, state.tangent_size))
    for column in range(state.tangent_size):
        offset = np.zeros(state.tangent_size)
        offset[column] = epsilon
        plus = propagate(state.plus(offset)).minus(nominal)
        minus = propagate(state.plus(-offset)).minus(nominal)
        transition[:, column] = (plus - minus) / (2.0 * epsilon)
    return transition


def test_deterministic_propagation_preserves_estimator_blocks_and_ownership(
    orbital_state,
):
    wheel = RW(
        axis=np.array([1.0, 0.0, 0.0]),
        max_torque=0.01,
        J=0.001,
        h=0.02,
        h_max=0.1,
    )
    satellite = EstimatedSatellite(
        J_0=np.diag([0.5, 0.8, 1.2]),
        actuators=[wheel],
    )
    state = EstimatorState(
        w=[0.02, -0.015, 0.01],
        q=[0.9, 0.2, -0.3, 0.1],
        h=[0.03],
        act_bias=[0.01],
        sens_bias=[-0.02, 0.03],
        dist_param=[0.04],
        cov=np.eye(11) * 0.2,
        int_cov=np.eye(11) * 0.01,
    ).normalized()
    original = state.copy()

    propagated = propagate_state(
        state,
        satellite,
        np.zeros(satellite.control_len),
        0.01,
        orbital_state,
        orbital_state,
        midpoint_orbital_state=orbital_state,
    )

    assert isinstance(propagated, EstimatorState)
    assert propagated is not state
    assert state == original
    assert wheel.h == 0.02
    assert not np.array_equal(propagated.as_array(), state.as_array())
    np.testing.assert_array_equal(propagated.act_bias, state.act_bias)
    np.testing.assert_array_equal(propagated.sens_bias, state.sens_bias)
    np.testing.assert_array_equal(propagated.dist_param, state.dist_param)
    np.testing.assert_array_equal(propagated.cov, state.cov)
    np.testing.assert_array_equal(propagated.int_cov, state.int_cov)


def test_cg5_propagation_lets_satellite_build_stage_orbital_states(orbital_state):
    satellite = _RecordingSatellite()
    state = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])

    propagate_state(
        state,
        satellite,
        np.zeros(0),
        0.01,
        orbital_state,
        orbital_state,
        midpoint_orbital_state=orbital_state,
        quaternion_integrator="cg5",
    )

    assert satellite.quat_as_vec is False
    assert satellite.mid_orbital_state is None


def test_cg5_propagation_accepts_five_stage_orbital_states(orbital_state):
    satellite = _RecordingSatellite()
    state = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])
    stages = [orbital_state] * 5

    propagate_state(
        state,
        satellite,
        np.zeros(0),
        0.01,
        orbital_state,
        orbital_state,
        midpoint_orbital_state=stages,
        quaternion_integrator="cg5",
    )

    assert satellite.mid_orbital_state is stages


def test_cg5_propagation_rejects_wrong_number_of_stage_orbital_states(orbital_state):
    satellite = _RecordingSatellite()
    state = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="five stage states"):
        propagate_state(
            state,
            satellite,
            np.zeros(0),
            0.01,
            orbital_state,
            orbital_state,
            midpoint_orbital_state=[orbital_state] * 4,
            quaternion_integrator="cg5",
        )


@pytest.mark.parametrize(
    "wheel_count", [0, 1, 3], ids=["no_wheel", "one_wheel", "three_wheels"]
)
def test_discrete_transition_matches_nonlinear_propagation(
    orbital_state, wheel_count
):
    if wheel_count:
        actuators = [
            RW(
                axis=np.eye(3)[index],
                max_torque=0.01,
                J=0.001,
                h=0.02,
                h_max=0.1,
            )
            for index in range(wheel_count)
        ]
        state = _estimate(wheel_momentum=np.full(wheel_count, 0.02))
    else:
        actuators = [MTQ(axis=np.array([1.0, 0.0, 0.0]), max_torque=1.0)]
        state = _estimate()
    satellite = EstimatedSatellite(
        J_0=np.diag([0.5, 0.8, 1.2]),
        actuators=actuators,
    )
    dt = 0.01

    analytic, _ = discretize_process_noise(
        state,
        satellite,
        np.zeros(satellite.control_len),
        orbital_state,
        dt,
        unmodeled_dynamics_psd=0.0,
    )
    numerical = _numerical_discrete_transition(satellite, state, orbital_state, dt)

    np.testing.assert_allclose(analytic, numerical, rtol=2.0e-6, atol=2.0e-6)


def test_builtin_measurement_stack_jacobian_matches_local_finite_difference(
    orbital_state,
):
    star = SimpleNamespace(
        s_eci=np.array([0.3, -0.4, np.sqrt(0.75)]),
        vmag=1.0,
    )
    vector_tracker = StarTracker(fov=np.pi)
    vector_tracker._select_star = lambda q, os: star
    quaternion_tracker = StarTrackerQuaternion(fov=np.pi, min_stars=1)
    quaternion_tracker._select_stars = lambda q, os: [star]
    sensors = [
        Gyro(axis=np.array([1.0, 2.0, -1.0])),
        MTM(axis=np.array([-1.0, 2.0, 1.0])),
        SunSensor(axis=np.array([1.0, 0.2, -0.1]), efficiency=0.8),
        SunPair(axis=np.array([0.2, 1.0, 0.3]), efficiency=(0.7, 0.9)),
        vector_tracker,
        quaternion_tracker,
        EarthHorizonSensor(boresight=np.array([-1.0, 0.0, 0.0]), fov=np.pi),
    ]
    stack = EstimatedSatellite(sensors=sensors).measurement_stack
    state = _estimate()
    active = np.ones(len(stack.entries), dtype=bool)
    predicted = stack.predict(state, orbital_state, active)
    analytic = stack.jacobian(state, orbital_state, active)

    epsilon = 1.0e-7
    numerical = np.empty_like(analytic)
    for column in range(state.tangent_size):
        offset = np.zeros(state.tangent_size)
        offset[column] = epsilon
        plus = stack.predict(state.plus(offset), orbital_state, active)
        minus = stack.predict(state.plus(-offset), orbital_state, active)
        plus_local = stack.residual(plus, predicted, active)
        minus_local = stack.residual(minus, predicted, active)
        numerical[:, column] = (plus_local - minus_local) / (2.0 * epsilon)

    np.testing.assert_allclose(analytic, numerical, rtol=2.0e-5, atol=2.0e-8)


def test_quaternion_innovation_and_correction_have_the_same_sign():
    tracker = StarTrackerQuaternion(noise=Noise(std_noise=np.full(4, 1.0e-3)))
    stack = EstimatedSatellite(sensors=[tracker]).measurement_stack
    predicted = _estimate()
    true_delta = np.zeros(predicted.tangent_size)
    true_delta[predicted.tangent_slices["attitude"]] = [0.04, -0.02, 0.01]
    truth = predicted.plus(true_delta)
    active = np.array([True])

    innovation = stack.residual(truth.q, predicted.q, active)
    H = stack.jacobian(predicted, None, active)
    R = stack.covariance(predicted, active)
    prior = Covariance.identity(predicted.tangent_size, scale=0.1)
    gain, posterior = prior.updated_linear(H, R)
    corrected = predicted.plus(gain @ innovation)

    np.testing.assert_allclose(innovation, true_delta[3:6], atol=1.0e-14)
    assert np.linalg.norm(truth.minus(corrected)) < np.linalg.norm(
        truth.minus(predicted)
    )
    assert np.isclose(np.linalg.norm(corrected.q), 1.0)
    np.testing.assert_allclose(posterior.as_matrix(), posterior.as_matrix().T)
    assert np.linalg.eigvalsh(posterior.as_matrix()).min() >= -1.0e-14


def test_every_measurement_subset_has_consistent_compact_dimensions():
    tracker = StarTrackerQuaternion(noise=Noise(std_noise=np.full(4, 0.1)))
    tracker.clean_reading = lambda state, os: state.q.copy()
    wheel = RW(
        axis=np.array([1.0, 0.0, 0.0]),
        max_torque=0.1,
        J=0.01,
        h=0.2,
        h_max=1.0,
        h_meas_noise=Noise(std_noise=0.2),
    )
    satellite = EstimatedSatellite(
        sensors=[
            Gyro(
                axis=np.array([1.0, 0.0, 0.0]),
                noise=Noise(std_noise=0.1),
            ),
            tracker,
        ],
        actuators=[wheel],
    )
    stack = satellite.measurement_stack
    state = _estimate(wheel_momentum=[0.2])
    measured = stack.predict(state, None)

    residual_sizes = [1, 3, 1]
    for selected in product((False, True), repeat=len(stack.entries)):
        active = np.asarray(selected)
        predicted = stack.predict(state, None, active)
        expected_size = sum(size for size, use in zip(residual_sizes, active) if use)

        assert stack.residual(measured, predicted, active).shape == (expected_size,)
        assert stack.jacobian(state, None, active).shape == (
            expected_size,
            state.tangent_size,
        )
        assert stack.covariance(state, active).shape == (expected_size, expected_size)


def test_linear_ekf_primitives_match_in_full_and_square_root_forms():
    initial = np.array([[2.0, 0.3], [0.3, 1.0]])
    transition = np.array([[1.0, 0.2], [0.0, 1.0]])
    process_noise = np.diag([0.1, 0.2])
    measurement_jacobian = np.array([[1.0, -0.5]])
    measurement_noise = np.array([[0.3]])
    expected_prediction = transition @ initial @ transition.T + process_noise
    innovation_covariance = (
        measurement_jacobian @ expected_prediction @ measurement_jacobian.T
        + measurement_noise
    )
    expected_gain = (
        expected_prediction
        @ measurement_jacobian.T
        @ np.linalg.inv(innovation_covariance)
    )
    residual = np.eye(2) - expected_gain @ measurement_jacobian
    expected_posterior = (
        residual @ expected_prediction @ residual.T
        + expected_gain @ measurement_noise @ expected_gain.T
    )

    results = []
    for form in ("full", "sqrt"):
        predicted = Covariance(initial, form=form).predicted_linear(
            transition, process_noise
        )
        gain, posterior = predicted.updated_linear(
            measurement_jacobian, measurement_noise
        )
        results.append((gain, posterior.as_matrix()))

    np.testing.assert_allclose(results[0][0], results[1][0])
    np.testing.assert_allclose(results[0][1], results[1][1])
    np.testing.assert_allclose(results[0][0], expected_gain)
    np.testing.assert_allclose(results[0][1], expected_posterior)
    np.testing.assert_allclose(results[0][1], results[0][1].T)
    assert np.linalg.eigvalsh(results[0][1]).min() >= -1.0e-14


def test_cg5_propagation_runs_on_a_real_satellite_without_stage_states(orbital_state):
    """The satellite-built stage path must work on a real satellite.

    The other ``cg5`` contracts use ``_RecordingSatellite``, which never calls
    the integrator, so the branch that builds its own stage orbital states was
    never executed against real code.
    """
    satellite = EstimatedSatellite(
        J_0=np.diag([0.5, 0.8, 1.2]),
        actuators=[MTQ(axis=np.array([1.0, 0.0, 0.0]), max_torque=1.0)],
    )
    state = EstimatorState(w=[0.02, -0.01, 0.015], q=[0.9, 0.2, -0.3, 0.1]).normalized()

    propagated = propagate_state(
        state,
        satellite,
        np.zeros(1),
        0.5,
        orbital_state,
        orbital_state,
        quaternion_integrator="cg5",
    )

    assert np.isclose(np.linalg.norm(propagated.q), 1.0)
    assert np.all(np.isfinite(propagated.as_array()))


def test_cg5_and_rk4_agree_on_a_short_step(orbital_state):
    satellite = EstimatedSatellite(
        J_0=np.diag([0.5, 0.8, 1.2]),
        actuators=[MTQ(axis=np.array([1.0, 0.0, 0.0]), max_torque=1.0)],
    )
    state = EstimatorState(w=[0.02, -0.01, 0.015], q=[0.9, 0.2, -0.3, 0.1]).normalized()

    rk4 = propagate_state(
        state, satellite, np.zeros(1), 0.5, orbital_state, orbital_state,
        quaternion_integrator="rk4",
    )
    cg5 = propagate_state(
        state, satellite, np.zeros(1), 0.5, orbital_state, orbital_state,
        quaternion_integrator="cg5",
    )

    assert np.allclose(cg5.q, rk4.q, atol=1.0e-5)
    assert np.allclose(cg5.w, rk4.w, atol=1.0e-7)


@pytest.mark.parametrize("initial_bias", [0.0, 1.0e-3])
@pytest.mark.parametrize("bias_rate", [0.0, 1.0e-6])
@pytest.mark.parametrize("estimate_bias", [False, True])
def test_sensor_bias_state_and_jacobian_agree_in_every_quadrant(
    orbital_state, initial_bias, bias_rate, estimate_bias
):
    """Bias-state allocation and Jacobian width must key on the same thing.

    ``Bias.__bool__`` is value-dependent (false when both the offset and the
    rate are zero), so gating a *shape* on it disagrees with the layout, which
    keys on ``estimate_bias``. All four quadrants of (bias present, estimated)
    must stay consistent.
    """
    gyro = Gyro(
        axis=np.array([1.0, 0.0, 0.0]),
        noise=Noise(noise=0.0, std_noise=0.0),
        bias=Bias(bias=np.array([initial_bias]), std_bias=np.array([bias_rate])),
        estimate_bias=estimate_bias,
    )
    satellite = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=[gyro])
    state = EstimatorState(
        w=[0.05, 0.0, 0.0],
        q=[1.0, 0.0, 0.0, 0.0],
        sens_bias=np.zeros(satellite.att_sens_bias_len),
    )

    expected = 1 if estimate_bias else 0
    assert satellite.att_sens_bias_len == expected
    assert state.sens_bias.size == expected

    stack = satellite.measurement_stack
    jacobian = stack.jacobian(state, orbital_state, np.ones(len(stack.entries), dtype=bool))
    assert jacobian.shape[1] == state.tangent_size

    bias_columns = jacobian[:, state.slice("sensor_bias", coordinates="tangent")]
    assert bias_columns.shape[1] == expected
    if estimate_bias:
        # The bias must be a live parameter, not a dead state.
        assert np.any(bias_columns != 0.0)


@pytest.mark.parametrize("initial_bias", [0.0, 1.0e-3])
@pytest.mark.parametrize("bias_rate", [0.0, 1.0e-6])
@pytest.mark.parametrize("estimate_bias", [False, True])
def test_actuator_bias_state_and_jacobian_agree_in_every_quadrant(
    orbital_state, initial_bias, bias_rate, estimate_bias
):
    """Same contract for actuators, whose Jacobians previously gated on truthiness.

    ``estimate_bias=True`` with an identically zero bias (the default) used to
    return a zero-row Jacobian while the state allocated a bias element, which
    raised inside ``dynJacCore``.
    """
    actuators = [
        MTQ(
            axis=axis,
            max_torque=1.0,
            noise=Noise(noise=0.0, std_noise=0.0),
            bias=Bias(bias=np.array([initial_bias]), std_bias=np.array([bias_rate])),
            estimate_bias=estimate_bias,
        )
        for axis in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    ]
    satellite = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), actuators=actuators)
    state = EstimatorState(
        w=[0.02, -0.01, 0.015],
        q=[1.0, 0.0, 0.0, 0.0],
        act_bias=np.zeros(satellite.act_bias_len),
    )

    expected = 3 if estimate_bias else 0
    assert satellite.act_bias_len == expected
    assert state.act_bias.size == expected

    blocks = satellite.dynJacCore(state, np.zeros(3), orbital_state)
    assert np.asarray(blocks[2]).shape[0] == expected
