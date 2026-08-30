import numpy as np

from ADCS.estimators.measurement_stack import MeasurementStack
from ADCS.satellite_hardware.actuators import RW
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import Gyro, Sensor, StarTrackerQuaternion
from ADCS.state import EstimatorState, State


def make_state(*, h=(), sens_bias=()):
    return EstimatorState(
        w=np.array([0.1, 0.0, 0.0]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        h=np.asarray(h, dtype=float),
        sens_bias=np.asarray(sens_bias, dtype=float),
    )


def make_wheel():
    return RW(
        axis=np.array([1.0, 0.0, 0.0]),
        max_torque=0.1,
        J=0.01,
        h=0.4,
        h_max=1.0,
        h_meas_noise=Noise(std_noise=0.3),
    )


class _VectorSensor(Sensor):
    def __init__(self, *, estimate_bias=False):
        super().__init__(
            output_length=3,
            noise=Noise(std_noise=np.array([0.1, 0.2, 0.3])),
            bias=Bias(bias=np.zeros(3), std_bias=np.zeros(3)),
            estimate_bias=estimate_bias,
        )
        self.clean_reading_calls = 0

    def clean_reading(self, x, os):
        self.clean_reading_calls += 1
        return np.asarray(x.w, dtype=float)

    def basestate_jac(self, x, os):
        jacobian = np.zeros((7, 3))
        jacobian[:3, :] = np.eye(3)
        return jacobian


class _PredictedAvailabilitySensor(Sensor):
    def __init__(self):
        super().__init__(output_length=1, noise=Noise(std_noise=0.2))

    def clean_reading(self, x, os):
        return np.nan if x.w[0] < 0.0 else x.w[0]


def test_stack_owns_sensor_then_wheel_order_and_additive_ekf_blocks():
    gyro = Gyro(
        axis=np.array([1.0, 0.0, 0.0]),
        noise=Noise(std_noise=0.2),
        bias=Bias(std_bias=0.0),
        estimate_bias=True,
    )
    satellite = EstimatedSatellite(sensors=[gyro], actuators=[make_wheel()])
    stack = satellite.measurement_stack
    state = make_state(h=[0.3], sens_bias=[0.2])

    assert stack.source_order == ("gyro[0]", "reaction_wheel[0]")
    np.testing.assert_allclose(stack.predict(state, None), [0.3, 0.3])
    mask = stack.active_mask([0.7, 0.4])
    assert np.array_equal(mask, [True, True])
    np.testing.assert_allclose(stack.residual([0.7, 0.4], [0.3, 0.3], mask), [0.4, 0.1])
    np.testing.assert_allclose(stack.covariance(state, mask).as_matrix(), np.diag([0.04, 0.09]))

    H = stack.jacobian(state, None, mask)
    assert H.shape == (2, state.tangent_size)
    np.testing.assert_allclose(H[0, [0, state.tangent_slices["sensor_bias"].start]], [1.0, 1.0])
    assert H[1, state.tangent_slices["wheel_momentum"].start] == 1.0


def test_stack_nan_mask_and_multirate_selection_are_entry_wise():
    sensors = [
        Gyro(axis=np.array([1.0, 0.0, 0.0]), sample_time=0.5),
        Gyro(axis=np.array([0.0, 1.0, 0.0]), sample_time=1.0),
    ]
    stack = MeasurementStack(EstimatedSatellite(sensors=sensors))

    np.testing.assert_array_equal(stack.active_mask([0.1, np.nan]), [True, False])
    np.testing.assert_array_equal(stack.active_mask([0.1, 0.2], time_s=0.5), [True, False])
    np.testing.assert_array_equal(stack.active_mask([0.1, 0.2], time_s=1.0), [True, True])
    np.testing.assert_allclose(stack.active_measurements([0.1, 0.2], [False, True]), [0.2])


def test_numeric_entry_mask_is_treated_like_bool_mask():
    sensors = [
        Gyro(axis=np.array([1.0, 0.0, 0.0])),
        Gyro(axis=np.array([0.0, 1.0, 0.0])),
    ]
    stack = MeasurementStack(EstimatedSatellite(sensors=sensors))
    state = make_state()

    numeric_mask = [1, 0]
    bool_mask = [True, False]

    np.testing.assert_array_equal(stack._entry_mask(numeric_mask), bool_mask)
    np.testing.assert_array_equal(stack._entry_mask([1, 1]), [True, True])
    np.testing.assert_array_equal(stack._entry_mask([0, 0]), [False, False])
    np.testing.assert_allclose(
        stack.active_measurements([0.1, 0.2], numeric_mask),
        stack.active_measurements([0.1, 0.2], bool_mask),
    )
    np.testing.assert_array_equal(
        np.isnan(stack.predict(state, None, active_mask=numeric_mask)),
        [False, True],
    )


def test_sources_are_identifiable_and_semantically_selectable():
    gyros = [
        Gyro(axis=np.array([1.0, 0.0, 0.0])),
        Gyro(axis=np.array([0.0, 1.0, 0.0])),
    ]
    vector = _VectorSensor()
    wheel = make_wheel()
    stack = MeasurementStack(
        EstimatedSatellite(sensors=[*gyros, vector], actuators=[wheel])
    )

    assert stack.source_order == (
        "gyro[0]",
        "gyro[1]",
        "vector_sensor[0]",
        "reaction_wheel[0]",
    )
    assert stack.entry("gyro[1]").source is gyros[1]
    assert stack.entry("sensor[1]").name == "gyro[1]"
    assert stack.entry(wheel).kind == "reaction_wheel"
    assert stack.entry(2).raw_size == 3
    with np.testing.assert_raises_regex(ValueError, "matched 2"):
        stack.entry(Gyro)

    np.testing.assert_array_equal(stack.mask("gyro"), [True, True, False, False])
    np.testing.assert_array_equal(
        stack.mask("sensors", exclude=["gyro[0]", vector]),
        [False, True, False, False],
    )
    np.testing.assert_array_equal(
        stack.mask(Gyro, "reaction_wheels"),
        [True, True, False, True],
    )
    assert tuple(entry.name for entry in stack.selected("gyro")) == (
        "gyro[0]",
        "gyro[1]",
    )
    np.testing.assert_array_equal(
        stack.raw_mask("vector_sensor"),
        [False, False, True, True, True, False],
    )
    np.testing.assert_array_equal(
        stack.residual_mask(["gyro[1]", "reaction_wheels"]),
        [False, True, False, False, False, True],
    )


def test_semantic_selection_works_anywhere_an_entry_mask_is_accepted():
    sensors = [
        Gyro(axis=np.array([1.0, 0.0, 0.0])),
        Gyro(axis=np.array([0.0, 1.0, 0.0])),
    ]
    stack = MeasurementStack(EstimatedSatellite(sensors=sensors))
    state = make_state()

    active = stack.active_mask([0.1, 0.2], enabled="gyro[1]")
    predicted = stack.predict(state, None, active_mask="gyro[1]")

    np.testing.assert_array_equal(active, [False, True])
    np.testing.assert_array_equal(np.isnan(predicted), [True, False])
    assert stack.jacobian(state, None, "gyro[1]").shape == (1, state.tangent_size)


def test_quaternion_measurement_uses_three_dimensional_manifold_residual():
    tracker = StarTrackerQuaternion(noise=Noise(std_noise=np.full(4, 0.2)))
    stack = MeasurementStack(EstimatedSatellite(sensors=[tracker]))
    state = make_state()
    measured = State.quaternion_delta_from_vector([0.1, -0.05, 0.02])
    predicted = np.array([1.0, 0.0, 0.0, 0.0])
    mask = stack.active_mask(measured)

    np.testing.assert_allclose(stack.residual(measured, predicted, mask), [0.1, -0.05, 0.02])
    np.testing.assert_allclose(stack.covariance(state, mask).as_matrix(), np.eye(3) * 0.16)
    np.testing.assert_allclose(
        stack.jacobian(state, None, mask)[:, state.tangent_slices["attitude"]], np.eye(3)
    )


def test_all_inactive_has_empty_residual_covariance_and_jacobian_without_prediction_call():
    sensor = _VectorSensor()
    stack = MeasurementStack(EstimatedSatellite(sensors=[sensor]))
    state = make_state()
    mask = np.array([False])

    predicted = stack.predict(state, None, active_mask=mask)

    np.testing.assert_array_equal(np.isnan(predicted), [True, True, True])
    assert sensor.clean_reading_calls == 0
    assert stack.residual([1.0, 2.0, 3.0], predicted, mask).shape == (0,)
    assert stack.covariance(state, mask).shape == (0, 0)
    assert stack.jacobian(state, None, mask).shape == (0, state.tangent_size)


def test_three_vector_additive_sensor_uses_sensor_bias_layout_and_jacobian():
    sensor = _VectorSensor(estimate_bias=True)
    satellite = EstimatedSatellite(sensors=[sensor])
    stack = satellite.measurement_stack
    state = make_state(sens_bias=[0.4, -0.2, 0.1])
    mask = np.array([True])

    np.testing.assert_allclose(stack.predict(state, None), state.w + state.sens_bias)
    H = stack.jacobian(state, None, mask)
    np.testing.assert_allclose(H[:, :3], np.eye(3))
    np.testing.assert_allclose(H[:, state.tangent_slices["sensor_bias"]], np.eye(3))
    assert satellite.sensor_bias_slice(0) == state.full_slices["sensor_bias"]


def test_reaction_wheel_entry_can_be_masked_out_of_all_compact_outputs():
    gyro = Gyro(axis=np.array([1.0, 0.0, 0.0]), noise=Noise(std_noise=0.2))
    stack = MeasurementStack(EstimatedSatellite(sensors=[gyro], actuators=[make_wheel()]))
    state = make_state(h=[0.3])
    mask = np.array([True, False])
    predicted = stack.predict(state, None, active_mask=mask)

    np.testing.assert_allclose(stack.residual([0.2, 9.0], predicted, mask), [0.1])
    np.testing.assert_allclose(stack.covariance(state, mask).as_matrix(), [[0.04]])
    assert stack.jacobian(state, None, mask).shape == (1, state.tangent_size)


def test_nonfinite_prediction_is_removed_from_effective_mask():
    stack = MeasurementStack(
        EstimatedSatellite(sensors=[_PredictedAvailabilitySensor()])
    )
    state = EstimatorState(w=[-0.1, 0.0, 0.0], q=[1.0, 0.0, 0.0, 0.0])
    measured = np.array([0.3])
    candidate = stack.active_mask(measured)
    predicted = stack.predict(state, None, active_mask=candidate)
    effective = stack.active_mask(measured, predicted=predicted)

    np.testing.assert_array_equal(candidate, [True])
    np.testing.assert_array_equal(effective, [False])
    assert stack.residual(measured, predicted, effective).size == 0
    assert stack.covariance(state, effective).shape == (0, 0)


def test_quaternion_tracker_unavailable_at_estimate_is_masked_without_residual_error():
    tracker = StarTrackerQuaternion(noise=Noise(std_noise=np.full(4, 0.2)))
    tracker._select_stars = lambda q, os: []
    stack = MeasurementStack(EstimatedSatellite(sensors=[tracker]))
    state = make_state()
    measured = np.array([1.0, 0.0, 0.0, 0.0])
    candidate = stack.active_mask(measured)
    predicted = stack.predict(state, None, active_mask=candidate)
    effective = stack.active_mask(measured, predicted=predicted)

    np.testing.assert_array_equal(np.isnan(predicted), [True, True, True, True])
    np.testing.assert_array_equal(effective, [False])
    assert stack.residual(measured, predicted, effective).size == 0


def test_quaternion_tracker_right_error_jacobian_at_nonidentity_attitude():
    tracker = StarTrackerQuaternion(noise=Noise(std_noise=np.full(4, 0.2)))
    stack = MeasurementStack(EstimatedSatellite(sensors=[tracker]))
    q = np.array([0.8, 0.2, -0.3, 0.45])
    q /= np.linalg.norm(q)
    state = EstimatorState(w=np.zeros(3), q=q)
    attitude_delta = np.array([0.03, -0.02, 0.01])
    delta = np.zeros(state.tangent_size)
    delta[state.tangent_slices["attitude"]] = attitude_delta
    measured = state.plus(delta, quaternion_order="right").q
    mask = np.array([True])

    np.testing.assert_allclose(
        stack.residual(measured, state.q, mask), attitude_delta, atol=1e-14
    )
    np.testing.assert_allclose(
        stack.jacobian(state, None, mask)[:, state.tangent_slices["attitude"]],
        np.eye(3),
    )


def test_additive_jacobian_matches_predict_finite_difference():
    stack = MeasurementStack(EstimatedSatellite(sensors=[_VectorSensor()]))
    state = make_state()
    mask = np.array([True])
    analytic = stack.jacobian(state, None, mask)
    numerical = np.zeros_like(analytic)
    epsilon = 1e-7
    for column in range(state.tangent_size):
        offset = np.zeros(state.tangent_size)
        offset[column] = epsilon
        plus = stack.predict(state.plus(offset), None)
        minus = stack.predict(state.plus(-offset), None)
        numerical[:, column] = (plus - minus) / (2.0 * epsilon)

    np.testing.assert_allclose(analytic, numerical, atol=1e-10)


def test_accumulated_decimal_time_remains_due_in_seconds():
    sensor = Gyro(axis=np.array([1.0, 0.0, 0.0]), sample_time=0.1)
    stack = MeasurementStack(EstimatedSatellite(sensors=[sensor]))
    accumulated = sum(0.1 for _ in range(100_000))

    np.testing.assert_array_equal(
        stack.active_mask([0.0], time_s=accumulated), [True]
    )


def test_satellite_caches_measurement_stack():
    satellite = EstimatedSatellite(sensors=[_VectorSensor()])
    assert satellite.measurement_stack is satellite.measurement_stack
