import numpy as np

from ADCS.estimators.measurement_stack import MeasurementStack
from ADCS.satellite_hardware.actuators import RW
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import Gyro, StarTrackerQuaternion
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

    assert stack.source_order == ("sensor[0]", "reaction_wheel[0]")
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
