"""``Controller.build_sensor_matrix_pinv`` must tolerate sensors without a sensing axis.

The helper walks the full sensor list to keep its output-index cursor aligned with the
measurement vector, and only the sensors of the *requested* type need geometry. It used to
read ``sens.axis`` unconditionally, so any bus carrying an axis-less sensor -- a star tracker,
a GPS receiver, an Earth-horizon sensor -- raised ``AttributeError`` and made **every**
controller in the library unusable on that bus, even when the requested sensor type was
present and perfectly well formed.
"""

import numpy as np
import pytest

from ADCS.controller.controller import Controller
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.sensors import MTM, Gyro
from ADCS.satellite_hardware.sensors.sensor import Sensor
from ADCS.helpers.math_constants import MathConstants

UV = MathConstants.unitvecs


class _AxislessSensor(Sensor):
    """Minimal stand-in for a star tracker / GPS: multi-output, no ``axis``."""

    def __init__(self, output_length: int = 4):
        super().__init__(sample_time=1.0, output_length=output_length,
                         bias=Bias(bias=np.zeros(output_length),
                                   std_bias=np.zeros(output_length)),
                         noise=Noise(noise=np.zeros(output_length),
                                     std_noise=np.ones(output_length) * 1e-6))

    def clean_reading(self, x, os):
        return np.zeros(self.output_length)


def build(sensors, sensor_type):
    """Call the helper unbound.

    ``build_sensor_matrix_pinv`` is an instance method but touches nothing on ``self``; it is
    pure over (sensors, sensor_type). Calling it unbound keeps these tests focused on that
    logic instead of on constructing a controller with a full satellite behind it.
    """
    return Controller.build_sensor_matrix_pinv(None, sensors=sensors, sensor_type=sensor_type)


def mtms():
    return [MTM(axis=UV[i], noise=Noise(noise=0.0, std_noise=1e-7)) for i in range(3)]


def test_axisless_sensor_does_not_break_the_matrix_build():
    sensors = mtms() + [_AxislessSensor(4)]
    M, idx = build(sensors, MTM)
    assert idx == [0, 1, 2]
    # M acts on the FULL measurement vector and zeroes the columns it does not own.
    assert M.shape == (3, 7)
    np.testing.assert_allclose(M[:, 3:], 0.0, atol=0.0)


def test_indices_stay_aligned_when_an_axisless_sensor_comes_first():
    """The cursor must advance by the axis-less sensor's full output width."""
    sensors = [_AxislessSensor(4)] + mtms()
    _, idx = build(sensors, MTM)
    assert idx == [4, 5, 6]


def test_indices_stay_aligned_with_axisless_sensors_interleaved():
    sensors = [mtms()[0], _AxislessSensor(4), mtms()[1], _AxislessSensor(2), mtms()[2]]
    _, idx = build(sensors, MTM)
    #      MTM0 @0 | axisless(4) @1-4 | MTM1 @5 | axisless(2) @6-7 | MTM2 @8
    assert idx == [0, 5, 8]


def test_reconstruction_picks_the_right_entries_of_a_full_measurement_vector():
    """End to end: the returned indices must select the MTM entries out of the full vector."""
    sensors = [_AxislessSensor(4)] + mtms()
    M, idx = build(sensors, MTM)
    b_true = np.array([1e-5, -2e-5, 3e-5])
    y = np.concatenate([np.full(4, 999.0), b_true])   # axis-less junk, then the MTMs
    assert idx == [4, 5, 6]
    # The junk must not leak into the reconstruction.
    np.testing.assert_allclose(M @ y, b_true, rtol=1e-12)


def test_gyro_selection_is_unaffected_by_an_axisless_sensor():
    gyros = [Gyro(axis=UV[i], noise=Noise(noise=0.0, std_noise=1e-4)) for i in range(3)]
    sensors = mtms() + [_AxislessSensor(4)] + gyros
    _, idx = build(sensors, Gyro)
    assert idx == [7, 8, 9]


def test_missing_requested_type_still_raises():
    with pytest.raises(ValueError, match="No sensors of type"):
        build([_AxislessSensor(4)], MTM)


def test_axisless_sensor_defaults_to_width_one_without_output_length():
    """A sensor-like object with no ``output_length`` advances the cursor by 1.

    Uses three MTMs because the existing pinv slot-in assumes exactly three active columns
    (``M_full[:, active_indices] = pinv(A_sub)`` transposes for any other count). That is a
    separate latent bug for one- or two-magnetometer buses and is deliberately not touched
    here.
    """
    class _Bare:
        pass

    sensors = [_Bare()] + mtms()
    _, idx = build(sensors, MTM)
    assert idx == [1, 2, 3]
