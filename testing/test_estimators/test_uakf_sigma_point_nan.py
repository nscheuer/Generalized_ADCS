"""UAKF must survive a sensor whose availability depends on the state.

``update_core`` deactivates sensors whose **actual** reading is NaN. That is not sufficient
for any sensor with a state-dependent availability envelope -- a star tracker with sun or
Earth-limb keep-outs, or a slew-rate limit. The sigma points deliberately explore attitudes
the true state is not at, so individual sigma points can fall outside the envelope and return
NaN while the real measurement is perfectly valid. Those NaNs reached ``covyy`` and the gain
solve failed as a bare ``LinAlgError("Matrix is singular. (probably)")``, which is both a crash
and a misleading diagnosis.
"""

import numpy as np
import pytest

from ADCS.estimators.attitude_estimators import UAKF
from ADCS.helpers.math_constants import MathConstants
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import MTM, Gyro
from ADCS.satellite_hardware.sensors.sensor import Sensor

UV = MathConstants.unitvecs


class FlakySensor(Sensor):
    """3-output sensor that returns NaN for states outside a narrow attitude window.

    Stands in for a star tracker keep-out: the *truth* is inside the window (so the real
    reading is finite and the sensor is not masked by the existing NaN check), while sigma
    points spread outside it and return NaN.
    """

    # Class-level counters: the UAKF evaluates each sigma point on a COPY of the satellite,
    # so per-instance counters never see the sigma-point calls that this test is about.
    n_nan = 0
    n_calls = 0

    def __init__(self, q_ref, half_angle_rad=1e-8, output_length=3):
        self.q_ref = np.asarray(q_ref, float)
        self.half_angle = float(half_angle_rad)
        super().__init__(
            sample_time=1.0, output_length=output_length,
            bias=Bias(bias=np.zeros(output_length), std_bias=np.zeros(output_length)),
            noise=Noise(noise=np.zeros(output_length),
                        std_noise=np.full(output_length, 1e-6)),
        )

    def clean_reading(self, x, os):
        FlakySensor.n_calls += 1
        q = np.asarray(x, float)[3:7]
        ang = 2.0 * np.arccos(np.clip(abs(float(q @ self.q_ref)), 0.0, 1.0))
        if ang > self.half_angle:
            FlakySensor.n_nan += 1
            return np.full(self.output_length, np.nan)
        return np.zeros(self.output_length)

    @staticmethod
    def reset_counters():
        FlakySensor.n_nan = 0
        FlakySensor.n_calls = 0


def make_sat(extra_sensors):
    sensors = (
        [MTM(axis=UV[i], noise=Noise(noise=0.0, std_noise=1e-7)) for i in range(3)]
        + [Gyro(axis=UV[i], noise=Noise(noise=0.0, std_noise=1e-4),
                bias=Bias(bias=0.0, std_bias=1e-6), estimate_bias=True)
           for i in range(3)]
        + list(extra_sensors)
    )
    return EstimatedSatellite(
        mass=4.0, J_0=np.diagflat([0.13, 0.10, 0.05]),
        actuators=[MTQ(axis=UV[i], max_torque=0.2) for i in range(3)],
        sensors=sensors,
    )


def make_filter(sat, dt=1.0):
    n = 7 + sat.number_RW + sat.act_bias_len + sat.att_sens_bias_len + sat.dist_param_len
    x0 = np.concatenate([np.zeros(3), [1.0, 0.0, 0.0, 0.0], np.zeros(n - 7)])
    red = n - 1
    P = np.diagflat(np.concatenate([[1e-6] * 3, [1e-4] * 3, [1e-8] * (red - 6)]))
    Q = np.eye(red) * 1e-12
    return UAKF(est_sat=sat, J2000=0.22, x_hat=x0, P_hat=P, Q_hat=Q, dt=dt,
                cross_term=True, quat_as_vec=False)


def make_os():
    return Orbital_State(ephem=Ephemeris(), J2000=0.22,
                         R=np.array([6778.0, 0.0, 0.0]),
                         V=np.array([0.0, 7.67, 0.0]))


def test_sigma_point_nans_do_not_break_the_update():
    FlakySensor.reset_counters()
    """The real reading is valid; only sigma points fall outside the sensor's envelope."""
    q_ref = np.array([1.0, 0.0, 0.0, 0.0])
    flaky = FlakySensor(q_ref, half_angle_rad=1e-8)
    sat = make_sat([flaky])
    est = make_filter(sat)
    os_ = make_os()

    # Truth sits exactly at q_ref, so the actual reading is finite and the existing
    # actual-reading NaN mask does NOT fire.
    x_true = np.concatenate([np.zeros(3), q_ref])
    y = sat.sensor_readings(x=x_true, os=os_)
    assert np.all(np.isfinite(y)), "actual reading must be finite for this test to bite"

    out = est.update(u=np.zeros(3), sensors=y, os=os_)

    assert FlakySensor.n_nan > 0, "sigma points should have strayed outside the envelope"
    assert np.all(np.isfinite(np.asarray(out, float)))


def test_repeated_updates_stay_finite():
    q_ref = np.array([1.0, 0.0, 0.0, 0.0])
    flaky = FlakySensor(q_ref, half_angle_rad=1e-8)
    sat = make_sat([flaky])
    est = make_filter(sat)
    os_ = make_os()
    x_true = np.concatenate([np.zeros(3), q_ref])

    for _ in range(25):
        y = sat.sensor_readings(x=x_true, os=os_)
        out = np.asarray(est.update(u=np.zeros(3), sensors=y, os=os_), float)
        assert np.all(np.isfinite(out))
    assert np.all(np.isfinite(est.x_hat.cov))


def test_sensor_with_all_finite_predictions_is_still_used():
    FlakySensor.reset_counters()
    """The guard must not deactivate healthy sensors."""
    always_ok = FlakySensor(np.array([1.0, 0.0, 0.0, 0.0]), half_angle_rad=np.pi)
    sat = make_sat([always_ok])
    est = make_filter(sat)
    os_ = make_os()
    x_true = np.concatenate([np.zeros(3), [1.0, 0.0, 0.0, 0.0]])
    y = sat.sensor_readings(x=x_true, os=os_)
    out = est.update(u=np.zeros(3), sensors=y, os=os_)
    assert FlakySensor.n_nan == 0
    assert np.all(np.isfinite(np.asarray(out, float)))


def test_actual_reading_nan_is_still_masked():
    """The pre-existing behaviour must be preserved: a NaN real reading deactivates."""
    far = np.array([0.0, 1.0, 0.0, 0.0])          # 180 deg from truth -> always NaN
    flaky = FlakySensor(far, half_angle_rad=1e-8)
    sat = make_sat([flaky])
    est = make_filter(sat)
    os_ = make_os()
    x_true = np.concatenate([np.zeros(3), [1.0, 0.0, 0.0, 0.0]])
    y = sat.sensor_readings(x=x_true, os=os_)
    assert np.any(np.isnan(y)), "this sensor should produce a NaN actual reading"
    out = est.update(u=np.zeros(3), sensors=y, os=os_)
    assert np.all(np.isfinite(np.asarray(out, float)))


def test_baseline_without_flaky_sensor_is_unchanged():
    """A bus with no state-dependent sensor must be bit-identical to before the guard."""
    sat = make_sat([])
    est = make_filter(sat)
    os_ = make_os()
    x_true = np.concatenate([np.zeros(3), [1.0, 0.0, 0.0, 0.0]])
    outs = []
    for _ in range(5):
        y = sat.sensor_readings(x=x_true, os=os_)
        outs.append(np.asarray(est.update(u=np.zeros(3), sensors=y, os=os_), float))
    assert all(np.all(np.isfinite(o)) for o in outs)
