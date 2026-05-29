import numpy as np
import pytest

from testing.test_estimators.srukf.helpers import (
    make_mtqs,
    make_orbital_state,
    make_satellites,
    make_state,
    make_srukf,
    sensor_family,
)


@pytest.mark.parametrize(
    ("family", "expected_dim"),
    [
        ("gyro", 3),
        ("mtm", 3),
        ("sunpair", 3),
        ("sunsensor", 3),
        ("earth_horizon", 3),
        ("star_tracker", 3),
        ("star_tracker_quaternion", 4),
    ],
)
def test_sensor_family_builds_and_matches_covariance_dimensions(monkeypatch, family, expected_dim):
    real_sensors = sensor_family(family)
    est_sensors = sensor_family(family)
    real_sat, est_sat = make_satellites(sensors=real_sensors, estimated_sensors=est_sensors, actuators=make_mtqs(), estimated_actuators=make_mtqs())
    srukf = make_srukf(est_sat)
    os = make_orbital_state()
    x_true = make_state()
    if family == "star_tracker":
        monkeypatch.setattr(real_sat.attitude_sensors[0], "reading", lambda x, os, dmode=None: np.array([0.0, 0.0, 1.0]))
        monkeypatch.setattr(est_sat.attitude_sensors[0], "reading", lambda x, os, dmode=None: np.array([0.0, 0.0, 1.0]))
    if family == "star_tracker_quaternion":
        monkeypatch.setattr(real_sat.attitude_sensors[0], "reading", lambda x, os, dmode=None: np.array([1.0, 0.0, 0.0, 0.0]))
        monkeypatch.setattr(est_sat.attitude_sensors[0], "reading", lambda x, os, dmode=None: np.array([1.0, 0.0, 0.0, 0.0]))
    sensors = real_sat.noiseless_sensor_readings(x_true, os)
    cov = est_sat.sensor_cov([True] * len(est_sat.attitude_sensors))
    srukf.update(u=np.zeros(len(real_sat.actuators)), sensors=sensors, os=os)
    assert sensors.size == expected_dim
    assert cov.shape == (expected_dim, expected_dim)
    assert np.isfinite(srukf.x_hat.val).all()


def test_rw_measurements_append_to_sensor_configuration_dimension():
    real_sensors = sensor_family("gyro")
    est_sensors = sensor_family("gyro")
    from testing.test_estimators.srukf.helpers import make_rws
    real_sat, est_sat = make_satellites(
        sensors=real_sensors,
        estimated_sensors=est_sensors,
        actuators=make_mtqs() + make_rws(),
        estimated_actuators=make_mtqs() + make_rws(),
    )
    srukf = make_srukf(est_sat)
    sensors = real_sat.noiseless_sensor_readings(make_state(h=np.array([1.0, 1.0, 1.0])), make_orbital_state())
    cov = est_sat.sensor_cov([True] * len(est_sat.attitude_sensors))
    srukf.update(u=np.zeros(len(real_sat.actuators)), sensors=sensors, os=make_orbital_state())
    assert sensors.size == 6
    assert cov.shape == (6, 6)
