import numpy as np
import pytest

import ADCS.estimators.attitude_estimators.attitude_UAKF as uakf_module
from testing.test_estimators.ukf.helpers import (
    make_baseline_sensors,
    make_orbital_state,
    make_satellites,
    make_state,
    make_ukf,
    quat_error_deg,
)


def test_one_step_update_reduces_attitude_error():
    real_sat, est_sat = make_satellites(sensors=make_baseline_sensors(), estimated_sensors=make_baseline_sensors())
    ukf = make_ukf(est_sat)
    x_true = make_state(q=np.array([0.99, 0.08, -0.03, 0.09]))
    os = make_orbital_state()
    sensors = real_sat.noiseless_sensor_readings(x_true, os)
    initial_error = quat_error_deg(x_true[3:7], ukf.x_hat.val[3:7])

    x_hat = ukf.update(u=np.zeros(len(real_sat.actuators)), sensors=sensors, os=os)
    final_error = quat_error_deg(x_true[3:7], x_hat[3:7])

    assert final_error < initial_error


def test_update_with_finite_measurements_keeps_state_and_covariance_finite():
    real_sat, est_sat = make_satellites(sensors=make_baseline_sensors(), estimated_sensors=make_baseline_sensors())
    ukf = make_ukf(est_sat)
    sensors = real_sat.noiseless_sensor_readings(make_state(), make_orbital_state())

    ukf.update(u=np.zeros(len(real_sat.actuators)), sensors=sensors, os=make_orbital_state())

    assert np.isfinite(ukf.x_hat.val).all()
    assert np.isfinite(ukf.x_hat.cov).all()
    assert np.isclose(np.linalg.norm(ukf.x_hat.val[3:7]), 1.0)


def test_update_handles_mixed_active_and_inactive_sensors():
    real_sat, est_sat = make_satellites(sensors=make_baseline_sensors(), estimated_sensors=make_baseline_sensors())
    ukf = make_ukf(est_sat)
    os = make_orbital_state()
    sensors = real_sat.noiseless_sensor_readings(make_state(), os)
    sensors[3:6] = np.nan

    ukf.update(u=np.zeros(len(real_sat.actuators)), sensors=sensors, os=os)

    assert np.isfinite(ukf.x_hat.val).all()


def test_update_raises_linalgerror_when_solver_fails(monkeypatch):
    real_sat, est_sat = make_satellites(sensors=make_baseline_sensors(), estimated_sensors=make_baseline_sensors())
    ukf = make_ukf(est_sat)
    sensors = real_sat.noiseless_sensor_readings(make_state(), make_orbital_state())

    monkeypatch.setattr(uakf_module.scipy.linalg, "solve", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("boom")))

    with pytest.raises(np.linalg.LinAlgError):
        ukf.update(u=np.zeros(len(real_sat.actuators)), sensors=sensors, os=make_orbital_state())
