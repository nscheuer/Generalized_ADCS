import numpy as np
import pytest

from testing.test_estimators.ukf.helpers import make_baseline_sensors, make_satellites, make_ukf
from testing.test_estimators.ukf.scenarios import bias_scenario


def test_gyro_bias_states_are_included_when_requested():
    sensors = make_baseline_sensors(estimate_gyro_bias=True)
    _, est_sat = make_satellites(sensors=sensors, estimated_sensors=sensors)

    assert est_sat.att_sens_bias_len == 3
    assert len(est_sat.att_sens_bias_inds) == 3


def test_match_estimate_synchronizes_gyro_biases_into_estimated_satellite():
    sensors = make_baseline_sensors(estimate_gyro_bias=True)
    _, est_sat = make_satellites(sensors=sensors, estimated_sensors=sensors)
    ukf = make_ukf(est_sat)
    ukf.x_hat.val[7:10] = np.array([9.0e-4, -7.0e-4, 5.0e-4])

    est_sat.match_estimate(ukf.x_hat, ukf.dt)

    gyro_biases = np.array([sensor.bias.bias.item() for sensor in est_sat.attitude_sensors[3:6]])
    assert np.allclose(gyro_biases, np.array([9.0e-4, -7.0e-4, 5.0e-4]))


def test_short_bias_sequence_moves_bias_estimate_toward_truth():
    result = bias_scenario()
    true_bias = np.array([8.0e-4, -6.0e-4, 5.0e-4])
    initial = np.linalg.norm(result.estimate[0, 7:10] - true_bias)
    final = np.linalg.norm(result.estimate[-1, 7:10] - true_bias)

    assert final < 2.0e-3
    assert final < max(initial, 1.0e-2)



def test_bias_convergence_scenario_recovers_final_bias():
    result = bias_scenario()
    true_bias = np.array([8.0e-4, -6.0e-4, 5.0e-4])
    final_error = np.abs(result.estimate[-1, 7:10] - true_bias)

    assert np.all(final_error < 1.5e-3)
