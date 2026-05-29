import numpy as np
import pytest

from testing.test_estimators.ukf.helpers import make_baseline_sensors, make_mtqs, make_orbital_state, make_rws, make_satellites, make_state, make_ukf
from testing.test_estimators.ukf.scenarios import reaction_wheel_scenario


def test_reaction_wheel_states_extend_estimator_state_layout():
    sensors = make_baseline_sensors()
    _, est_sat = make_satellites(
        sensors=sensors,
        estimated_sensors=make_baseline_sensors(),
        actuators=make_mtqs() + make_rws(),
        estimated_actuators=make_mtqs() + make_rws(),
    )
    ukf = make_ukf(est_sat)

    assert est_sat.number_RW == 3
    assert ukf.x_hat.val.size == 10


def test_reaction_wheel_measurements_are_included_in_sensor_covariance():
    sensors = make_baseline_sensors()
    real_sat, est_sat = make_satellites(
        sensors=sensors,
        estimated_sensors=make_baseline_sensors(),
        actuators=make_mtqs() + make_rws(),
        estimated_actuators=make_mtqs() + make_rws(),
    )
    cov = est_sat.sensor_cov([True] * len(est_sat.attitude_sensors))
    sensors_vec = real_sat.noiseless_sensor_readings(make_state(h=np.array([1.0, 1.0, 1.0])), make_orbital_state())

    assert cov.shape == (12, 12)
    assert sensors_vec.size == 12


def test_one_step_update_uses_rw_measurements_and_keeps_state_finite():
    sensors = make_baseline_sensors()
    real_sat, est_sat = make_satellites(
        sensors=sensors,
        estimated_sensors=make_baseline_sensors(),
        actuators=make_mtqs() + make_rws(h=0.8),
        estimated_actuators=make_mtqs() + make_rws(h=0.0),
    )
    ukf = make_ukf(est_sat)
    x_true = make_state(h=np.array([0.8, 0.8, 0.8]))
    sensors_vec = real_sat.noiseless_sensor_readings(x_true, make_orbital_state())

    ukf.update(u=np.zeros(len(real_sat.actuators)), sensors=sensors_vec, os=make_orbital_state())

    assert np.isfinite(ukf.x_hat.val).all()


def test_estimated_satellite_syncs_rw_momentum_from_estimate():
    sensors = make_baseline_sensors()
    _, est_sat = make_satellites(
        sensors=sensors,
        estimated_sensors=make_baseline_sensors(),
        actuators=make_mtqs() + make_rws(),
        estimated_actuators=make_mtqs() + make_rws(),
    )
    ukf = make_ukf(est_sat)
    ukf.x_hat.val[7:10] = np.array([0.7, 0.8, 0.9])

    est_sat.match_estimate(ukf.x_hat, ukf.dt)

    assert np.allclose(est_sat.RWhs(), np.array([0.7, 0.8, 0.9]))


@pytest.mark.slow
def test_rw_convergence_scenario_tracks_wheel_momentum():
    result = reaction_wheel_scenario()
    final_error = np.linalg.norm(result.estimate[-1, 7:10] - result.truth[-1, 7:10])

    assert final_error < 0.5
