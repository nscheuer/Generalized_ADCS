import numpy as np
import matplotlib.pyplot as plt
import pytest

from ADCS.estimators.orbit_estimators import Orbit_EKF
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import GPS

from testing.test_orbits._helpers import make_reference_orbital_state


def make_estimator(gps_sensors=None, dt=10.0, q_scale=1.0, p_scale=1.0):
    gps_sensors = gps_sensors or [GPS(noise=Noise(noise=np.zeros(6), std_noise=np.ones(6) * 0.01))]
    est_sat = EstimatedSatellite(sensors=gps_sensors)
    os_hat0 = Orbital_State(
        ephem=make_reference_orbital_state().ephem,
        J2000=0.22 - TimeConstants.sec2cent,
        R=np.array([7000.0, 7000.0, 0.0]),
        V=np.array([0.0, 0.0, 8.0]),
        fast=True,
    )
    return Orbit_EKF(
        est_sat=est_sat,
        J2000=os_hat0.J2000,
        os_hat=os_hat0,
        P_hat=np.diag([500.0**2] * 3 + [0.5**2] * 3) * p_scale,
        Q_hat=np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0]) * q_scale,
        dt=dt,
    )


def make_truth_orbit(tf=200.0, dt=10.0):
    os0 = Orbital_State(
        ephem=make_reference_orbital_state().ephem,
        J2000=0.22 - TimeConstants.sec2cent,
        R=7000.0 * np.array([0.0, -np.sqrt(2) / 2.0, -np.sqrt(2) / 2.0]),
        V=np.array([8.0, 0.0, 0.0]),
        fast=True,
    )
    end_time = 0.22 + tf * TimeConstants.sec2cent
    return Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=True, fast=False, verbose=False)


def run_orbit_ekf(tf=200.0, dt=10.0, noisy=True):
    orbit = make_truth_orbit(tf=tf, dt=dt)
    gps = GPS(noise=Noise(noise=np.zeros(6), std_noise=np.ones(6) * 0.01))
    estimator = make_estimator(gps_sensors=[GPS(noise=Noise(noise=np.zeros(6), std_noise=np.ones(6) * 0.01))], dt=dt)

    n = int(tf / dt)
    time_hist = np.zeros(n)
    true_hist = np.zeros((n, 6))
    est_hist = np.zeros((n, 6))
    cov_hist = []

    for k in range(n):
        t = k * dt
        j2000 = 0.22 + t * TimeConstants.sec2cent
        os_true = orbit.get_os(j2000)
        meas = gps.clean_reading(None, os_true)
        if noisy:
            meas = meas + gps.noise.std_noise * np.random.default_rng(123 + k).standard_normal(6)
        est_os = estimator.update([meas], J2000=j2000)

        time_hist[k] = t
        true_hist[k] = np.hstack([os_true.R, os_true.V])
        est_hist[k] = np.hstack([est_os.os.R, est_os.os.V])
        cov_hist.append(est_os.P.copy())

    return time_hist, true_hist, est_hist, cov_hist


def test_orbit_ekf_reset_builds_block_measurement_covariance():
    gps_a = GPS(noise=Noise(noise=np.zeros(6), std_noise=np.ones(6) * 0.1))
    gps_b = GPS(noise=Noise(noise=np.zeros(6), std_noise=np.ones(6) * 0.2))
    ekf = make_estimator(gps_sensors=[gps_a, gps_b])

    expected = np.diag(np.concatenate([np.ones(6) * 0.01, np.ones(6) * 0.04]))
    assert ekf.R.shape == (12, 12)
    assert np.allclose(ekf.R, expected)


def test_orbit_ekf_reset_rejects_bad_covariance_shapes():
    gps = GPS(noise=Noise(noise=np.zeros(6), std_noise=np.ones(6) * 0.01))
    est_sat = EstimatedSatellite(sensors=[gps])
    os_hat0 = make_reference_orbital_state()

    with pytest.raises(ValueError):
        Orbit_EKF(est_sat=est_sat, J2000=os_hat0.J2000, os_hat=os_hat0, P_hat=np.eye(5), Q_hat=np.eye(6), dt=10.0)

    with pytest.raises(ValueError):
        Orbit_EKF(est_sat=est_sat, J2000=os_hat0.J2000, os_hat=os_hat0, P_hat=np.eye(6), Q_hat=np.eye(5), dt=10.0)


def test_orbit_ekf_without_measurements_returns_prediction():
    ekf = make_estimator(dt=10.0)
    prev = ekf.os_hat.copy()

    updated = ekf.update([], J2000=prev.os.J2000 + 10.0 * TimeConstants.sec2cent)

    expected = prev.os.propagate_orbit_rk4(dt=10.0, J2_perturbation_on=True, fast=True)
    assert np.allclose(updated.os.R, expected.R)
    assert np.allclose(updated.os.V, expected.V)
    assert updated.P.shape == (6, 6)
    assert np.allclose(updated.P, updated.P.T)


def test_orbit_ekf_accepts_position_only_measurements():
    ekf = make_estimator(dt=10.0)
    truth = make_reference_orbital_state()
    pos_only = truth.ECEF

    updated = ekf.update([pos_only], J2000=truth.J2000)

    assert updated.os.R.shape == (3,)
    assert updated.os.V.shape == (3,)
    assert np.all(np.isfinite(updated.os.R))
    assert np.all(np.isfinite(updated.os.V))


def test_orbit_ekf_rejects_invalid_measurement_length():
    ekf = make_estimator(dt=10.0)

    with pytest.raises(ValueError):
        ekf.update([np.zeros(4)], J2000=0.22)


def test_orbit_ekf_update_reduces_measurement_error_for_clean_gps():
    truth = make_reference_orbital_state()
    ekf = make_estimator(dt=10.0)

    pred = ekf.os_hat.os.propagate_orbit_rk4(dt=10.0, J2_perturbation_on=True, fast=True)
    gps_meas = truth.clean_reading(None, truth) if hasattr(truth, "clean_reading") else None
    gps_sensor = GPS(noise=Noise(noise=np.zeros(6), std_noise=np.ones(6) * 0.01))
    gps_meas = gps_sensor.clean_reading(None, truth)

    pred_meas = np.hstack([pred.R, pred.V])
    truth_meas = np.hstack([truth.R, truth.V])
    updated = ekf.update([gps_meas], J2000=truth.J2000)
    updated_meas = np.hstack([updated.os.R, updated.os.V])

    assert np.linalg.norm(updated_meas - truth_meas) < np.linalg.norm(pred_meas - truth_meas)


@pytest.mark.slow
def test_orbit_ekf_converges_in_short_run():
    _, true_hist, est_hist, _ = run_orbit_ekf(tf=120.0, dt=10.0, noisy=False)

    initial_error = np.linalg.norm(est_hist[0, :3] - true_hist[0, :3])
    final_error = np.linalg.norm(est_hist[-1, :3] - true_hist[-1, :3])

    assert final_error < initial_error


def plot_orbit_ekf(tf=200.0, dt=10.0):
    time_hist, true_hist, est_hist, _ = run_orbit_ekf(tf=tf, dt=dt, noisy=True)
    pos_err = np.linalg.norm(est_hist[:, :3] - true_hist[:, :3], axis=1)
    vel_err = np.linalg.norm(est_hist[:, 3:] - true_hist[:, 3:], axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(time_hist, pos_err)
    axes[0].set_ylabel("Position error [km]")
    axes[0].grid(True)
    axes[1].plot(time_hist, vel_err)
    axes[1].set_ylabel("Velocity error [km/s]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()


def main():
    plot_orbit_ekf(tf=500.0, dt=5.0)


if __name__ == "__main__":
    main()
