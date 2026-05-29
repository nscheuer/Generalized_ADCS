import numpy as np
import pytest

from ADCS.satellite_hardware.errors import AnisotropicNoise, Noise
from ADCS.satellite_hardware.sensors import Gyro, MTM, StarTracker
from testing.test_estimators.srukf.helpers import make_mtqs, make_orbital_state, make_satellites, make_state, make_srukf
from testing.test_estimators.srukf.scenarios import startracker_dropout_scenario


def build_startracker_pair():
    real_sensors = [
        *[MTM(axis=axis, noise=Noise(noise=0.0, std_noise=1.0e-8)) for axis in np.eye(3)],
        *[Gyro(axis=axis, noise=Noise(noise=0.0, std_noise=1.0e-5)) for axis in np.eye(3)],
        StarTracker(
            boresight=np.array([0.0, 0.0, 1.0]),
            fov=np.deg2rad(120.0),
            sun_exclusion=np.deg2rad(5.0),
            anisotropic_noise=AnisotropicNoise(std_cross=1.0e-6, std_roll=2.0e-6),
        ),
    ]
    est_sensors = [
        *[MTM(axis=axis, noise=Noise(noise=0.0, std_noise=1.0e-8)) for axis in np.eye(3)],
        *[Gyro(axis=axis, noise=Noise(noise=0.0, std_noise=1.0e-5)) for axis in np.eye(3)],
        StarTracker(
            boresight=np.array([0.0, 0.0, 1.0]),
            fov=np.deg2rad(120.0),
            sun_exclusion=np.deg2rad(5.0),
            anisotropic_noise=AnisotropicNoise(std_cross=1.0e-6, std_roll=2.0e-6),
        ),
    ]
    return make_satellites(sensors=real_sensors, estimated_sensors=est_sensors, actuators=make_mtqs(), estimated_actuators=make_mtqs())


def test_startracker_covariance_and_measurement_dimensions():
    _, est_sat = build_startracker_pair()
    cov = est_sat.sensor_cov([True] * len(est_sat.attitude_sensors))
    assert cov.shape == (9, 9)


def test_startracker_visible_measurement_update_runs(monkeypatch):
    real_sat, est_sat = build_startracker_pair()
    srukf = make_srukf(est_sat)
    monkeypatch.setattr(real_sat.attitude_sensors[-1], "reading", lambda x, os, dmode=None: np.array([0.0, 0.0, 1.0]))
    monkeypatch.setattr(est_sat.attitude_sensors[-1], "reading", lambda x, os, dmode=None: np.array([0.0, 0.0, 1.0]))
    sensors = real_sat.noiseless_sensor_readings(make_state(), make_orbital_state())
    srukf.update(u=np.zeros(len(real_sat.actuators)), sensors=sensors, os=make_orbital_state())
    assert np.isfinite(srukf.x_hat.val).all()


def test_startracker_hidden_measurement_update_masks_dropouts(monkeypatch):
    real_sat, est_sat = build_startracker_pair()
    srukf = make_srukf(est_sat)
    monkeypatch.setattr(real_sat.attitude_sensors[-1], "reading", lambda x, os, dmode=None: np.array([np.nan, np.nan, np.nan]))
    monkeypatch.setattr(est_sat.attitude_sensors[-1], "reading", lambda x, os, dmode=None: np.array([0.0, 0.0, 1.0]))
    sensors = real_sat.sensor_readings(make_state(), make_orbital_state())
    srukf.update(u=np.zeros(len(real_sat.actuators)), sensors=sensors, os=make_orbital_state())
    assert np.isfinite(srukf.x_hat.val).all()



def test_startracker_dropout_scenario_stays_finite_and_improves():
    result = startracker_dropout_scenario()
    assert np.isfinite(result.estimate).all()
    assert np.linalg.norm(result.estimate[-1, :3]) < 0.05
