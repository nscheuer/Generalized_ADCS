import numpy as np

from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.sensors import EarthHorizonSensor, Gyro
from testing.test_estimators.ukf.helpers import make_mtqs, make_orbital_state, make_rws, make_satellites, make_state, make_ukf


def test_expand_sensor_mask_repeats_entries_by_output_length_and_adds_rw_outputs():
    sensors = [
        EarthHorizonSensor(boresight=np.array([-1.0, 0.0, 0.0]), fov=np.deg2rad(140.0), noise=Noise(noise=np.zeros(3), std_noise=np.full(3, 1.0e-5))),
        Gyro(axis=np.array([1.0, 0.0, 0.0]), noise=Noise(noise=0.0, std_noise=1.0e-5)),
    ]
    _, est_sat = make_satellites(sensors=sensors, estimated_sensors=sensors, actuators=make_mtqs() + make_rws(), estimated_actuators=make_mtqs() + make_rws())
    ukf = make_ukf(est_sat)

    mask = ukf._expand_sensor_mask([True, False, True, True, True])

    assert mask.tolist() == [True, True, True, False, True, True, True]


def test_nan_in_vector_sensor_disables_only_that_sensor(monkeypatch):
    sensors = [
        EarthHorizonSensor(boresight=np.array([-1.0, 0.0, 0.0]), fov=np.deg2rad(140.0), noise=Noise(noise=np.zeros(3), std_noise=np.full(3, 1.0e-5))),
        Gyro(axis=np.array([1.0, 0.0, 0.0]), noise=Noise(noise=0.0, std_noise=1.0e-5)),
    ]
    real_sat, est_sat = make_satellites(sensors=sensors, estimated_sensors=sensors)
    ukf = make_ukf(est_sat)
    captured = {}
    original = est_sat.sensor_cov

    def wrapped(which_sensors):
        captured["which"] = list(which_sensors)
        return original(which_sensors)

    monkeypatch.setattr(est_sat, "sensor_cov", wrapped)
    os = make_orbital_state()
    readings = real_sat.noiseless_sensor_readings(make_state(), os)
    readings[:3] = np.nan

    ukf.update(u=np.zeros(len(real_sat.actuators)), sensors=readings, os=os)

    assert captured["which"] == [False, True]


def test_rw_measurements_remain_included_when_attitude_sensors_drop_out(monkeypatch):
    sensors = [
        Gyro(axis=np.array([1.0, 0.0, 0.0]), noise=Noise(noise=0.0, std_noise=1.0e-5)),
        Gyro(axis=np.array([0.0, 1.0, 0.0]), noise=Noise(noise=0.0, std_noise=1.0e-5)),
        Gyro(axis=np.array([0.0, 0.0, 1.0]), noise=Noise(noise=0.0, std_noise=1.0e-5)),
    ]
    real_sat, est_sat = make_satellites(sensors=sensors, estimated_sensors=sensors, actuators=make_mtqs() + make_rws(), estimated_actuators=make_mtqs() + make_rws())
    ukf = make_ukf(est_sat)
    os = make_orbital_state()
    readings = real_sat.noiseless_sensor_readings(make_state(h=np.array([1.0, 1.0, 1.0])), os)
    readings[:3] = np.nan

    ukf.update(u=np.zeros(len(real_sat.actuators)), sensors=readings, os=os)

    cov = est_sat.sensor_cov([False, False, False])
    assert cov.shape == (3, 3)


def test_mixed_scalar_and_vector_sensors_produce_expected_output_dimension():
    sensors = [
        EarthHorizonSensor(boresight=np.array([-1.0, 0.0, 0.0]), fov=np.deg2rad(140.0), noise=Noise(noise=np.zeros(3), std_noise=np.full(3, 1.0e-5))),
        Gyro(axis=np.array([1.0, 0.0, 0.0]), noise=Noise(noise=0.0, std_noise=1.0e-5)),
    ]
    _, est_sat = make_satellites(sensors=sensors, estimated_sensors=sensors)
    ukf = make_ukf(est_sat)

    mask = ukf._expand_sensor_mask([True, True])

    assert mask.size == 4
