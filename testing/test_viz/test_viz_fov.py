"""Sensor field-of-view normalization to a half-cone angle."""

import numpy as np
import pytest

from ADCS.satellite_hardware.sensors import (
    EarthHorizonSensor,
    Gyro,
    StarTracker,
    StarTrackerQuaternion,
    SunPair,
    SunSensor,
)
from ADCS.viz import sensor_fov


@pytest.mark.parametrize("cls", [StarTracker, StarTrackerQuaternion])
def test_star_trackers_store_a_full_angle(cls):
    sensor = cls(boresight=np.array([1.0, 0.0, 0.0]), fov=np.deg2rad(20.0))
    fov = sensor_fov(sensor)
    assert fov.fov_half_rad == pytest.approx(np.deg2rad(10.0))
    np.testing.assert_allclose(fov.boresight_body, [1.0, 0.0, 0.0])
    assert not fov.assumed and not fov.bidirectional


def test_horizon_sensor_stores_a_half_angle():
    sensor = EarthHorizonSensor(boresight=np.array([0.0, 0.0, -1.0]), fov=np.deg2rad(60.0))
    fov = sensor_fov(sensor)
    assert fov.fov_half_rad == pytest.approx(np.deg2rad(60.0))
    np.testing.assert_allclose(fov.boresight_body, [0.0, 0.0, -1.0])


def test_sun_sensor_is_an_assumed_hemisphere():
    fov = sensor_fov(SunSensor(axis=np.array([0.0, 2.0, 0.0]), efficiency=1.0))
    assert fov.fov_half_rad == pytest.approx(np.pi / 2)
    np.testing.assert_allclose(fov.boresight_body, [0.0, 1.0, 0.0])
    assert fov.assumed and not fov.bidirectional


def test_sun_pair_sees_both_hemispheres():
    fov = sensor_fov(SunPair(axis=np.array([1.0, 0.0, 0.0]), efficiency=(1.0, 1.0)))
    assert fov.assumed and fov.bidirectional


def test_non_directional_sensors_have_no_fov():
    assert sensor_fov(Gyro(axis=np.array([1.0, 0.0, 0.0]))) is None
