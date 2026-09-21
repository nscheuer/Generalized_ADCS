import numpy as np
import pytest

from ADCS.helpers.math_constants import MathConstants
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import Gyro, MTM


UNIT_VECTORS = MathConstants.unitvecs


def make_satellite() -> Satellite:
    reaction_wheels = [RW(axis=UNIT_VECTORS[index], max_torque=4.51, J=0.22, h=1.0, h_max=3.8) for index in range(3)]
    gyros = [
        Gyro(
            axis=UNIT_VECTORS[index],
            bias=Bias(bias=0.01, std_bias=1e-4),
            noise=Noise(noise=0.0, std_noise=1e-4),
        )
        for index in range(3)
    ]
    magnetometers = [MTM(axis=UNIT_VECTORS[index]) for index in range(3)]
    return Satellite(
        mass=4.0,
        J_0=np.diagflat([3.4, 2.9, 1.3]),
        actuators=[MTQ(axis=UNIT_VECTORS[0], max_torque=0.1)] + reaction_wheels,
        sensors=gyros + magnetometers,
    )


def test_from_satellite_creates_independent_sensor_container():
    satellite = make_satellite()
    estimated_satellite = EstimatedSatellite.from_satellite(satellite)
    assert estimated_satellite.sensors is not satellite.sensors
    assert all(clone is not original for clone, original in zip(estimated_satellite.sensors, satellite.sensors))


def test_from_satellite_creates_independent_actuator_container():
    satellite = make_satellite()
    estimated_satellite = EstimatedSatellite.from_satellite(satellite)
    assert estimated_satellite.actuators is not satellite.actuators
    assert all(clone is not original for clone, original in zip(estimated_satellite.actuators, satellite.actuators))


def test_from_satellite_creates_independent_disturbance_container():
    satellite = make_satellite()
    estimated_satellite = EstimatedSatellite.from_satellite(satellite)
    assert estimated_satellite.disturbances is not satellite.disturbances


def test_from_satellite_copies_nested_bias_and_noise_objects():
    satellite = make_satellite()
    estimated_satellite = EstimatedSatellite.from_satellite(satellite)
    assert estimated_satellite.sensors[0].bias is not satellite.sensors[0].bias
    assert estimated_satellite.sensors[0].noise is not satellite.sensors[0].noise


def test_reaction_wheel_state_mutation_does_not_leak_to_original_satellite():
    satellite = make_satellite()
    estimated_satellite = EstimatedSatellite.from_satellite(satellite)
    satellite.actuators[1].h = 1.0
    estimated_satellite.actuators[1].h = 99.0
    assert satellite.actuators[1].h == pytest.approx(1.0)


def test_cloned_actuator_produces_same_torque_as_original():
    satellite = make_satellite()
    estimated_satellite = EstimatedSatellite.from_satellite(satellite)

    class StubOrbitalState:
        J2000 = 0.2

    true_torque = np.asarray(satellite.actuators[1].torque(0.5, None, StubOrbitalState()), dtype=float)
    estimated_torque = np.asarray(estimated_satellite.actuators[1].torque(0.5, None, StubOrbitalState()), dtype=float)
    np.testing.assert_allclose(estimated_torque, true_torque, rtol=0, atol=0)
