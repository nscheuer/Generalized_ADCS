"""Test that ``EstimatedSatellite.from_satellite()`` deep-copies hardware.

These tests verify that the cloned satellite uses independent sensor,
actuator, and disturbance objects, that state changes do not leak back to
the original satellite, and that the copied hardware preserves behaviour.
"""

import numpy as np
import pytest

from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import Gyro, MTM
from ADCS.satellite_hardware.actuators import RW, MTQ
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.helpers.math_constants import MathConstants

_UV = MathConstants.unitvecs


def _sat():
    rws = [RW(axis=_UV[j], max_torque=4.51, J=0.22, h=1.0, h_max=3.8)
           for j in range(3)]
    gyros = [Gyro(axis=_UV[j], bias=Bias(bias=0.01, std_bias=1e-4),
                  noise=Noise(noise=0.0, std_noise=1e-4)) for j in range(3)]
    mtms = [MTM(axis=_UV[j]) for j in range(3)]
    return Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]),
                     actuators=[MTQ(axis=_UV[0], max_torque=0.1)] + rws,
                     sensors=gyros + mtms)


def test_from_satellite_hardware_objects_are_independent_instances():
    sat = _sat()
    est = EstimatedSatellite.from_satellite(sat)
    assert est.sensors is not sat.sensors
    assert est.actuators is not sat.actuators
    assert est.disturbances is not sat.disturbances
    for a, b in zip(est.sensors, sat.sensors):
        assert a is not b, "from_satellite shares a sensor instance with the plant"
    for a, b in zip(est.actuators, sat.actuators):
        assert a is not b, "from_satellite shares an actuator instance with the plant"
    # nested mutable state must also be distinct
    assert est.sensors[0].bias is not sat.sensors[0].bias
    assert est.sensors[0].noise is not sat.sensors[0].noise


def test_from_satellite_state_mutation_does_not_leak_to_plant():
    """Writing reaction-wheel momentum on the estimated satellite must NOT
    change the true satellite (RED on origin/main: shared instance)."""
    sat = _sat()
    est = EstimatedSatellite.from_satellite(sat)
    sat.actuators[1].h = 1.0
    est.actuators[1].h = 99.0
    assert sat.actuators[1].h == pytest.approx(1.0), (
        "estimated-satellite RW momentum mutation leaked into the true "
        "satellite -> from_satellite aliases hardware")


def test_from_satellite_preserves_behaviour():
    """Deep-copy must not change behaviour: the cloned actuator produces an
    identical torque for the same command."""
    sat = _sat()
    est = EstimatedSatellite.from_satellite(sat)

    class _OS:
        J2000 = 0.2

    t_true = np.asarray(sat.actuators[1].torque(0.5, None, _OS()), float)
    t_est = np.asarray(est.actuators[1].torque(0.5, None, _OS()), float)
    np.testing.assert_allclose(t_est, t_true, rtol=0, atol=0)
