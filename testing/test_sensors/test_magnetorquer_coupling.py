"""The plant's magnetometers see the field of the spacecraft's own torquers.

``MagnetorquerCoupling`` is one matrix K (tesla per A m^2) from the total
torquer dipole moment, in the body frame, to the field at the magnetometer,
including whatever soft iron lies between. Each torquer remembers the moment
it last applied; the satellite connects the coupling to its torquers when it
is built; the estimator's clone has no coupling.
"""

from __future__ import annotations

import numpy as np
import pytest

from ADCS.helpers.math_helpers import normalize, rot_mat
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.satellite_hardware.errors import Bias, MagnetorquerCoupling, Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite, Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.state import State

TRUTH = State(w=np.array([0.02, -0.01, 0.015]), q=normalize(np.array([0.9, 0.2, -0.3, 0.1])))
K = np.array([[2.0e-6, 1.0e-7, 0.0], [1.0e-7, 1.5e-6, 2.0e-7], [0.0, 2.0e-7, 3.0e-6]])  # tesla per A m^2, with soft-iron cross terms


def _orbital_state() -> Orbital_State:
    return Orbital_State(ephem=Ephemeris(), J2000=0.22, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]))


def _satellite(coupled: bool = True) -> Satellite:
    coupling = MagnetorquerCoupling(K) if coupled else None
    return Satellite(
        J_0=np.eye(3),
        sensors=[MTM(axis, torquer_coupling=coupling) for axis in np.eye(3)],
        actuators=[MTQ(axis, max_torque=1.0) for axis in np.eye(3)],
    )


def test_a_torquer_remembers_the_moment_it_applied():
    os = _orbital_state()
    torquer = MTQ(np.array([1.0, 0.0, 0.0]), max_torque=0.5, bias=Bias(bias=0.01), noise=Noise(std_noise=0.0))
    assert torquer.last_moment == 0.0
    torquer.torque(0.2, TRUTH, os)
    assert torquer.last_moment == pytest.approx(0.21)  # command plus bias
    torquer.torque(5.0, TRUTH, os)
    assert torquer.last_moment == pytest.approx(0.51)  # saturated, then biased


def test_magnetometers_read_the_torquers_field():
    os = _orbital_state()
    satellite = _satellite()
    ambient = rot_mat(TRUTH.q).T @ os.B
    np.testing.assert_allclose(satellite.noiseless_sensor_readings(TRUTH, os), ambient)  # nothing commanded yet
    command = np.array([0.3, -0.1, 0.2])
    satellite.act_torque(TRUTH, command, os)
    expected = ambient + K @ command
    np.testing.assert_allclose(satellite.noiseless_sensor_readings(TRUTH, os), expected)
    for sensor in satellite.sensors:
        assert sensor.torquer_coupling.torquers == satellite.mtq_actuators
        np.testing.assert_allclose(sensor.torquer_coupling.moment(), command)


def test_without_a_coupling_nothing_changes():
    os = _orbital_state()
    satellite = _satellite(coupled=False)
    satellite.act_torque(TRUTH, np.array([0.3, -0.1, 0.2]), os)
    np.testing.assert_allclose(satellite.noiseless_sensor_readings(TRUTH, os), rot_mat(TRUTH.q).T @ os.B)


def test_the_estimator_clone_has_no_coupling_and_the_jacobian_is_unchanged():
    os = _orbital_state()
    satellite = _satellite()
    estimated = EstimatedSatellite.from_satellite(satellite)
    assert all(sensor.torquer_coupling is None for sensor in estimated.sensors)
    satellite.act_torque(TRUTH, np.array([0.3, -0.1, 0.2]), os)
    for truth_sensor, model in zip(satellite.sensors, estimated.sensors):
        np.testing.assert_allclose(truth_sensor.basestate_jac(TRUTH, os), model.basestate_jac(TRUTH, os))  # state independent
        assert not np.allclose(truth_sensor.clean_reading(TRUTH, os), model.clean_reading(TRUTH, os))


def test_coupling_is_validated():
    with pytest.raises(ValueError, match="3x3"):
        MagnetorquerCoupling(np.eye(2))
    with pytest.raises(ValueError, match="3x3"):
        MagnetorquerCoupling(np.full((3, 3), np.nan))
    unattached = MagnetorquerCoupling(K)
    np.testing.assert_allclose(unattached.field(), np.zeros(3))
