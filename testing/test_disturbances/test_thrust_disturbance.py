"""Thrust_Disturbance: one body-fixed force, and the torque its lever arm implies."""

import numpy as np

from ADCS.satellite_hardware.disturbances import Thrust_Disturbance
from ADCS.satellite_hardware.errors.noise import Noise
from testing.test_disturbances._helpers import (
    make_orbital_state,
    make_satellite,
    make_state,
)

F0 = np.array([0.0, 0.0, 1.0e-3])


def test_force_is_the_nominal_thrust_and_through_com_gives_no_torque():
    thrust = Thrust_Disturbance(F0)
    sat = make_satellite(COM=np.array([0.01, -0.02, 0.03]), disturbances=[thrust])
    x = make_state()
    os = make_orbital_state()

    assert np.allclose(thrust.force(sat=sat, x=x, os=os), F0)
    assert np.allclose(thrust.torque(sat=sat, x=x, os=os), np.zeros(3))


def test_torque_is_lever_arm_about_com_cross_force():
    location = np.array([0.1, -0.05, -0.2])
    thrust = Thrust_Disturbance(F0, location=location)
    sat = make_satellite(COM=np.array([0.01, -0.02, 0.03]), disturbances=[thrust])
    x = make_state()
    os = make_orbital_state()

    expected = np.cross(location - sat.COM, F0)

    assert np.allclose(thrust.torque(sat=sat, x=x, os=os), expected)
    assert np.allclose(thrust.force(sat=sat, x=x, os=os), F0)


def test_thrust_does_not_depend_on_attitude():
    thrust = Thrust_Disturbance(F0, location=np.array([0.1, 0.0, 0.0]))
    sat = make_satellite(disturbances=[thrust])
    os = make_orbital_state()
    x_a = make_state(q=np.array([1.0, 0.0, 0.0, 0.0]))
    x_b = make_state(q=np.array([0.5, 0.5, -0.5, 0.5]))

    assert np.allclose(thrust.force(sat=sat, x=x_a, os=os), thrust.force(sat=sat, x=x_b, os=os))
    assert np.allclose(thrust.torque(sat=sat, x=x_a, os=os), thrust.torque(sat=sat, x=x_b, os=os))
    assert np.array_equal(thrust.torque_qjac(sat, x_a, os), np.zeros((4, 3)))
    assert np.array_equal(thrust.torque_qqhess(sat, x_a, os), np.zeros((4, 4, 3)))
    assert np.array_equal(thrust.force_qjac(sat, x_a, os), np.zeros((4, 3)))
    assert np.array_equal(thrust.force_qqhess(sat, x_a, os), np.zeros((4, 4, 3)))


def test_noise_is_drawn_on_update_and_held_between_updates():
    np.random.seed(3)
    thrust = Thrust_Disturbance(F0, noise=Noise(std_noise=1.0e-4))
    sat = make_satellite(disturbances=[thrust])
    x = make_state()
    os = make_orbital_state()

    assert np.allclose(thrust.force(sat=sat, x=x, os=os), F0)  # nominal until the first draw

    thrust.update()
    first = thrust.force(sat=sat, x=x, os=os).copy()
    assert not np.allclose(first, F0)
    assert np.allclose(thrust.force(sat=sat, x=x, os=os), first)  # held
    assert np.allclose(thrust.force(sat=sat, x=x, os=os), first)

    thrust.update()
    second = thrust.force(sat=sat, x=x, os=os)
    assert not np.allclose(second, first)
    assert np.linalg.norm(second - F0) < 10.0 * 1.0e-4 * np.sqrt(3.0)


def test_update_errors_drives_update():
    np.random.seed(5)
    thrust = Thrust_Disturbance(F0, noise=Noise(std_noise=1.0e-4))
    sat = make_satellite(disturbances=[thrust])
    x = make_state()
    os = make_orbital_state()

    sat.update_errors(os.J2000)

    assert not np.allclose(thrust.force(sat=sat, x=x, os=os), F0)


def test_thrust_reaches_satellite_sums_and_dynamics():
    location = np.array([0.1, -0.05, -0.2])
    thrust = Thrust_Disturbance(F0, location=location)
    sat = make_satellite(COM=np.array([0.01, -0.02, 0.03]), disturbances=[thrust])
    x = make_state()
    os = make_orbital_state()
    torque = thrust.torque(sat=sat, x=x, os=os)

    assert np.allclose(sat.dist_forces(x=x, os=os), F0)
    assert np.allclose(sat.dist_torques(x=x, os=os), torque)
    xdot = sat.dynamics_core(x=x, u=np.zeros(0), orbital_state=os)
    assert np.allclose(xdot[:3], sat.invJ_noRW @ torque)
