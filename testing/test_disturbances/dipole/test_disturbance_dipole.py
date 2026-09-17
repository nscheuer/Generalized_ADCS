import numpy as np

from ADCS.helpers.math_helpers import normalize
from ADCS.satellite_hardware.disturbances import Dipole_Disturbance
from ADCS.satellite_hardware.errors import Noise
from ADCS.state import State
from testing.test_disturbances._helpers import (
    fd_quat_hess,
    fd_quat_jac,
    fd_vec_jac,
    make_orbital_state,
    make_satellite,
    make_state,
)


def fd_quat_matrix_jac(fun, x: State, eps: float = 1.0e-6) -> np.ndarray:
    sample = fun(x)
    jac = np.zeros((4,) + sample.shape)
    for index in range(4):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus.q[index] += eps
        x_minus.q[index] -= eps
        x_plus.q[:] = normalize(x_plus.q)
        x_minus.q[:] = normalize(x_minus.q)
        jac[index] = (fun(x_plus) - fun(x_minus)) / (2.0 * eps)
    return jac


def test_dipole_estimation_flag_sets_vector_length():
    disturbance = Dipole_Disturbance(np.array([0.01, -0.02, 0.03]), estimate_dist=True)

    assert disturbance.estimate_dist
    assert disturbance.estimated_vector_length == 3


def test_dipole_update_applies_configured_noise_offset():
    disturbance = Dipole_Disturbance(
        np.array([0.01, -0.02, 0.03]),
        noise=Noise(noise=np.array([0.004, -0.001, 0.002])),
    )

    disturbance.update()

    assert np.allclose(disturbance.current_torque, np.array([0.014, -0.021, 0.032]))


def test_dipole_torque_matches_cross_product():
    disturbance = Dipole_Disturbance(np.array([0.03, -0.02, 0.01]))
    x = make_state(q=np.array([0.8, -0.2, 0.3, 0.4]))
    os = make_orbital_state(B=np.array([2.0e-5, -4.0e-5, 1.0e-5]))
    expected = np.cross(disturbance.current_torque, os.get_state_vector(x=x)["b"])

    torque = disturbance.torque(x=x, os=os)

    assert np.allclose(torque, expected)


def test_dipole_torque_is_zero_when_dipole_is_parallel_to_field():
    disturbance = Dipole_Disturbance(np.array([2.0, -1.0, 4.0]))
    x = make_state()
    os = make_orbital_state(B=np.array([2.0e-5, -1.0e-5, 4.0e-5]))

    torque = disturbance.torque(x=x, os=os)

    assert np.allclose(torque, np.zeros(3))


def test_dipole_torque_qjac_matches_finite_difference():
    disturbance = Dipole_Disturbance(np.array([0.02, -0.03, 0.01]))
    x = make_state(q=np.array([0.7, 0.1, -0.2, 0.5]))
    os = make_orbital_state(B=np.array([2.0e-5, -1.0e-5, 4.0e-5]))
    expected = fd_quat_jac(lambda xx: disturbance.torque(x=xx, os=os), x)

    jacobian = disturbance.torque_qjac(x=x, os=os)

    assert np.allclose(jacobian, expected, atol=1.0e-5, rtol=1.0e-4)


def test_dipole_torque_qqhess_matches_finite_difference():
    disturbance = Dipole_Disturbance(np.array([0.02, -0.03, 0.01]))
    x = make_state(q=np.array([0.7, 0.1, -0.2, 0.5]))
    os = make_orbital_state(B=np.array([2.0e-5, -1.0e-5, 4.0e-5]))
    expected = fd_quat_hess(lambda xx: disturbance.torque(x=xx, os=os), x)

    hessian = disturbance.torque_qqhess(x=x, os=os)

    assert np.allclose(hessian, expected, atol=2.0e-5, rtol=2.0e-4)


def test_dipole_torque_valjac_matches_finite_difference():
    disturbance = Dipole_Disturbance(np.array([0.02, -0.03, 0.01]))
    x = make_state(q=np.array([0.9, -0.1, 0.2, 0.3]))
    os = make_orbital_state(B=np.array([2.0e-5, -1.0e-5, 4.0e-5]))
    nominal = disturbance.current_torque.copy()

    def torque_from_dipole(value: np.ndarray) -> np.ndarray:
        disturbance.current_torque = value
        return disturbance.torque(x=x, os=os)

    expected = fd_vec_jac(torque_from_dipole, nominal)
    disturbance.current_torque = nominal

    jacobian = disturbance.torque_valjac(x=x, os=os)

    assert np.allclose(jacobian, expected, atol=1.0e-10, rtol=1.0e-8)


def test_dipole_torque_qvalhess_matches_finite_difference():
    disturbance = Dipole_Disturbance(np.array([0.02, -0.03, 0.01]))
    x = make_state(q=np.array([0.7, 0.1, -0.2, 0.5]))
    os = make_orbital_state(B=np.array([2.0e-5, -1.0e-5, 4.0e-5]))
    expected = fd_quat_matrix_jac(lambda xx: disturbance.torque_valjac(x=xx, os=os), x)

    hessian = disturbance.torque_qvalhess(x=x, os=os)

    assert np.allclose(hessian, expected, atol=7.0e-5, rtol=5.0e-4)


def test_dipole_disturbance_is_used_in_satellite_dynamics():
    disturbance = Dipole_Disturbance(np.array([0.03, -0.02, 0.01]))
    sat = make_satellite(disturbances=[disturbance])
    x = make_state()
    os = make_orbital_state(B=np.array([2.0e-5, -4.0e-5, 1.0e-5]))
    torque = disturbance.torque(x=x, os=os)

    xdot = sat.dynamics_core(x=x, u=np.zeros(0), orbital_state=os)

    assert np.allclose(xdot[:3], sat.invJ_noRW @ torque)
    assert np.allclose(xdot[3:7], np.zeros(4))
