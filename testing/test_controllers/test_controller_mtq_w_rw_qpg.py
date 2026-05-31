import numpy as np
import pytest

from ADCS.controller import MTQ_w_RW_QPG

from testing.test_controllers._mtq_rw_qp_test_helpers import assert_command_bounds
from testing.test_controllers._mtq_rw_qp_test_helpers import achieved_torque
from testing.test_controllers._mtq_rw_qp_test_helpers import make_controller
from testing.test_controllers._mtq_rw_qp_test_helpers import make_satellite
from testing.test_controllers._mtq_rw_qp_test_helpers import plain_bounded_lsq


@pytest.fixture
def qpg_satellite():
    return make_satellite(include_rw=True)


@pytest.fixture
def qpg_controller_zero(qpg_satellite):
    return make_controller(MTQ_w_RW_QPG, qpg_satellite, 1.0, 1.0, 0.0, 0.0)


@pytest.fixture
def qpg_controller_spin_weighted(qpg_satellite):
    return make_controller(MTQ_w_RW_QPG, qpg_satellite, 1.0, 1.0, 50.0, 0.0)


def test_qpg_zero_torque_request_returns_zero_command(qpg_satellite, qpg_controller_zero) -> None:
    u_rw, u_mtq, alpha = qpg_controller_zero.allocate_max_torque_in_direction(
        np.zeros(3),
        np.array([0.0, 0.0, 2.0e-5]),
        qpg_satellite,
        np.array([1.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(u_rw, np.zeros(1))
    np.testing.assert_allclose(u_mtq, np.zeros(3))
    assert alpha == 1.0


def test_qpg_gamma_zero_matches_plain_bounded_lsq(qpg_satellite, qpg_controller_zero) -> None:
    tau_des = np.array([1.0e-3, 2.0e-3, 8.0e-3])
    b_body = np.array([0.0, 0.0, 2.0e-5])

    plain_u, plain_tau = plain_bounded_lsq(qpg_satellite, tau_des, b_body)
    u_rw, u_mtq, alpha = qpg_controller_zero.allocate_max_torque_in_direction(
        tau_des,
        b_body,
        qpg_satellite,
        np.array([1.0, 0.0, 1.0]),
    )
    tau_ach = achieved_torque(qpg_satellite, u_rw, u_mtq, b_body)

    np.testing.assert_allclose(np.concatenate([u_rw, u_mtq]), plain_u, atol=1.0e-10)
    np.testing.assert_allclose(tau_ach, plain_tau, atol=1.0e-10)
    assert alpha >= 0.0


def test_qpg_tracks_exact_feasible_request(qpg_satellite, qpg_controller_spin_weighted) -> None:
    tau_des = np.array([3.0e-3, 0.0, 0.0])
    b_body = np.array([0.0, 0.0, 2.0e-5])

    u_rw, u_mtq, alpha = qpg_controller_spin_weighted.allocate_max_torque_in_direction(
        tau_des,
        b_body,
        qpg_satellite,
        np.array([1.0, 0.0, 0.0]),
    )
    tau_ach = achieved_torque(qpg_satellite, u_rw, u_mtq, b_body)

    np.testing.assert_allclose(tau_ach, tau_des, atol=1.0e-10)
    assert_command_bounds(u_rw, u_mtq, qpg_controller_spin_weighted.rw_umax, qpg_controller_spin_weighted.mtq_umax)
    assert np.isclose(alpha, 1.0)


def test_qpg_positive_gamma_increases_achieved_projection_along_spin_axis(
    qpg_satellite,
    qpg_controller_zero,
    qpg_controller_spin_weighted,
) -> None:
    tau_des = np.array([1.0e-3, 2.0e-3, 8.0e-3])
    b_body = np.array([0.0, 0.0, 2.0e-5])
    omega = np.array([1.0, 1.0, 0.0])

    u_rw_0, u_mtq_0, alpha_0 = qpg_controller_zero.allocate_max_torque_in_direction(
        tau_des, b_body, qpg_satellite, omega
    )
    u_rw_g, u_mtq_g, alpha_g = qpg_controller_spin_weighted.allocate_max_torque_in_direction(
        tau_des, b_body, qpg_satellite, omega
    )

    tau_0 = achieved_torque(qpg_satellite, u_rw_0, u_mtq_0, b_body)
    tau_g = achieved_torque(qpg_satellite, u_rw_g, u_mtq_g, b_body)

    assert float(np.dot(omega, tau_g)) >= float(np.dot(omega, tau_0)) - 1.0e-12
    assert alpha_g >= alpha_0 - 1.0e-12
    assert_command_bounds(u_rw_g, u_mtq_g, qpg_controller_spin_weighted.rw_umax, qpg_controller_spin_weighted.mtq_umax)


def test_qpg_zero_omega_makes_gamma_irrelevant(qpg_satellite, qpg_controller_zero, qpg_controller_spin_weighted) -> None:
    tau_des = np.array([4.0e-3, 5.0e-3, 6.0e-3])
    b_body = np.array([0.0, 0.0, 2.0e-5])
    omega = np.zeros(3)

    u_rw_0, u_mtq_0, alpha_0 = qpg_controller_zero.allocate_max_torque_in_direction(
        tau_des, b_body, qpg_satellite, omega
    )
    u_rw_g, u_mtq_g, alpha_g = qpg_controller_spin_weighted.allocate_max_torque_in_direction(
        tau_des, b_body, qpg_satellite, omega
    )

    np.testing.assert_allclose(u_rw_0, u_rw_g, atol=1.0e-10)
    np.testing.assert_allclose(u_mtq_0, u_mtq_g, atol=1.0e-10)
    assert np.isclose(alpha_0, alpha_g)
