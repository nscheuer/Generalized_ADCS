import numpy as np
import pytest

from ADCS.controller import MTQ_w_RW_QPW

from testing.test_controllers._mtq_rw_qp_test_helpers import assert_command_bounds
from testing.test_controllers._mtq_rw_qp_test_helpers import achieved_torque
from testing.test_controllers._mtq_rw_qp_test_helpers import make_controller
from testing.test_controllers._mtq_rw_qp_test_helpers import make_satellite
from testing.test_controllers._mtq_rw_qp_test_helpers import plain_bounded_lsq


@pytest.fixture
def qpw_satellite():
    return make_satellite(include_rw=True)


@pytest.fixture
def qpw_controller(qpw_satellite):
    return make_controller(MTQ_w_RW_QPW, qpw_satellite, 1.0, 1.0, 0.0)


def test_qpw_zero_torque_request_returns_zero_command(qpw_satellite, qpw_controller) -> None:
    u_rw, u_mtq, alpha = qpw_controller.allocate_max_torque_in_direction(
        np.zeros(3),
        np.array([0.0, 0.0, 2.0e-5]),
        qpw_satellite,
    )

    np.testing.assert_allclose(u_rw, np.zeros(1))
    np.testing.assert_allclose(u_mtq, np.zeros(3))
    assert alpha == 1.0


def test_qpw_tracks_exact_feasible_request(qpw_satellite, qpw_controller) -> None:
    tau_des = np.array([3.0e-3, 0.0, 0.0])
    b_body = np.array([0.0, 0.0, 2.0e-5])

    u_rw, u_mtq, alpha = qpw_controller.allocate_max_torque_in_direction(tau_des, b_body, qpw_satellite)
    tau_ach = achieved_torque(qpw_satellite, u_rw, u_mtq, b_body)

    np.testing.assert_allclose(tau_ach, tau_des, atol=1.0e-10)
    assert_command_bounds(u_rw, u_mtq, qpw_controller.rw_umax, qpw_controller.mtq_umax)
    assert np.isclose(alpha, 1.0)


def test_qpw_returns_zero_for_parallel_field_request_without_reaction_wheels() -> None:
    satellite = make_satellite(include_rw=False)
    controller = make_controller(MTQ_w_RW_QPW, satellite, 1.0, 1.0, 0.0)
    tau_des = np.array([0.0, 0.0, 1.0e-3])
    b_body = np.array([0.0, 0.0, 1.0])

    u_rw, u_mtq, alpha = controller.allocate_max_torque_in_direction(tau_des, b_body, satellite)
    tau_ach = achieved_torque(satellite, u_rw, u_mtq, b_body)

    np.testing.assert_allclose(u_rw, np.zeros(0))
    np.testing.assert_allclose(tau_ach, np.zeros(3), atol=1.0e-12)
    assert_command_bounds(u_rw, u_mtq, np.zeros(0), controller.mtq_umax)
    assert alpha == 0.0


def test_qpw_reduces_direction_error_relative_to_plain_bounded_lsq(
    qpw_satellite,
    qpw_controller,
) -> None:
    tau_des = np.array([5.0e-3, 5.0e-3, 5.0e-3])
    b_body = np.array([0.0, 0.0, 2.0e-5])
    tau_hat = tau_des / np.linalg.norm(tau_des)

    plain_u, plain_tau = plain_bounded_lsq(qpw_satellite, tau_des, b_body)
    u_rw, u_mtq, alpha = qpw_controller.allocate_max_torque_in_direction(tau_des, b_body, qpw_satellite)
    tau_ach = achieved_torque(qpw_satellite, u_rw, u_mtq, b_body)

    qpw_residual = tau_ach - tau_des
    plain_residual = plain_tau - tau_des
    qpw_perp = np.linalg.norm(qpw_residual - np.dot(qpw_residual, tau_hat) * tau_hat)
    plain_perp = np.linalg.norm(plain_residual - np.dot(plain_residual, tau_hat) * tau_hat)

    assert qpw_perp <= plain_perp + 1.0e-12
    assert_command_bounds(u_rw, u_mtq, qpw_controller.rw_umax, qpw_controller.mtq_umax)
    assert alpha >= 0.0
    assert plain_u.shape == (4,)


def test_qpw_alpha_matches_projection_of_achieved_torque(qpw_satellite, qpw_controller) -> None:
    tau_des = np.array([1.0e-3, 2.0e-3, 8.0e-3])
    b_body = np.array([0.0, 0.0, 2.0e-5])

    u_rw, u_mtq, alpha = qpw_controller.allocate_max_torque_in_direction(tau_des, b_body, qpw_satellite)
    tau_ach = achieved_torque(qpw_satellite, u_rw, u_mtq, b_body)
    tau_hat = tau_des / np.linalg.norm(tau_des)
    expected_alpha = max(0.0, float(np.dot(tau_ach, tau_hat) / np.linalg.norm(tau_des)))

    assert np.isclose(alpha, expected_alpha)
