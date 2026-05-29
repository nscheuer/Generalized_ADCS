import numpy as np
import pytest

from ADCS.controller import MTQ_w_RW_QPC

from testing.test_controllers._mtq_rw_qp_test_helpers import assert_command_bounds
from testing.test_controllers._mtq_rw_qp_test_helpers import combined_actuation_matrix
from testing.test_controllers._mtq_rw_qp_test_helpers import achieved_torque
from testing.test_controllers._mtq_rw_qp_test_helpers import make_controller
from testing.test_controllers._mtq_rw_qp_test_helpers import make_satellite
from testing.test_controllers._mtq_rw_qp_test_helpers import plain_bounded_lsq


@pytest.fixture
def qpc_satellite():
    return make_satellite(include_rw=True)


@pytest.fixture
def qpc_controller(qpc_satellite):
    return make_controller(MTQ_w_RW_QPC, qpc_satellite, 1.0, 1.0, 0.0)


def test_qpc_zero_torque_request_returns_zero_command(qpc_satellite, qpc_controller) -> None:
    u_rw, u_mtq, alpha = qpc_controller.allocate_max_torque_in_direction(
        np.zeros(3),
        np.array([0.0, 0.0, 2.0e-5]),
        qpc_satellite,
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0]),
    )

    np.testing.assert_allclose(u_rw, np.zeros(1))
    np.testing.assert_allclose(u_mtq, np.zeros(3))
    assert alpha == 1.0


def test_qpc_tracks_exact_feasible_request(qpc_satellite, qpc_controller) -> None:
    tau_des = np.array([3.0e-3, 0.0, 0.0])
    b_body = np.array([0.0, 0.0, 2.0e-5])

    u_rw, u_mtq, alpha = qpc_controller.allocate_max_torque_in_direction(
        tau_des,
        b_body,
        qpc_satellite,
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0]),
    )
    tau_ach = achieved_torque(qpc_satellite, u_rw, u_mtq, b_body)

    np.testing.assert_allclose(tau_ach, tau_des, atol=1.0e-10)
    assert_command_bounds(u_rw, u_mtq, qpc_controller.rw_umax, qpc_controller.mtq_umax)
    assert np.isclose(alpha, float(np.dot(tau_ach, tau_des / np.linalg.norm(tau_des)) / np.linalg.norm(tau_des)))


def test_qpc_blocks_positive_power_when_desired_torque_is_dissipative(
    qpc_satellite,
    qpc_controller,
) -> None:
    tau_des = np.array([1.0e-3, -5.0e-3, 3.0e-3])
    b_body = np.array([0.0, 0.0, 2.0e-5])
    omega = np.array([1.0, 1.0, 0.0])

    plain_u, plain_tau = plain_bounded_lsq(qpc_satellite, tau_des, b_body)
    u_rw, u_mtq, alpha = qpc_controller.allocate_max_torque_in_direction(
        tau_des,
        b_body,
        qpc_satellite,
        omega,
        np.array([0.0]),
    )
    tau_ach = achieved_torque(qpc_satellite, u_rw, u_mtq, b_body)

    assert float(np.dot(omega, tau_des)) < 0.0
    assert float(np.dot(omega, plain_tau)) > 0.0
    assert float(np.dot(omega, tau_ach)) <= 1.0e-9
    assert alpha >= 0.0
    assert plain_u.shape == (4,)


def test_qpc_returns_zero_for_parallel_field_request_without_reaction_wheels() -> None:
    satellite = make_satellite(include_rw=False)
    controller = make_controller(MTQ_w_RW_QPC, satellite, 1.0, 1.0, 0.0)
    tau_des = np.array([0.0, 0.0, 1.0e-3])
    b_body = np.array([0.0, 0.0, 1.0])

    u_rw, u_mtq, alpha = controller.allocate_max_torque_in_direction(
        tau_des,
        b_body,
        satellite,
        np.array([1.0, 0.0, 0.0]),
        np.zeros(0),
    )
    tau_ach = achieved_torque(satellite, u_rw, u_mtq, b_body)

    np.testing.assert_allclose(u_rw, np.zeros(0))
    np.testing.assert_allclose(tau_ach, np.zeros(3), atol=1.0e-12)
    assert_command_bounds(u_rw, u_mtq, np.zeros(0), controller.mtq_umax)
    assert alpha == 0.0


def test_qpc_lp_scaling_preserves_direction_and_bounds(qpc_satellite, qpc_controller) -> None:
    tau_des = np.array([2.0e-2, 3.0e-3, 4.0e-3])
    b_body = np.array([0.0, 0.0, 2.0e-5])
    a_total, rw_limits, mtq_limits = combined_actuation_matrix(qpc_satellite, b_body)
    lower = np.concatenate([-rw_limits, -mtq_limits])
    upper = np.concatenate([rw_limits, mtq_limits])

    tau_lp, alpha, u_sol = qpc_controller.solve_lp_scaling(tau_des, a_total, lower, upper)

    assert alpha >= 0.0
    assert_command_bounds(u_sol[:1], u_sol[1:], rw_limits, mtq_limits)
    np.testing.assert_allclose(np.cross(tau_lp, tau_des), np.zeros(3), atol=1.0e-10)
