"""
Phase 4 unit tests: allocation methods and actuator set assembly.

Tests cover:
    - actuator_set.py: B_tau assembly for RW-only, MTQ-only, mixed
    - pseudoinverse.py: pinv + clip
    - lp.py: direction-preserving LP
    - qp.py: bounded least-squares QP
    - qpw.py: direction-weighted QP
    - qpc.py: energy-constrained QP
    - allocator.py: routing to all methods
    - Cross-method comparisons (LP vs QP direction error, etc.)
"""

import numpy as np
import pytest

from ADCS.pipeline.data import ActuatorGroup, AllocationConfig, AllocationResult
from ADCS.pipeline.allocation.actuator_set import assemble_B_tau
from ADCS.pipeline.allocation.pseudoinverse import allocate_pseudoinverse
from ADCS.pipeline.allocation.lp import allocate_lp
from ADCS.pipeline.allocation.qp import allocate_qp
from ADCS.pipeline.allocation.qpw import allocate_qpw
from ADCS.pipeline.allocation.qpc import allocate_qpc
from ADCS.pipeline.allocation.allocator import allocation_step
from ADCS.helpers.math_helpers import skewsym


# ---------------------------------------------------------------------------
# Fixtures: standard actuator configurations
# ---------------------------------------------------------------------------

@pytest.fixture
def mtq_3axis():
    """3 orthogonal magnetorquers along body axes."""
    return ActuatorGroup(
        group_type='mtq',
        axes=np.eye(3),
        u_max=np.array([1.0, 1.0, 1.0]),
        indices=np.array([0, 1, 2]),
    )


@pytest.fixture
def rw_3axis():
    """3 orthogonal reaction wheels along body axes."""
    return ActuatorGroup(
        group_type='rw',
        axes=np.eye(3),
        u_max=np.array([0.01, 0.01, 0.01]),
        indices=np.array([0, 1, 2]),
    )


@pytest.fixture
def rw_1axis():
    """Single reaction wheel along Z."""
    return ActuatorGroup(
        group_type='rw',
        axes=np.array([[0.0], [0.0], [1.0]]),
        u_max=np.array([0.01]),
        indices=np.array([3]),
    )


@pytest.fixture
def B_body_nominal():
    """Nominal magnetic field in body frame (roughly LEO magnitude)."""
    return np.array([2e-5, 1e-5, 3e-5])


# ---------------------------------------------------------------------------
# Test: actuator_set.py — B_tau assembly
# ---------------------------------------------------------------------------

class TestAssembleBTau:
    """Tests for assemble_B_tau."""

    def test_rw_only(self, rw_3axis):
        """RW-only: B_tau equals the axes matrix."""
        B_body = np.array([0.0, 0.0, 0.0])
        B_tau, u_min, u_max = assemble_B_tau([rw_3axis], B_body)
        assert B_tau.shape == (3, 3)
        np.testing.assert_array_almost_equal(B_tau, np.eye(3))
        np.testing.assert_array_almost_equal(u_max, [0.01, 0.01, 0.01])
        np.testing.assert_array_almost_equal(u_min, [-0.01, -0.01, -0.01])

    def test_mtq_only(self, mtq_3axis, B_body_nominal):
        """MTQ-only: B_tau = -skew(B) @ A_mtq."""
        B_tau, u_min, u_max = assemble_B_tau([mtq_3axis], B_body_nominal)
        expected = -skewsym(B_body_nominal) @ np.eye(3)
        assert B_tau.shape == (3, 3)
        np.testing.assert_array_almost_equal(B_tau, expected)

    def test_mixed_rw_mtq(self, mtq_3axis, rw_1axis, B_body_nominal):
        """Mixed: [MTQ columns | RW columns]."""
        # MTQ indices [0,1,2], RW index [3]
        B_tau, u_min, u_max = assemble_B_tau([mtq_3axis, rw_1axis], B_body_nominal)
        assert B_tau.shape == (3, 4)
        # First 3 columns are MTQ effectiveness
        expected_mtq = -skewsym(B_body_nominal) @ np.eye(3)
        np.testing.assert_array_almost_equal(B_tau[:, :3], expected_mtq)
        # Last column is RW axis
        np.testing.assert_array_almost_equal(B_tau[:, 3], [0, 0, 1])

    def test_empty_groups(self):
        """No actuator groups returns empty matrices."""
        B_tau, u_min, u_max = assemble_B_tau([], np.zeros(3))
        assert B_tau.shape == (3, 0)
        assert len(u_min) == 0
        assert len(u_max) == 0

    def test_mtq_perpendicular_to_B(self, mtq_3axis):
        """MTQ torque is always perpendicular to B."""
        B = np.array([1.0, 0.0, 0.0])
        B_tau, _, _ = assemble_B_tau([mtq_3axis], B)
        # Each column of B_tau should be perp to B
        for i in range(B_tau.shape[1]):
            col = B_tau[:, i]
            if np.linalg.norm(col) > 1e-12:
                assert abs(np.dot(col, B)) < 1e-12, \
                    f"Column {i} not perpendicular to B"


# ---------------------------------------------------------------------------
# Test: pseudoinverse.py
# ---------------------------------------------------------------------------

class TestPseudoinverse:
    """Tests for pseudoinverse allocation."""

    def test_exact_solution_rw_only(self, rw_3axis):
        """3 orthogonal RWs: exact solution without clipping."""
        B_tau = rw_3axis.axes.copy()
        u_min = -rw_3axis.u_max
        u_max = rw_3axis.u_max
        tau = np.array([0.005, -0.003, 0.002])
        indices = rw_3axis.indices

        result = allocate_pseudoinverse(tau, B_tau, u_min, u_max, 3, indices)
        np.testing.assert_array_almost_equal(result.tau_achieved, tau, decimal=10)
        assert result.feasible
        assert abs(result.alpha - 1.0) < 1e-6

    def test_clipping(self, rw_3axis):
        """Request exceeds RW bounds — should clip."""
        B_tau = rw_3axis.axes.copy()
        u_min = -rw_3axis.u_max
        u_max = rw_3axis.u_max
        tau = np.array([0.05, 0.0, 0.0])  # 5x max
        indices = rw_3axis.indices

        result = allocate_pseudoinverse(tau, B_tau, u_min, u_max, 3, indices)
        assert not result.feasible
        assert result.alpha < 1.0
        assert abs(result.u[0]) <= 0.01 + 1e-12

    def test_zero_desired(self, rw_3axis):
        """Zero desired torque => zero commands, alpha=1."""
        B_tau = rw_3axis.axes.copy()
        tau = np.zeros(3)
        result = allocate_pseudoinverse(
            tau, B_tau, -rw_3axis.u_max, rw_3axis.u_max, 3, rw_3axis.indices,
        )
        np.testing.assert_array_almost_equal(result.u, np.zeros(3))
        assert result.alpha == 1.0


# ---------------------------------------------------------------------------
# Test: lp.py — direction-preserving LP
# ---------------------------------------------------------------------------

class TestLP:
    """Tests for LP allocation."""

    def test_direction_preserved(self, rw_3axis):
        """LP should produce zero direction error."""
        B_tau = rw_3axis.axes.copy()
        tau = np.array([0.005, -0.003, 0.002])
        config = AllocationConfig(method='lp')

        result = allocate_lp(
            tau, B_tau, -rw_3axis.u_max, rw_3axis.u_max,
            3, rw_3axis.indices, config,
        )
        assert result.direction_error < 1e-6
        assert result.alpha > 0.99

    def test_saturated_direction_preserved(self, rw_3axis):
        """Even when saturated, direction error should be zero."""
        B_tau = rw_3axis.axes.copy()
        tau = np.array([0.05, 0.05, 0.05])  # 5x max
        config = AllocationConfig(method='lp')

        result = allocate_lp(
            tau, B_tau, -rw_3axis.u_max, rw_3axis.u_max,
            3, rw_3axis.indices, config,
        )
        assert result.direction_error < 1e-6
        assert result.alpha < 1.0
        assert not result.feasible

    def test_lp_mtq_along_B(self, mtq_3axis):
        """Torque along B is unachievable for MTQ — should project."""
        B = np.array([1.0, 0.0, 0.0])
        B_tau, u_min, u_max = assemble_B_tau([mtq_3axis], B)
        tau = np.array([1.0, 0.0, 0.0])  # along B
        config = AllocationConfig(method='lp', lp_project_when_infeasible=True)

        result = allocate_lp(
            tau, B_tau, u_min, u_max,
            3, mtq_3axis.indices, config,
        )
        # Should return zero since tau is entirely along B (unachievable)
        assert result.alpha < 1e-6

    def test_lp_mtq_perp_to_B(self, mtq_3axis):
        """Torque perpendicular to B is achievable for MTQ."""
        B = np.array([1.0, 0.0, 0.0])
        B_tau, u_min, u_max = assemble_B_tau([mtq_3axis], B)
        tau = np.array([0.0, 0.5, 0.0])  # perp to B
        config = AllocationConfig(method='lp')

        result = allocate_lp(
            tau, B_tau, u_min, u_max,
            3, mtq_3axis.indices, config,
        )
        assert result.alpha > 0.0
        assert result.direction_error < 1e-6

    def test_zero_desired(self, rw_3axis):
        """Zero desired torque => zero commands, alpha=1."""
        config = AllocationConfig(method='lp')
        result = allocate_lp(
            np.zeros(3), rw_3axis.axes, -rw_3axis.u_max, rw_3axis.u_max,
            3, rw_3axis.indices, config,
        )
        assert result.alpha == 1.0
        np.testing.assert_array_almost_equal(result.u, np.zeros(3))

    def test_mixed_rw_mtq(self, mtq_3axis, rw_1axis, B_body_nominal):
        """LP with mixed RW+MTQ actuators."""
        groups = [mtq_3axis, rw_1axis]
        B_tau, u_min, u_max = assemble_B_tau(groups, B_body_nominal)
        indices = np.concatenate([mtq_3axis.indices, rw_1axis.indices])
        tau = np.array([1e-6, 1e-6, 1e-6])
        config = AllocationConfig(method='lp')

        result = allocate_lp(
            tau, B_tau, u_min, u_max, 4, indices, config,
        )
        assert result.alpha > 0.0
        assert result.direction_error < 1e-3


# ---------------------------------------------------------------------------
# Test: qp.py — bounded least-squares
# ---------------------------------------------------------------------------

class TestQP:
    """Tests for QP allocation."""

    def test_exact_solution(self, rw_3axis):
        """Feasible request => exact solution."""
        B_tau = rw_3axis.axes.copy()
        tau = np.array([0.005, -0.003, 0.002])
        config = AllocationConfig(method='qp')

        result = allocate_qp(
            tau, B_tau, -rw_3axis.u_max, rw_3axis.u_max,
            3, rw_3axis.indices, config,
        )
        np.testing.assert_array_almost_equal(result.tau_achieved, tau, decimal=6)
        assert result.feasible

    def test_saturated_minimizes_error(self, rw_3axis):
        """Saturated: QP minimizes ||tau_ach - tau_des||."""
        B_tau = rw_3axis.axes.copy()
        tau = np.array([0.05, 0.0, 0.0])  # 5x max
        config = AllocationConfig(method='qp')

        result = allocate_qp(
            tau, B_tau, -rw_3axis.u_max, rw_3axis.u_max,
            3, rw_3axis.indices, config,
        )
        assert not result.feasible
        # QP should use maximum X torque
        assert abs(result.u[0] - 0.01) < 1e-6

    def test_regularization(self, rw_3axis):
        """Regularization reduces command magnitudes."""
        B_tau = rw_3axis.axes.copy()
        tau = np.array([0.005, 0.0, 0.0])
        config_noreg = AllocationConfig(method='qp', lambda_reg=0.0)
        config_reg = AllocationConfig(method='qp', lambda_reg=10.0)

        result_noreg = allocate_qp(
            tau, B_tau, -rw_3axis.u_max, rw_3axis.u_max,
            3, rw_3axis.indices, config_noreg,
        )
        result_reg = allocate_qp(
            tau, B_tau, -rw_3axis.u_max, rw_3axis.u_max,
            3, rw_3axis.indices, config_reg,
        )
        # Regularized solution should have smaller or equal ||u||
        assert np.linalg.norm(result_reg.u) <= np.linalg.norm(result_noreg.u) + 1e-6

    def test_zero_desired(self, rw_3axis):
        """Zero desired => zero commands."""
        config = AllocationConfig(method='qp')
        result = allocate_qp(
            np.zeros(3), rw_3axis.axes, -rw_3axis.u_max, rw_3axis.u_max,
            3, rw_3axis.indices, config,
        )
        assert result.alpha == 1.0


# ---------------------------------------------------------------------------
# Test: qpw.py — direction-weighted QP
# ---------------------------------------------------------------------------

class TestQPW:
    """Tests for direction-weighted QP allocation."""

    def test_exact_solution(self, rw_3axis):
        """Feasible request => exact solution regardless of weights."""
        B_tau = rw_3axis.axes.copy()
        tau = np.array([0.005, -0.003, 0.002])
        config = AllocationConfig(method='qpw')

        result = allocate_qpw(
            tau, B_tau, -rw_3axis.u_max, rw_3axis.u_max,
            3, rw_3axis.indices, config,
        )
        np.testing.assert_array_almost_equal(result.tau_achieved, tau, decimal=6)

    def test_direction_bias(self, mtq_3axis, B_body_nominal):
        """QPW should have smaller direction error than basic QP."""
        B_tau, u_min, u_max = assemble_B_tau([mtq_3axis], B_body_nominal)
        # Request with some component along B (partially unachievable)
        tau = np.array([1e-4, 2e-4, 1e-4])

        config_qp = AllocationConfig(method='qp')
        config_qpw = AllocationConfig(method='qpw', w_parallel=1.0, w_perpendicular=100.0)

        result_qp = allocate_qp(
            tau, B_tau, u_min, u_max, 3, mtq_3axis.indices, config_qp,
        )
        result_qpw = allocate_qpw(
            tau, B_tau, u_min, u_max, 3, mtq_3axis.indices, config_qpw,
        )
        # QPW should have smaller or equal direction error
        assert result_qpw.direction_error <= result_qp.direction_error + 1e-3

    def test_zero_desired(self, rw_3axis):
        """Zero desired => zero commands."""
        config = AllocationConfig(method='qpw')
        result = allocate_qpw(
            np.zeros(3), rw_3axis.axes, -rw_3axis.u_max, rw_3axis.u_max,
            3, rw_3axis.indices, config,
        )
        assert result.alpha == 1.0


# ---------------------------------------------------------------------------
# Test: qpc.py — energy-constrained QP
# ---------------------------------------------------------------------------

class TestQPC:
    """Tests for energy-constrained QP allocation."""

    def test_exact_feasible(self, rw_3axis):
        """Feasible request with no energy constraint active."""
        B_tau = rw_3axis.axes.copy()
        tau = np.array([0.005, -0.003, 0.002])
        config = AllocationConfig(method='qpc')
        omega = np.array([0.0, 0.0, 0.01])

        result = allocate_qpc(
            tau, B_tau, -rw_3axis.u_max, rw_3axis.u_max,
            3, rw_3axis.indices, config, omega=omega,
        )
        np.testing.assert_array_almost_equal(result.tau_achieved, tau, decimal=4)

    def test_energy_gate_prevents_spin_up(self, rw_3axis):
        """Energy gate should prevent torque that spins up the body."""
        B_tau = rw_3axis.axes.copy()
        # Desired torque opposes rotation => omega^T @ tau_des < 0
        omega = np.array([0.1, 0.0, 0.0])
        tau = np.array([-0.005, 0.0, 0.0])  # braking torque
        config = AllocationConfig(method='qpc')

        result = allocate_qpc(
            tau, B_tau, -rw_3axis.u_max, rw_3axis.u_max,
            3, rw_3axis.indices, config, omega=omega,
        )
        # Achieved torque should not inject energy: omega^T @ tau_ach <= 0
        power = float(np.dot(omega, result.tau_achieved))
        assert power <= 1e-6, f"Energy gate violated: power = {power}"

    def test_no_omega_fallback(self, rw_3axis):
        """Without omega, falls back to unconstrained QP."""
        B_tau = rw_3axis.axes.copy()
        tau = np.array([0.005, 0.0, 0.0])
        config = AllocationConfig(method='qpc')

        result = allocate_qpc(
            tau, B_tau, -rw_3axis.u_max, rw_3axis.u_max,
            3, rw_3axis.indices, config, omega=None,
        )
        np.testing.assert_array_almost_equal(result.tau_achieved, tau, decimal=6)

    def test_zero_desired(self, rw_3axis):
        """Zero desired => zero commands."""
        config = AllocationConfig(method='qpc')
        result = allocate_qpc(
            np.zeros(3), rw_3axis.axes, -rw_3axis.u_max, rw_3axis.u_max,
            3, rw_3axis.indices, config,
        )
        assert result.alpha == 1.0


# ---------------------------------------------------------------------------
# Test: allocator.py — routing
# ---------------------------------------------------------------------------

class TestAllocatorRouting:
    """Tests for the top-level allocation_step router."""

    def test_magnetic_cross_routing(self, mtq_3axis, B_body_nominal):
        """magnetic_cross method routes to the cross-product allocator."""
        config = AllocationConfig(method='magnetic_cross')
        tau = np.array([1e-5, 2e-5, 0.0])
        result = allocation_step(
            tau, [mtq_3axis], config, B_body_nominal, 3,
        )
        assert isinstance(result, AllocationResult)
        assert result.u.shape == (3,)

    def test_lp_routing(self, rw_3axis, B_body_nominal):
        """LP method routes correctly."""
        config = AllocationConfig(method='lp')
        tau = np.array([0.005, 0.0, 0.0])
        result = allocation_step(
            tau, [rw_3axis], config, B_body_nominal, 3,
        )
        assert result.direction_error < 1e-6

    def test_qp_routing(self, rw_3axis, B_body_nominal):
        """QP method routes correctly."""
        config = AllocationConfig(method='qp')
        tau = np.array([0.005, 0.0, 0.0])
        result = allocation_step(
            tau, [rw_3axis], config, B_body_nominal, 3,
        )
        np.testing.assert_array_almost_equal(result.tau_achieved, tau, decimal=6)

    def test_qpw_routing(self, rw_3axis, B_body_nominal):
        """QPW method routes correctly."""
        config = AllocationConfig(method='qpw')
        tau = np.array([0.005, 0.0, 0.0])
        result = allocation_step(
            tau, [rw_3axis], config, B_body_nominal, 3,
        )
        assert isinstance(result, AllocationResult)

    def test_qpc_routing(self, rw_3axis, B_body_nominal):
        """QPC method routes correctly."""
        config = AllocationConfig(method='qpc')
        tau = np.array([0.005, 0.0, 0.0])
        omega = np.array([0.01, 0.0, 0.0])
        result = allocation_step(
            tau, [rw_3axis], config, B_body_nominal, 3, omega=omega,
        )
        assert isinstance(result, AllocationResult)

    def test_pseudoinverse_routing(self, rw_3axis, B_body_nominal):
        """Pseudoinverse method routes correctly."""
        config = AllocationConfig(method='pseudoinverse')
        tau = np.array([0.005, 0.0, 0.0])
        result = allocation_step(
            tau, [rw_3axis], config, B_body_nominal, 3,
        )
        assert isinstance(result, AllocationResult)

    def test_unknown_method_raises(self, rw_3axis, B_body_nominal):
        """Unknown method should raise ValueError."""
        config = AllocationConfig(method='invalid_method')
        with pytest.raises(ValueError, match="Unknown allocation method"):
            allocation_step(
                np.zeros(3), [rw_3axis], config, B_body_nominal, 3,
            )


# ---------------------------------------------------------------------------
# Test: cross-method comparisons
# ---------------------------------------------------------------------------

class TestCrossMethodComparisons:
    """Compare properties across allocation methods."""

    def test_lp_zero_direction_error(self, mtq_3axis, B_body_nominal):
        """LP should always have near-zero direction error."""
        B_tau, u_min, u_max = assemble_B_tau([mtq_3axis], B_body_nominal)
        # Request perpendicular to B (achievable)
        B_hat = B_body_nominal / np.linalg.norm(B_body_nominal)
        # Find a direction perpendicular to B
        perp = np.cross(B_hat, [1, 0, 0])
        if np.linalg.norm(perp) < 0.1:
            perp = np.cross(B_hat, [0, 1, 0])
        perp = perp / np.linalg.norm(perp) * 1e-5
        config = AllocationConfig(method='lp')

        result = allocate_lp(
            perp, B_tau, u_min, u_max, 3, mtq_3axis.indices, config,
        )
        assert result.direction_error < 1e-3

    def test_qp_may_have_direction_error(self, mtq_3axis, B_body_nominal):
        """QP may sacrifice direction to minimize magnitude error."""
        B_tau, u_min, u_max = assemble_B_tau([mtq_3axis], B_body_nominal)
        # Partially achievable request
        tau = np.array([5e-5, 5e-5, 5e-5])
        config = AllocationConfig(method='qp')

        result = allocate_qp(
            tau, B_tau, u_min, u_max, 3, mtq_3axis.indices, config,
        )
        # QP result is valid — just verify it produced something
        assert np.linalg.norm(result.tau_achieved) > 0

    def test_all_methods_respect_bounds(self, mtq_3axis, rw_1axis, B_body_nominal):
        """All methods should produce commands within bounds."""
        groups = [mtq_3axis, rw_1axis]
        B_tau, u_min, u_max = assemble_B_tau(groups, B_body_nominal)
        indices = np.concatenate([mtq_3axis.indices, rw_1axis.indices])
        tau = np.array([1e-4, 2e-4, 3e-4])

        for method in ['lp', 'qp', 'qpw', 'qpc', 'pseudoinverse']:
            config = AllocationConfig(method=method)
            result = allocation_step(
                tau, groups, config, B_body_nominal, 4,
            )
            # Check all commands are within bounds
            for group in groups:
                idx = group.indices
                cmds = result.u[idx]
                assert np.all(cmds >= group.u_min - 1e-10), \
                    f"{method}: commands below u_min"
                assert np.all(cmds <= group.u_max + 1e-10), \
                    f"{method}: commands above u_max"

    def test_all_methods_zero_desired(self, rw_3axis, B_body_nominal):
        """All methods should return zero commands for zero desired torque."""
        for method in ['lp', 'qp', 'qpw', 'qpc', 'pseudoinverse']:
            config = AllocationConfig(method=method)
            result = allocation_step(
                np.zeros(3), [rw_3axis], config, B_body_nominal, 3,
            )
            np.testing.assert_array_almost_equal(
                result.u, np.zeros(3),
                err_msg=f"{method} failed zero-torque test",
            )

    def test_3mtq_1rw_config(self, mtq_3axis, rw_1axis, B_body_nominal):
        """3MTQ + 1RW: all methods produce valid results."""
        groups = [mtq_3axis, rw_1axis]
        tau = np.array([1e-5, 1e-5, 1e-5])

        for method in ['lp', 'qp', 'qpw', 'pseudoinverse']:
            config = AllocationConfig(method=method)
            result = allocation_step(
                tau, groups, config, B_body_nominal, 4,
            )
            assert result.u.shape == (4,)
            assert result.alpha >= 0.0

    def test_3mtq_3rw_full_actuation(self, mtq_3axis, B_body_nominal):
        """3MTQ + 3RW: 6 DOFs for 3 torque axes — overactuated."""
        rw_3 = ActuatorGroup(
            group_type='rw',
            axes=np.eye(3),
            u_max=np.array([0.01, 0.01, 0.01]),
            indices=np.array([3, 4, 5]),
        )
        groups = [mtq_3axis, rw_3]
        tau = np.array([0.005, -0.003, 0.002])

        for method in ['lp', 'qp', 'qpw', 'pseudoinverse']:
            config = AllocationConfig(method=method)
            result = allocation_step(
                tau, groups, config, B_body_nominal, 6,
            )
            assert result.u.shape == (6,)
            # With 6 DOFs, should be feasible
            np.testing.assert_array_almost_equal(
                result.tau_achieved, tau, decimal=4,
                err_msg=f"{method} failed 3MTQ+3RW feasibility",
            )


# ---------------------------------------------------------------------------
# Test: AllocationResult fields
# ---------------------------------------------------------------------------

class TestAllocationResult:
    """Verify AllocationResult fields are populated correctly."""

    def test_fields_present(self, rw_3axis, B_body_nominal):
        """All result fields should be populated."""
        config = AllocationConfig(method='lp')
        result = allocation_step(
            np.array([0.005, 0.0, 0.0]), [rw_3axis], config, B_body_nominal, 3,
        )
        assert hasattr(result, 'u')
        assert hasattr(result, 'tau_achieved')
        assert hasattr(result, 'alpha')
        assert hasattr(result, 'direction_error')
        assert hasattr(result, 'feasible')

    def test_alpha_range(self, rw_3axis, B_body_nominal):
        """Alpha should be non-negative."""
        for method in ['lp', 'qp', 'qpw', 'pseudoinverse']:
            config = AllocationConfig(method=method)
            result = allocation_step(
                np.array([0.005, 0.0, 0.0]), [rw_3axis],
                config, B_body_nominal, 3,
            )
            assert result.alpha >= 0.0, f"{method}: alpha < 0"
