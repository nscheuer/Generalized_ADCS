"""
Phase 5 unit tests: momentum management and actuator failure handling.

Tests cover:
    - momentum.py: compute_desat_torque, nullspace desaturation,
                   weighted desaturation, scheduled desaturation,
                   MTQ authority, coupled scaling
    - actuator_set.py: mask_failed_actuators
    - allocator.py: desaturation integration, failure handling
    - data.py: DesaturationConfig
"""

import numpy as np
import pytest

from ADCS.pipeline.data import (
    ActuatorGroup, AllocationConfig, AllocationResult, DesaturationConfig,
)
from ADCS.pipeline.allocation.actuator_set import assemble_B_tau, mask_failed_actuators
from ADCS.pipeline.allocation.momentum import (
    compute_desat_torque,
    apply_nullspace_desaturation,
    build_weighted_desat_system,
    apply_scheduled_desaturation,
    compute_mtq_authority,
    _compute_coupled_scale,
)
from ADCS.pipeline.allocation.allocator import (
    allocation_step, _build_group_indices, _find_rw_columns,
)
from ADCS.helpers.math_helpers import skewsym


# ---------------------------------------------------------------------------
# Fixtures
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
        indices=np.array([3, 4, 5]),
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


@pytest.fixture
def desat_config_default():
    """Default desaturation config."""
    return DesaturationConfig(
        strategy='nullspace',
        k_desat=0.01,
    )


@pytest.fixture
def groups_mtq_rw(mtq_3axis, rw_3axis):
    """Combined 3MTQ + 3RW actuator groups."""
    return [mtq_3axis, rw_3axis]


# ---------------------------------------------------------------------------
# Test: DesaturationConfig dataclass
# ---------------------------------------------------------------------------

class TestDesaturationConfig:

    def test_defaults(self):
        """Default h_rw_target should be zero vector."""
        cfg = DesaturationConfig()
        np.testing.assert_array_equal(cfg.h_rw_target, np.zeros(3))
        assert cfg.strategy == 'nullspace'
        assert cfg.k_desat == 0.01

    def test_custom_target(self):
        """Custom h_rw_target should be preserved."""
        h_tgt = np.array([0.001, 0.0, -0.001])
        cfg = DesaturationConfig(h_rw_target=h_tgt)
        np.testing.assert_array_equal(cfg.h_rw_target, h_tgt)


# ---------------------------------------------------------------------------
# Test: compute_desat_torque
# ---------------------------------------------------------------------------

class TestComputeDesatTorque:

    def test_zero_momentum(self, desat_config_default):
        """Zero RW momentum -> zero desat torque."""
        tau = compute_desat_torque(np.zeros(3), desat_config_default)
        np.testing.assert_array_almost_equal(tau, np.zeros(3))

    def test_positive_momentum(self, desat_config_default):
        """Positive h_rw -> negative desat torque (drives momentum down)."""
        h_rw = np.array([0.1, 0.0, 0.0])
        tau = compute_desat_torque(h_rw, desat_config_default)
        assert tau[0] < 0.0
        np.testing.assert_array_almost_equal(
            tau, -0.01 * h_rw,
        )

    def test_below_threshold(self):
        """Below h_rw_threshold -> zero desat torque."""
        cfg = DesaturationConfig(k_desat=0.01, h_rw_threshold=0.5)
        h_rw = np.array([0.1, 0.0, 0.0])  # norm=0.1 < 0.5
        tau = compute_desat_torque(h_rw, cfg)
        np.testing.assert_array_almost_equal(tau, np.zeros(3))

    def test_above_threshold(self):
        """Above h_rw_threshold -> nonzero desat torque."""
        cfg = DesaturationConfig(k_desat=0.01, h_rw_threshold=0.05)
        h_rw = np.array([0.1, 0.0, 0.0])  # norm=0.1 > 0.05
        tau = compute_desat_torque(h_rw, cfg)
        assert np.linalg.norm(tau) > 0.0

    def test_custom_target(self):
        """Desat torque uses (h_rw - h_target), not just h_rw."""
        h_tgt = np.array([0.05, 0.0, 0.0])
        cfg = DesaturationConfig(k_desat=0.01, h_rw_target=h_tgt)
        h_rw = np.array([0.1, 0.0, 0.0])
        tau = compute_desat_torque(h_rw, cfg)
        expected = -0.01 * (h_rw - h_tgt)
        np.testing.assert_array_almost_equal(tau, expected)


# ---------------------------------------------------------------------------
# Test: _compute_coupled_scale
# ---------------------------------------------------------------------------

class TestCoupledScale:

    def test_within_bounds(self):
        """If secondary fits entirely within bounds, beta=1."""
        u_pri = np.array([0.0, 0.0, 0.0])
        u_sec = np.array([0.001, 0.001, 0.001])
        u_min = np.full(3, -0.01)
        u_max = np.full(3, 0.01)
        beta = _compute_coupled_scale(u_pri, u_sec, u_min, u_max)
        assert beta == 1.0

    def test_exceeds_bounds(self):
        """Secondary exceeds bounds -> beta < 1."""
        u_pri = np.array([0.005, 0.0, 0.0])
        u_sec = np.array([0.01, 0.0, 0.0])  # would push to 0.015 > 0.01
        u_min = np.full(3, -0.01)
        u_max = np.full(3, 0.01)
        beta = _compute_coupled_scale(u_pri, u_sec, u_min, u_max)
        assert 0.0 < beta < 1.0
        # beta * 0.01 + 0.005 <= 0.01 -> beta <= 0.5
        assert abs(beta - 0.5) < 1e-12

    def test_zero_secondary(self):
        """Zero secondary -> beta=1."""
        u_pri = np.array([0.005, 0.0, 0.0])
        u_sec = np.zeros(3)
        u_min = np.full(3, -0.01)
        u_max = np.full(3, 0.01)
        beta = _compute_coupled_scale(u_pri, u_sec, u_min, u_max)
        assert beta == 1.0

    def test_negative_direction(self):
        """Negative secondary direction respects lower bound."""
        u_pri = np.array([-0.005, 0.0, 0.0])
        u_sec = np.array([-0.01, 0.0, 0.0])
        u_min = np.full(3, -0.01)
        u_max = np.full(3, 0.01)
        beta = _compute_coupled_scale(u_pri, u_sec, u_min, u_max)
        assert abs(beta - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# Test: apply_nullspace_desaturation
# ---------------------------------------------------------------------------

class TestNullspaceDesaturation:

    def test_overactuated_no_torque_impact(self, mtq_3axis, rw_3axis, B_body_nominal):
        """Nullspace desat should not change achieved torque (overactuated)."""
        groups = [mtq_3axis, rw_3axis]
        B_tau, u_min, u_max = assemble_B_tau(groups, B_body_nominal)

        # Primary solution: simple QP result
        tau_desired = np.array([1e-6, 2e-6, -1e-6])
        u_primary = np.linalg.pinv(B_tau) @ tau_desired
        u_primary = np.clip(u_primary, u_min, u_max)
        tau_primary = B_tau @ u_primary

        # Apply nullspace desat
        h_rw = np.array([0.05, 0.02, -0.03])
        cfg = DesaturationConfig(k_desat=0.01, strategy='nullspace')
        rw_columns = np.array([3, 4, 5])  # RW columns in B_tau (after 3 MTQ cols)

        u_desat = apply_nullspace_desaturation(
            u_primary, B_tau, u_min, u_max,
            h_rw, rw_columns, cfg,
        )

        tau_after = B_tau @ u_desat
        # Torque should be (nearly) unchanged
        np.testing.assert_array_almost_equal(tau_after, tau_primary, decimal=8)

    def test_square_system_no_nullspace(self, rw_3axis, B_body_nominal):
        """Square system (3x3) has no nullspace -> u unchanged."""
        groups = [rw_3axis]
        B_tau, u_min, u_max = assemble_B_tau(groups, B_body_nominal)

        u_primary = np.array([0.001, -0.002, 0.003])
        h_rw = np.array([0.05, 0.0, 0.0])
        cfg = DesaturationConfig(k_desat=0.01, strategy='nullspace')
        rw_columns = np.array([0, 1, 2])

        u_desat = apply_nullspace_desaturation(
            u_primary, B_tau, u_min, u_max,
            h_rw, rw_columns, cfg,
        )

        np.testing.assert_array_almost_equal(u_desat, u_primary)

    def test_zero_momentum_no_change(self, mtq_3axis, rw_3axis, B_body_nominal):
        """Zero RW momentum -> no desat applied."""
        groups = [mtq_3axis, rw_3axis]
        B_tau, u_min, u_max = assemble_B_tau(groups, B_body_nominal)

        u_primary = np.zeros(6)
        h_rw = np.zeros(3)
        cfg = DesaturationConfig(k_desat=0.01, strategy='nullspace')
        rw_columns = np.array([3, 4, 5])

        u_desat = apply_nullspace_desaturation(
            u_primary, B_tau, u_min, u_max,
            h_rw, rw_columns, cfg,
        )

        np.testing.assert_array_almost_equal(u_desat, u_primary)

    def test_respects_bounds(self, mtq_3axis, rw_3axis, B_body_nominal):
        """Desat result respects actuator bounds (coupled scaling)."""
        groups = [mtq_3axis, rw_3axis]
        B_tau, u_min, u_max = assemble_B_tau(groups, B_body_nominal)

        u_primary = np.zeros(6)
        h_rw = np.array([10.0, 10.0, 10.0])  # large momentum
        cfg = DesaturationConfig(k_desat=1.0, strategy='nullspace')  # aggressive gain
        rw_columns = np.array([3, 4, 5])

        u_desat = apply_nullspace_desaturation(
            u_primary, B_tau, u_min, u_max,
            h_rw, rw_columns, cfg,
        )

        assert np.all(u_desat >= u_min - 1e-10)
        assert np.all(u_desat <= u_max + 1e-10)


# ---------------------------------------------------------------------------
# Test: build_weighted_desat_system
# ---------------------------------------------------------------------------

class TestWeightedDesatSystem:

    def test_augmented_system_shape(self, mtq_3axis, rw_3axis, B_body_nominal):
        """Augmented system should be [6 x n]."""
        groups = [mtq_3axis, rw_3axis]
        B_tau, u_min, u_max = assemble_B_tau(groups, B_body_nominal)

        h_rw = np.array([0.05, 0.0, 0.0])
        cfg = DesaturationConfig(strategy='weighted', k_desat=0.01, w_desat=1.0)

        A_aug, b_aug = build_weighted_desat_system(
            B_tau, np.array([1e-6, 0, 0]), u_min, u_max,
            h_rw, groups, cfg,
        )

        assert A_aug.shape == (6, B_tau.shape[1])
        assert b_aug.shape == (6,)

    def test_top_rows_match_B_tau(self, mtq_3axis, rw_3axis, B_body_nominal):
        """Top 3 rows of A_aug should equal B_tau."""
        groups = [mtq_3axis, rw_3axis]
        B_tau, u_min, u_max = assemble_B_tau(groups, B_body_nominal)

        h_rw = np.array([0.05, 0.0, 0.0])
        cfg = DesaturationConfig(strategy='weighted', k_desat=0.01, w_desat=1.0)
        tau_des = np.array([1e-6, 0, 0])

        A_aug, b_aug = build_weighted_desat_system(
            B_tau, tau_des, u_min, u_max,
            h_rw, groups, cfg,
        )

        np.testing.assert_array_almost_equal(A_aug[:3, :], B_tau)
        np.testing.assert_array_almost_equal(b_aug[:3], tau_des)

    def test_bottom_rows_rw_only(self, mtq_3axis, rw_3axis, B_body_nominal):
        """Bottom 3 rows of A_aug should have nonzero entries only at RW columns."""
        groups = [mtq_3axis, rw_3axis]
        B_tau, u_min, u_max = assemble_B_tau(groups, B_body_nominal)

        h_rw = np.array([0.05, 0.0, 0.0])
        cfg = DesaturationConfig(strategy='weighted', k_desat=0.01, w_desat=1.0)

        A_aug, b_aug = build_weighted_desat_system(
            B_tau, np.array([1e-6, 0, 0]), u_min, u_max,
            h_rw, groups, cfg,
        )

        # MTQ columns (first 3) should be zero in desat rows
        np.testing.assert_array_almost_equal(A_aug[3:, :3], np.zeros((3, 3)))
        # RW columns (last 3) should be nonzero
        assert np.linalg.norm(A_aug[3:, 3:]) > 0

    def test_w_desat_scales_bottom_rows(self, mtq_3axis, rw_3axis, B_body_nominal):
        """w_desat should scale the bottom rows by sqrt(w_desat)."""
        groups = [mtq_3axis, rw_3axis]
        B_tau, u_min, u_max = assemble_B_tau(groups, B_body_nominal)
        h_rw = np.array([0.05, 0.0, 0.0])

        cfg1 = DesaturationConfig(strategy='weighted', k_desat=0.01, w_desat=1.0)
        cfg4 = DesaturationConfig(strategy='weighted', k_desat=0.01, w_desat=4.0)

        A1, _ = build_weighted_desat_system(
            B_tau, np.array([1e-6, 0, 0]), u_min, u_max, h_rw, groups, cfg1,
        )
        A4, _ = build_weighted_desat_system(
            B_tau, np.array([1e-6, 0, 0]), u_min, u_max, h_rw, groups, cfg4,
        )

        # Bottom rows scale by sqrt(4)/sqrt(1) = 2
        np.testing.assert_array_almost_equal(A4[3:, :], 2.0 * A1[3:, :])


# ---------------------------------------------------------------------------
# Test: compute_mtq_authority
# ---------------------------------------------------------------------------

class TestMTQAuthority:

    def test_no_mtq(self, rw_3axis):
        """No MTQ group -> authority = 0."""
        authority = compute_mtq_authority(np.array([0, 0, 3e-5]), [rw_3axis])
        assert authority == 0.0

    def test_zero_B_field(self, mtq_3axis):
        """Zero B field -> authority = 0."""
        authority = compute_mtq_authority(np.zeros(3), [mtq_3axis])
        assert authority == 0.0

    def test_nonzero_authority(self, mtq_3axis, B_body_nominal):
        """General B field -> some authority in (0, 1]."""
        authority = compute_mtq_authority(B_body_nominal, [mtq_3axis])
        assert 0.0 < authority <= 1.0

    def test_perpendicular_gives_high_authority(self, mtq_3axis):
        """B perpendicular to some axes gives good authority."""
        B_body = np.array([1e-5, 1e-5, 0.0])  # perpendicular to Z MTQ
        authority = compute_mtq_authority(B_body, [mtq_3axis])
        assert authority > 0.0


# ---------------------------------------------------------------------------
# Test: apply_scheduled_desaturation
# ---------------------------------------------------------------------------

class TestScheduledDesaturation:

    def test_low_authority_no_desat(self, mtq_3axis, rw_3axis):
        """Low MTQ authority -> tau_desired unchanged."""
        groups = [mtq_3axis, rw_3axis]
        tau_desired = np.array([1e-6, 0, 0])
        h_rw = np.array([0.1, 0.0, 0.0])
        B_body = np.array([1e-15, 1e-15, 1e-15])  # near-zero B -> no authority
        cfg = DesaturationConfig(
            strategy='scheduled', k_desat=0.01, authority_threshold=0.1,
        )

        result = apply_scheduled_desaturation(
            tau_desired, h_rw, B_body, groups, cfg,
        )

        np.testing.assert_array_almost_equal(result, tau_desired)

    def test_high_authority_adds_desat(self, mtq_3axis, rw_3axis, B_body_nominal):
        """High MTQ authority -> desat torque added to tau_desired."""
        groups = [mtq_3axis, rw_3axis]
        tau_desired = np.array([1e-6, 0, 0])
        h_rw = np.array([0.1, 0.0, 0.0])
        cfg = DesaturationConfig(
            strategy='scheduled', k_desat=0.01,
            authority_threshold=0.0,  # always active
        )

        result = apply_scheduled_desaturation(
            tau_desired, h_rw, B_body_nominal, groups, cfg,
        )

        tau_desat = -0.01 * h_rw
        expected = tau_desired + tau_desat
        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_momentum_no_change(self, mtq_3axis, rw_3axis, B_body_nominal):
        """Zero momentum -> tau_desired unchanged."""
        groups = [mtq_3axis, rw_3axis]
        tau_desired = np.array([1e-6, 0, 0])
        h_rw = np.zeros(3)
        cfg = DesaturationConfig(strategy='scheduled', k_desat=0.01)

        result = apply_scheduled_desaturation(
            tau_desired, h_rw, B_body_nominal, groups, cfg,
        )

        np.testing.assert_array_almost_equal(result, tau_desired)


# ---------------------------------------------------------------------------
# Test: mask_failed_actuators
# ---------------------------------------------------------------------------

class TestMaskFailedActuators:

    def test_no_failures(self, mtq_3axis, rw_3axis, B_body_nominal):
        """No failures -> B_tau unchanged."""
        groups = [mtq_3axis, rw_3axis]
        B_tau, u_min, u_max = assemble_B_tau(groups, B_body_nominal)
        group_indices = _build_group_indices(groups)

        B_tau2, u_min2, u_max2, gi2 = mask_failed_actuators(
            B_tau, u_min, u_max,
            failed_indices=np.array([], dtype=int),
            group_indices=group_indices,
        )

        np.testing.assert_array_equal(B_tau2, B_tau)
        np.testing.assert_array_equal(u_min2, u_min)
        np.testing.assert_array_equal(u_max2, u_max)
        np.testing.assert_array_equal(gi2, group_indices)

    def test_fail_one_rw(self, mtq_3axis, rw_3axis, B_body_nominal):
        """Failing one RW removes its column entirely."""
        groups = [mtq_3axis, rw_3axis]
        B_tau, u_min, u_max = assemble_B_tau(groups, B_body_nominal)
        group_indices = _build_group_indices(groups)

        # Fail RW at full-command index 3 (first RW)
        B_tau2, u_min2, u_max2, gi2 = mask_failed_actuators(
            B_tau, u_min, u_max,
            failed_indices=np.array([3]),
            group_indices=group_indices,
        )

        # Should have one fewer column
        assert B_tau2.shape[1] == B_tau.shape[1] - 1
        # Full-command index 3 should not appear in group_indices
        assert 3 not in gi2

    def test_fail_all_mtqs(self, mtq_3axis, rw_3axis, B_body_nominal):
        """Failing all MTQs removes their columns, only RW remains."""
        groups = [mtq_3axis, rw_3axis]
        B_tau, u_min, u_max = assemble_B_tau(groups, B_body_nominal)
        group_indices = _build_group_indices(groups)

        B_tau2, u_min2, u_max2, gi2 = mask_failed_actuators(
            B_tau, u_min, u_max,
            failed_indices=np.array([0, 1, 2]),
            group_indices=group_indices,
        )

        # Only RW columns remain (3 columns)
        assert B_tau2.shape[1] == 3
        # RW columns should match original RW columns
        np.testing.assert_array_equal(B_tau2, B_tau[:, 3:])
        np.testing.assert_array_equal(gi2, np.array([3, 4, 5]))

    def test_original_unchanged(self, mtq_3axis, rw_3axis, B_body_nominal):
        """mask_failed_actuators should not modify the original arrays."""
        groups = [mtq_3axis, rw_3axis]
        B_tau, u_min, u_max = assemble_B_tau(groups, B_body_nominal)
        group_indices = _build_group_indices(groups)
        B_tau_orig = B_tau.copy()

        mask_failed_actuators(
            B_tau, u_min, u_max,
            failed_indices=np.array([3]),
            group_indices=group_indices,
        )

        np.testing.assert_array_equal(B_tau, B_tau_orig)


# ---------------------------------------------------------------------------
# Test: _find_rw_columns
# ---------------------------------------------------------------------------

class TestFindRWColumns:

    def test_mtq_then_rw(self, mtq_3axis, rw_3axis):
        """MTQ first, then RW -> RW columns are [3,4,5]."""
        groups = [mtq_3axis, rw_3axis]
        rw_cols = _find_rw_columns(groups)
        np.testing.assert_array_equal(rw_cols, [3, 4, 5])

    def test_rw_only(self, rw_3axis):
        """RW only -> columns are [0,1,2]."""
        groups = [rw_3axis]
        rw_cols = _find_rw_columns(groups)
        np.testing.assert_array_equal(rw_cols, [0, 1, 2])

    def test_mtq_only(self, mtq_3axis):
        """MTQ only -> no RW columns."""
        groups = [mtq_3axis]
        rw_cols = _find_rw_columns(groups)
        assert len(rw_cols) == 0


# ---------------------------------------------------------------------------
# Test: allocation_step with desaturation
# ---------------------------------------------------------------------------

class TestAllocatorDesatIntegration:

    def test_nullspace_desat_via_allocator(
        self, mtq_3axis, rw_3axis, B_body_nominal,
    ):
        """allocation_step with nullspace desat produces valid output."""
        groups = [mtq_3axis, rw_3axis]
        tau_desired = np.array([1e-6, 2e-6, -1e-6])
        h_rw = np.array([0.05, 0.0, 0.0])

        config = AllocationConfig(
            method='qp',
            enable_desaturation=True,
            desat_config=DesaturationConfig(
                strategy='nullspace', k_desat=0.01,
            ),
        )

        result = allocation_step(
            tau_desired=tau_desired,
            actuator_groups=groups,
            alloc_config=config,
            B_body=B_body_nominal,
            n_actuators=6,
            h_rw_body=h_rw,
        )

        assert isinstance(result, AllocationResult)
        assert result.u.shape == (6,)

    def test_weighted_desat_via_allocator(
        self, mtq_3axis, rw_3axis, B_body_nominal,
    ):
        """allocation_step with weighted desat produces valid output."""
        groups = [mtq_3axis, rw_3axis]
        tau_desired = np.array([1e-6, 2e-6, -1e-6])
        h_rw = np.array([0.05, 0.0, 0.0])

        config = AllocationConfig(
            method='qp',
            enable_desaturation=True,
            desat_config=DesaturationConfig(
                strategy='weighted', k_desat=0.01, w_desat=0.5,
            ),
        )

        result = allocation_step(
            tau_desired=tau_desired,
            actuator_groups=groups,
            alloc_config=config,
            B_body=B_body_nominal,
            n_actuators=6,
            h_rw_body=h_rw,
        )

        assert isinstance(result, AllocationResult)
        assert result.u.shape == (6,)

    def test_scheduled_desat_via_allocator(
        self, mtq_3axis, rw_3axis, B_body_nominal,
    ):
        """allocation_step with scheduled desat produces valid output."""
        groups = [mtq_3axis, rw_3axis]
        tau_desired = np.array([1e-6, 2e-6, -1e-6])
        h_rw = np.array([0.05, 0.0, 0.0])

        config = AllocationConfig(
            method='lp',
            enable_desaturation=True,
            desat_config=DesaturationConfig(
                strategy='scheduled', k_desat=0.01,
                authority_threshold=0.0,  # always active
            ),
        )

        result = allocation_step(
            tau_desired=tau_desired,
            actuator_groups=groups,
            alloc_config=config,
            B_body=B_body_nominal,
            n_actuators=6,
            h_rw_body=h_rw,
        )

        assert isinstance(result, AllocationResult)
        assert result.u.shape == (6,)

    def test_desat_disabled_by_default(
        self, mtq_3axis, rw_3axis, B_body_nominal,
    ):
        """Without enable_desaturation, h_rw_body is ignored."""
        groups = [mtq_3axis, rw_3axis]
        tau_desired = np.array([1e-6, 2e-6, -1e-6])

        config = AllocationConfig(method='qp', enable_desaturation=False)

        result_no_h = allocation_step(
            tau_desired=tau_desired,
            actuator_groups=groups,
            alloc_config=config,
            B_body=B_body_nominal,
            n_actuators=6,
        )

        result_with_h = allocation_step(
            tau_desired=tau_desired,
            actuator_groups=groups,
            alloc_config=config,
            B_body=B_body_nominal,
            n_actuators=6,
            h_rw_body=np.array([1.0, 1.0, 1.0]),
        )

        np.testing.assert_array_almost_equal(result_no_h.u, result_with_h.u)

    def test_no_desat_config_means_no_desat(
        self, mtq_3axis, rw_3axis, B_body_nominal,
    ):
        """enable_desaturation=True but desat_config=None -> no desat."""
        groups = [mtq_3axis, rw_3axis]
        tau_desired = np.array([1e-6, 2e-6, -1e-6])

        config = AllocationConfig(
            method='qp',
            enable_desaturation=True,
            desat_config=None,
        )

        result = allocation_step(
            tau_desired=tau_desired,
            actuator_groups=groups,
            alloc_config=config,
            B_body=B_body_nominal,
            n_actuators=6,
            h_rw_body=np.array([1.0, 1.0, 1.0]),
        )

        assert isinstance(result, AllocationResult)
        assert result.u.shape == (6,)


# ---------------------------------------------------------------------------
# Test: allocation_step with actuator failure
# ---------------------------------------------------------------------------

class TestAllocatorFailureIntegration:

    def test_fail_one_rw_still_allocates(
        self, mtq_3axis, rw_3axis, B_body_nominal,
    ):
        """Failing one RW -> allocation still works with remaining actuators."""
        groups = [mtq_3axis, rw_3axis]
        tau_desired = np.array([1e-6, 0, 0])

        config = AllocationConfig(method='qp')

        result = allocation_step(
            tau_desired=tau_desired,
            actuator_groups=groups,
            alloc_config=config,
            B_body=B_body_nominal,
            n_actuators=6,
            failed_actuators=np.array([3]),  # fail first RW
        )

        assert isinstance(result, AllocationResult)
        # Failed actuator should have zero command
        assert result.u[3] == 0.0

    def test_fail_all_mtqs_rw_still_works(
        self, mtq_3axis, rw_3axis, B_body_nominal,
    ):
        """Failing all MTQs -> only RWs produce torque."""
        groups = [mtq_3axis, rw_3axis]
        tau_desired = np.array([1e-6, 0, 0])

        config = AllocationConfig(method='qp')

        result = allocation_step(
            tau_desired=tau_desired,
            actuator_groups=groups,
            alloc_config=config,
            B_body=B_body_nominal,
            n_actuators=6,
            failed_actuators=np.array([0, 1, 2]),
        )

        # MTQ commands should be zero
        np.testing.assert_array_almost_equal(result.u[:3], np.zeros(3))
        # Some RW should be nonzero
        assert np.linalg.norm(result.u[3:]) > 0

    def test_no_failures_same_as_normal(
        self, mtq_3axis, rw_3axis, B_body_nominal,
    ):
        """No failures -> same result as without failed_actuators arg."""
        groups = [mtq_3axis, rw_3axis]
        tau_desired = np.array([1e-6, 2e-6, -1e-6])
        config = AllocationConfig(method='qp')

        result_normal = allocation_step(
            tau_desired=tau_desired,
            actuator_groups=groups,
            alloc_config=config,
            B_body=B_body_nominal,
            n_actuators=6,
        )

        result_empty = allocation_step(
            tau_desired=tau_desired,
            actuator_groups=groups,
            alloc_config=config,
            B_body=B_body_nominal,
            n_actuators=6,
            failed_actuators=np.array([], dtype=int),
        )

        np.testing.assert_array_almost_equal(result_normal.u, result_empty.u)
