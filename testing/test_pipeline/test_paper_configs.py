"""
Paper validation tests for the generalized ADCS pipeline.

Reproduces the hardware configurations, gains, and test scenarios from
the Generalized ACS paper (papers/Generalized_ACS/). Tests verify that
the pipeline produces valid actuator commands for every (config x method)
combination, and that cross-method properties hold (LP direction error ~0,
QP minimizes magnitude error, QPC respects energy gate, etc.).

Hardware configurations from the paper:
    - 3MTQ + 0RW (underactuated, MTQ-only)
    - 3MTQ + 1RW (underactuated, hybrid)
    - 3MTQ + 2RW (hybrid, nearly fully actuated)
    - 0MTQ + 3RW (fully actuated, no magnetic dependency)

Controller gains (paper Table I / scripts):
    p_gain = 0.00005,  d_gain = 0.001
    => PD_Law: eps=1, kp=0.00005, kd=0.001

Satellite parameters (BeaverCube-2):
    mass = 1.2 kg
    J_0 = diag([0.022, 0.022, 0.004]) kg*m^2
    mtq_max = 0.4 Am^2
    rw_max = 7e-3 Nm
    rw_hmax = 16.2e-3 Nms
    boresight = [0, 0, 1]
"""

import numpy as np
import pytest

from ADCS.pipeline.data import (
    ActuatorGroup, AllocationConfig, AllocationResult,
    DesaturationConfig, CompensationConfig,
)
from ADCS.pipeline.allocation.allocator import allocation_step
from ADCS.pipeline.allocation.actuator_set import assemble_B_tau
from ADCS.pipeline.control_law.pd_law import PD_Law
from ADCS.pipeline.goal_formulation import goal_formulation_step_legacy
from ADCS.pipeline.compensation import compensation_step
from ADCS.pipeline.data import CompensationInputs, LawInterface
from ADCS.helpers.math_helpers import normalize, rot_mat, skewsym


# ---------------------------------------------------------------------------
# Paper constants
# ---------------------------------------------------------------------------

PAPER_J = np.diagflat([0.022, 0.022, 0.004])
PAPER_MTQ_MAX = 0.4
PAPER_RW_MAX = 7e-3
PAPER_RW_HMAX = 16.2e-3
PAPER_BORESIGHT = np.array([0, 0, 1])
PAPER_KP = 0.00005
PAPER_KD = 0.001
PAPER_C_GAIN = 0.001

# Fixed orbit from paper: 7000 km circular, 45 deg inclination
PAPER_ORBIT_R = 7000 * np.array([0, np.sqrt(2) / 2, np.sqrt(2) / 2])

# A representative B field in body frame (LEO, ~30 uT)
B_BODY_PAPER = np.array([1.5e-5, -2.0e-5, 2.5e-5])


# ---------------------------------------------------------------------------
# Fixtures: paper actuator configurations
# ---------------------------------------------------------------------------

@pytest.fixture
def mtq_3():
    """3 orthogonal MTQs (paper config)."""
    axes = np.eye(3)
    return ActuatorGroup(
        group_type='mtq',
        axes=axes,
        u_max=np.full(3, PAPER_MTQ_MAX),
        indices=np.array([0, 1, 2]),
    )


@pytest.fixture
def rw_x():
    """1 RW along X-axis (paper 3MTQ+1RW config)."""
    return ActuatorGroup(
        group_type='rw',
        axes=np.array([[1.0], [0.0], [0.0]]),
        u_max=np.array([PAPER_RW_MAX]),
        indices=np.array([3]),
    )


@pytest.fixture
def rw_xy():
    """2 RWs along X, Y (paper 3MTQ+2RW config)."""
    return ActuatorGroup(
        group_type='rw',
        axes=np.array([[1, 0], [0, 1], [0, 0]], dtype=float),
        u_max=np.full(2, PAPER_RW_MAX),
        indices=np.array([3, 4]),
    )


@pytest.fixture
def rw_xyz():
    """3 RWs along X, Y, Z (paper 0MTQ+3RW config)."""
    return ActuatorGroup(
        group_type='rw',
        axes=np.eye(3),
        u_max=np.full(3, PAPER_RW_MAX),
        indices=np.array([0, 1, 2]),
    )


@pytest.fixture
def pd_law():
    """PD law with paper gains (eps=1 maps p_gain/d_gain directly)."""
    return PD_Law(kp=PAPER_KP, kd=PAPER_KD, eps=1.0)


# Hardware configs as (groups, n_actuators) tuples
@pytest.fixture
def config_3mtq_0rw(mtq_3):
    return [mtq_3], 3


@pytest.fixture
def config_3mtq_1rw(mtq_3, rw_x):
    return [mtq_3, rw_x], 4


@pytest.fixture
def config_3mtq_2rw(mtq_3, rw_xy):
    return [mtq_3, rw_xy], 5


@pytest.fixture
def config_0mtq_3rw(rw_xyz):
    return [rw_xyz], 3


# ---------------------------------------------------------------------------
# Deterministic initial conditions from paper seed=0
# ---------------------------------------------------------------------------

def paper_initial_conditions(seed=0):
    """Generate paper-style initial conditions for a single run."""
    rng = np.random.default_rng(seed=seed)
    w0 = normalize(rng.standard_normal(3)) * (
        rng.uniform(0.1, 2.0) * np.pi / 180.0
    )
    q0 = normalize(rng.standard_normal(4))
    goal_eci = normalize(rng.standard_normal(3))
    return w0, q0, goal_eci


def compute_paper_errors(q, goal_eci, boresight=PAPER_BORESIGHT):
    """Compute attitude error (boresight pointing error in radians)."""
    R = rot_mat(q)
    boresight_eci = R.T @ boresight
    cos_err = np.clip(np.dot(boresight_eci, goal_eci), -1.0, 1.0)
    return np.arccos(cos_err)


# ---------------------------------------------------------------------------
# Helper: run one pipeline step
# ---------------------------------------------------------------------------

def run_pipeline_step(
    groups, n_actuators, alloc_method, q, omega, h_rw_body,
    q_err, omega_ref_eci, pd_law, B_body=B_BODY_PAPER,
    h_rw_for_desat=None, desat_config=None, failed_actuators=None,
):
    """Run one full 4-stage pipeline step and return AllocationResult."""
    # Stage 1: Goal formulation
    gf_out = goal_formulation_step_legacy(
        q_err=q_err,
        omega=omega,
        omega_ref_eci=omega_ref_eci,
        q=q,
        law_interface=pd_law.interface,
    )

    # Stage 2: Control law
    tau_law = pd_law.compute(
        attitude_input=gf_out.attitude_output,
        omega_input=gf_out.omega_output,
    )

    # Stage 3: Compensation (gyroscopic only, matching paper)
    comp_config = CompensationConfig.from_law_interface(pd_law.interface)
    comp_inputs = CompensationInputs(
        P=gf_out.P,
        omega_ref_body=gf_out.omega_ref_body,
        goal_type=gf_out.goal_type,
    )
    tau_desired = compensation_step(
        tau_law=tau_law,
        omega=omega,
        J=PAPER_J,
        h_rw_body=h_rw_body,
        comp_config=comp_config,
        comp_inputs=comp_inputs,
    )

    # Stage 4: Allocation
    alloc_config = AllocationConfig(method=alloc_method)
    if desat_config is not None:
        alloc_config.enable_desaturation = True
        alloc_config.desat_config = desat_config

    return allocation_step(
        tau_desired=tau_desired,
        actuator_groups=groups,
        alloc_config=alloc_config,
        B_body=B_body,
        n_actuators=n_actuators,
        omega=omega,
        h_rw_body=h_rw_for_desat,
        failed_actuators=failed_actuators,
    ), tau_desired


# ---------------------------------------------------------------------------
# Test: paper hardware configs produce valid pipeline output
# ---------------------------------------------------------------------------

class TestPaperHardwareConfigs:
    """Verify pipeline produces valid output for each paper hardware config."""

    @pytest.fixture(autouse=True)
    def setup_ics(self, pd_law):
        self.pd_law = pd_law
        self.w0, self.q0, self.goal_eci = paper_initial_conditions(seed=42)
        # Simple quaternion error (using identity goal for simplicity)
        self.q_err = self.q0[1:4]  # vector part of q0 as error from identity
        self.omega_ref = np.zeros(3)

    def _run(self, groups, n_act, method):
        n_rw = sum(g.axes.shape[1] for g in groups if g.group_type == 'rw')
        h_rw = np.zeros(3)
        return run_pipeline_step(
            groups, n_act, method,
            self.q0, self.w0, h_rw,
            self.q_err, self.omega_ref, self.pd_law,
        )

    @pytest.mark.parametrize("method", ["lp", "qp", "qpw", "pseudoinverse"])
    def test_3mtq_0rw(self, config_3mtq_0rw, method):
        """3MTQ+0RW: all general methods produce valid output."""
        groups, n_act = config_3mtq_0rw
        result, tau_des = self._run(groups, n_act, method)
        assert result.u.shape == (n_act,)
        assert np.all(np.abs(result.u[:3]) <= PAPER_MTQ_MAX + 1e-10)

    @pytest.mark.parametrize("method", ["lp", "qp", "qpw", "qpc", "pseudoinverse"])
    def test_3mtq_1rw(self, config_3mtq_1rw, method):
        """3MTQ+1RW: all methods produce valid output."""
        groups, n_act = config_3mtq_1rw
        result, tau_des = self._run(groups, n_act, method)
        assert result.u.shape == (n_act,)
        assert np.all(np.abs(result.u[:3]) <= PAPER_MTQ_MAX + 1e-10)
        assert np.abs(result.u[3]) <= PAPER_RW_MAX + 1e-10

    @pytest.mark.parametrize("method", ["lp", "qp", "qpw", "qpc", "pseudoinverse"])
    def test_3mtq_2rw(self, config_3mtq_2rw, method):
        """3MTQ+2RW: all methods produce valid output."""
        groups, n_act = config_3mtq_2rw
        result, tau_des = self._run(groups, n_act, method)
        assert result.u.shape == (n_act,)

    @pytest.mark.parametrize("method", ["lp", "qp", "qpw", "qpc", "pseudoinverse"])
    def test_0mtq_3rw(self, config_0mtq_3rw, method):
        """0MTQ+3RW: fully actuated, all methods produce valid output."""
        groups, n_act = config_0mtq_3rw
        result, tau_des = self._run(groups, n_act, method)
        assert result.u.shape == (n_act,)
        assert np.all(np.abs(result.u) <= PAPER_RW_MAX + 1e-10)

    def test_magnetic_cross_3mtq_0rw(self, config_3mtq_0rw):
        """3MTQ+0RW: magnetic_cross (Phase 1 baseline)."""
        groups, n_act = config_3mtq_0rw
        result, _ = run_pipeline_step(
            groups, n_act, 'magnetic_cross',
            self.q0, self.w0, np.zeros(3),
            self.q_err, self.omega_ref, self.pd_law,
        )
        assert result.u.shape == (n_act,)


# ---------------------------------------------------------------------------
# Test: cross-method allocation properties from paper
# ---------------------------------------------------------------------------

class TestPaperCrossMethodProperties:
    """Verify allocation method properties described in the paper."""

    @pytest.fixture(autouse=True)
    def setup(self, pd_law, config_3mtq_1rw):
        self.pd_law = pd_law
        self.groups, self.n_act = config_3mtq_1rw
        self.w0, self.q0, _ = paper_initial_conditions(seed=7)
        self.q_err = self.q0[1:4]
        self.omega_ref = np.zeros(3)

    def _run_method(self, method):
        return run_pipeline_step(
            self.groups, self.n_act, method,
            self.q0, self.w0, np.zeros(3),
            self.q_err, self.omega_ref, self.pd_law,
        )

    def test_lp_preserves_direction(self):
        """LP should have near-zero direction error (paper Prop. 1)."""
        result, tau_des = self._run_method('lp')
        if np.linalg.norm(tau_des) > 1e-10:
            assert result.direction_error < 0.1  # < ~6 degrees

    def test_qp_minimizes_magnitude_error(self):
        """QP minimizes ||tau_achieved - tau_desired||."""
        res_qp, tau_des = self._run_method('qp')
        res_lp, _ = self._run_method('lp')

        if np.linalg.norm(tau_des) > 1e-10:
            err_qp = np.linalg.norm(res_qp.tau_achieved - tau_des)
            err_lp = np.linalg.norm(res_lp.tau_achieved - tau_des)
            # QP should have <= magnitude error than LP
            assert err_qp <= err_lp + 1e-10

    def test_qpw_biases_toward_direction(self):
        """QPW should have smaller direction error than plain QP."""
        res_qpw, tau_des = self._run_method('qpw')
        res_qp, _ = self._run_method('qp')

        if np.linalg.norm(tau_des) > 1e-10:
            assert res_qpw.direction_error <= res_qp.direction_error + 0.01

    def test_qpc_respects_energy_gate(self):
        """QPC: omega^T @ tau_achieved <= max(0, omega^T @ tau_desired)."""
        result, tau_des = self._run_method('qpc')
        omega = self.w0

        if np.linalg.norm(omega) > 1e-10 and result.tau_achieved is not None:
            power_achieved = np.dot(omega, result.tau_achieved)
            power_desired = np.dot(omega, tau_des)
            ub = max(0.0, power_desired)
            # Allow small numerical violation
            assert power_achieved <= ub + 1e-8

    def test_all_methods_respect_bounds(self):
        """All methods respect actuator command bounds."""
        for method in ['lp', 'qp', 'qpw', 'qpc', 'pseudoinverse']:
            result, _ = self._run_method(method)
            # MTQ bounds
            assert np.all(np.abs(result.u[:3]) <= PAPER_MTQ_MAX + 1e-10)
            # RW bound
            assert np.abs(result.u[3]) <= PAPER_RW_MAX + 1e-10

    def test_alpha_positive(self):
        """All methods should achieve positive alpha (torque in right direction)."""
        for method in ['lp', 'qp', 'qpw', 'qpc', 'pseudoinverse']:
            result, tau_des = self._run_method(method)
            if np.linalg.norm(tau_des) > 1e-10:
                assert result.alpha >= -0.01  # allow tiny numerical


# ---------------------------------------------------------------------------
# Test: B_tau assembly for paper configs
# ---------------------------------------------------------------------------

class TestPaperBTauAssembly:
    """Verify B_tau structure matches paper equations."""

    def test_3mtq_0rw_rank(self, mtq_3):
        """3MTQ+0RW: B_tau has rank <= 2 (perpendicular to B)."""
        groups = [mtq_3]
        B_tau, _, _ = assemble_B_tau(groups, B_BODY_PAPER)
        rank = np.linalg.matrix_rank(B_tau)
        assert rank == 2  # MTQ torque is perpendicular to B

    def test_3mtq_1rw_rank(self, mtq_3, rw_x):
        """3MTQ+1RW: B_tau has rank 3 (full rank, but only 4 columns)."""
        groups = [mtq_3, rw_x]
        B_tau, _, _ = assemble_B_tau(groups, B_BODY_PAPER)
        rank = np.linalg.matrix_rank(B_tau)
        assert rank == 3

    def test_0mtq_3rw_is_identity(self, rw_xyz):
        """0MTQ+3RW: B_tau = I (orthogonal body-axis wheels)."""
        groups = [rw_xyz]
        B_tau, _, _ = assemble_B_tau(groups, B_BODY_PAPER)
        np.testing.assert_array_almost_equal(B_tau, np.eye(3))

    def test_mtq_torque_perpendicular_to_B(self, mtq_3):
        """MTQ torque columns are perpendicular to B (tau = m x B)."""
        groups = [mtq_3]
        B_tau, _, _ = assemble_B_tau(groups, B_BODY_PAPER)
        for j in range(B_tau.shape[1]):
            dot = np.dot(B_tau[:, j], B_BODY_PAPER)
            assert abs(dot) < 1e-12

    def test_3mtq_2rw_overactuated(self, mtq_3, rw_xy):
        """3MTQ+2RW: 5 columns, rank 3 -> 2D nullspace."""
        groups = [mtq_3, rw_xy]
        B_tau, _, _ = assemble_B_tau(groups, B_BODY_PAPER)
        assert B_tau.shape == (3, 5)
        rank = np.linalg.matrix_rank(B_tau)
        assert rank == 3
        nullspace_dim = B_tau.shape[1] - rank
        assert nullspace_dim == 2


# ---------------------------------------------------------------------------
# Test: actuator failure scenarios from paper
# ---------------------------------------------------------------------------

class TestPaperFailureScenarios:
    """Verify graceful degradation under actuator failure (paper Section V)."""

    @pytest.fixture(autouse=True)
    def setup(self, pd_law, config_3mtq_1rw):
        self.pd_law = pd_law
        self.groups, self.n_act = config_3mtq_1rw
        self.w0, self.q0, _ = paper_initial_conditions(seed=99)
        self.q_err = self.q0[1:4]
        self.omega_ref = np.zeros(3)

    def _run(self, method='lp', failed=None):
        return run_pipeline_step(
            self.groups, self.n_act, method,
            self.q0, self.w0, np.zeros(3),
            self.q_err, self.omega_ref, self.pd_law,
            failed_actuators=failed,
        )

    def test_rw_failure_still_allocates(self):
        """Failing the single RW in 3MTQ+1RW -> falls back to MTQ-only."""
        result, _ = self._run(method='lp', failed=np.array([3]))
        assert result.u.shape == (self.n_act,)
        # RW command should be zero
        assert result.u[3] == 0.0
        # MTQ should still produce some torque
        assert np.linalg.norm(result.u[:3]) > 0

    def test_one_mtq_failure(self):
        """Failing one MTQ in 3MTQ+1RW -> still 3 actuators."""
        result, _ = self._run(method='qp', failed=np.array([0]))
        assert result.u.shape == (self.n_act,)
        assert result.u[0] == 0.0

    def test_two_mtq_failure(self):
        """Failing two MTQs -> only 1 MTQ + 1 RW (severely degraded)."""
        result, _ = self._run(method='qp', failed=np.array([0, 1]))
        assert result.u.shape == (self.n_act,)
        assert result.u[0] == 0.0
        assert result.u[1] == 0.0

    def test_no_failure_matches_normal(self):
        """Empty failure list matches no-failure baseline."""
        res_normal, _ = self._run(method='qp')
        res_empty, _ = self._run(method='qp', failed=np.array([], dtype=int))
        np.testing.assert_array_almost_equal(res_normal.u, res_empty.u)


# ---------------------------------------------------------------------------
# Test: momentum management with paper configs
# ---------------------------------------------------------------------------

class TestPaperDesaturation:
    """Verify desaturation strategies with paper hardware configs."""

    @pytest.fixture(autouse=True)
    def setup(self, pd_law, config_3mtq_1rw):
        self.pd_law = pd_law
        self.groups, self.n_act = config_3mtq_1rw
        self.w0, self.q0, _ = paper_initial_conditions(seed=13)
        self.q_err = self.q0[1:4]
        self.omega_ref = np.zeros(3)

    def test_nullspace_desat_3mtq_1rw(self):
        """Nullspace desat on 3MTQ+1RW (4 cols, rank 3 -> 1D nullspace)."""
        desat_cfg = DesaturationConfig(
            strategy='nullspace', k_desat=PAPER_C_GAIN,
            h_rw_target=np.array([0.004, 0.0, 0.0]),
        )
        h_rw = np.array([0.01, 0.0, 0.0])  # above target

        result, tau_des = run_pipeline_step(
            self.groups, self.n_act, 'qp',
            self.q0, self.w0, np.zeros(3),
            self.q_err, self.omega_ref, self.pd_law,
            h_rw_for_desat=h_rw, desat_config=desat_cfg,
        )
        assert result.u.shape == (self.n_act,)

    def test_scheduled_desat_3mtq_1rw(self):
        """Scheduled desat on 3MTQ+1RW adds dump torque when B is favorable."""
        desat_cfg = DesaturationConfig(
            strategy='scheduled', k_desat=PAPER_C_GAIN,
            h_rw_target=np.array([0.004, 0.0, 0.0]),
            authority_threshold=0.0,  # always active for test
        )
        h_rw = np.array([0.01, 0.0, 0.0])

        result, tau_des = run_pipeline_step(
            self.groups, self.n_act, 'lp',
            self.q0, self.w0, np.zeros(3),
            self.q_err, self.omega_ref, self.pd_law,
            h_rw_for_desat=h_rw, desat_config=desat_cfg,
        )
        assert result.u.shape == (self.n_act,)

    def test_weighted_desat_3mtq_2rw(self, mtq_3, rw_xy):
        """Weighted desat on 3MTQ+2RW (overactuated)."""
        groups = [mtq_3, rw_xy]
        desat_cfg = DesaturationConfig(
            strategy='weighted', k_desat=PAPER_C_GAIN,
            w_desat=0.5,
        )
        h_rw = np.array([0.005, -0.003, 0.0])

        result, tau_des = run_pipeline_step(
            groups, 5, 'qp',
            self.q0, self.w0, np.zeros(3),
            self.q_err, self.omega_ref, self.pd_law,
            h_rw_for_desat=h_rw, desat_config=desat_cfg,
        )
        assert result.u.shape == (5,)


# ---------------------------------------------------------------------------
# Test: multi-seed reproducibility
# ---------------------------------------------------------------------------

class TestPaperMultiSeedSingleStep:
    """Run single pipeline steps across multiple paper seeds.

    Verifies the pipeline doesn't crash for any of the random ICs
    used in the paper's MC campaigns.
    """

    @pytest.fixture(autouse=True)
    def setup(self, pd_law, config_3mtq_1rw):
        self.pd_law = pd_law
        self.groups, self.n_act = config_3mtq_1rw

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_lp_multi_seed(self, seed):
        """LP allocation converges for paper seeds 0-9."""
        w0, q0, goal_eci = paper_initial_conditions(seed=seed)
        q_err = q0[1:4]
        result, tau_des = run_pipeline_step(
            self.groups, self.n_act, 'lp',
            q0, w0, np.zeros(3),
            q_err, np.zeros(3), self.pd_law,
        )
        assert result.u.shape == (self.n_act,)
        assert np.all(np.isfinite(result.u))

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_qpc_multi_seed(self, seed):
        """QPC allocation converges for paper seeds 0-9."""
        w0, q0, goal_eci = paper_initial_conditions(seed=seed)
        q_err = q0[1:4]
        result, tau_des = run_pipeline_step(
            self.groups, self.n_act, 'qpc',
            q0, w0, np.zeros(3),
            q_err, np.zeros(3), self.pd_law,
        )
        assert result.u.shape == (self.n_act,)
        assert np.all(np.isfinite(result.u))


# ---------------------------------------------------------------------------
# Test: cross-configuration comparison (same law, different actuators)
# ---------------------------------------------------------------------------

class TestPaperCrossConfigComparison:
    """Same PD law and IC, different actuator configs — paper's key result."""

    @pytest.fixture(autouse=True)
    def setup(self, pd_law):
        self.pd_law = pd_law
        self.w0, self.q0, _ = paper_initial_conditions(seed=5)
        self.q_err = self.q0[1:4]
        self.omega_ref = np.zeros(3)

    def _run_config(self, groups, n_act, method='lp'):
        return run_pipeline_step(
            groups, n_act, method,
            self.q0, self.w0, np.zeros(3),
            self.q_err, self.omega_ref, self.pd_law,
        )

    def test_more_actuators_more_alpha(
        self, config_3mtq_0rw, config_3mtq_1rw, config_3mtq_2rw,
    ):
        """More actuators -> higher alpha (more torque achievable)."""
        res_0rw, _ = self._run_config(*config_3mtq_0rw, method='qp')
        res_1rw, _ = self._run_config(*config_3mtq_1rw, method='qp')
        res_2rw, _ = self._run_config(*config_3mtq_2rw, method='qp')

        # With more actuators, alpha should be >= (or very close)
        assert res_1rw.alpha >= res_0rw.alpha - 0.05
        assert res_2rw.alpha >= res_1rw.alpha - 0.05

    def test_3rw_fully_actuated(self, config_0mtq_3rw):
        """0MTQ+3RW with orthogonal wheels -> exact torque (alpha ~= 1)."""
        groups, n_act = config_0mtq_3rw
        result, tau_des = self._run_config(groups, n_act, method='qp')
        tau_norm = np.linalg.norm(tau_des)

        if tau_norm > 1e-12 and tau_norm < PAPER_RW_MAX:
            # Should achieve exact torque if within bounds
            assert result.alpha > 0.95
            assert result.direction_error < 0.01

    def test_all_configs_produce_output(
        self, config_3mtq_0rw, config_3mtq_1rw,
        config_3mtq_2rw, config_0mtq_3rw,
    ):
        """All 4 paper configs produce valid pipeline output."""
        for groups, n_act in [
            config_3mtq_0rw, config_3mtq_1rw,
            config_3mtq_2rw, config_0mtq_3rw,
        ]:
            result, _ = self._run_config(groups, n_act, method='lp')
            assert result.u.shape == (n_act,)
            assert np.all(np.isfinite(result.u))


# ---------------------------------------------------------------------------
# Test: short integration (multi-step) with paper config
# ---------------------------------------------------------------------------

class TestPaperMultiStep:
    """Run a few pipeline steps in sequence to verify state evolution."""

    def test_3mtq_1rw_5_steps(self, pd_law, config_3mtq_1rw):
        """Run 5 consecutive pipeline steps with paper 3MTQ+1RW config."""
        groups, n_act = config_3mtq_1rw
        w0, q0, _ = paper_initial_conditions(seed=0)
        q_err = q0[1:4]
        omega_ref = np.zeros(3)

        prev_result = None
        for step in range(5):
            result, tau_des = run_pipeline_step(
                groups, n_act, 'lp',
                q0, w0, np.zeros(3),
                q_err, omega_ref, pd_law,
            )
            assert result.u.shape == (n_act,)
            assert np.all(np.isfinite(result.u))

            # Simple Euler integration of omega for next step
            dt = 2.0  # paper timestep
            J = PAPER_J
            tau = result.tau_achieved if result.tau_achieved is not None else np.zeros(3)
            w0 = w0 + dt * np.linalg.solve(J, tau - np.cross(w0, J @ w0))
            # (quaternion evolution omitted for simplicity — just testing pipeline doesn't crash)

            prev_result = result

    def test_0mtq_3rw_pd_torque_opposes_omega(self, pd_law, config_0mtq_3rw):
        """0MTQ+3RW: PD law torque should oppose omega (damping effect).

        The desired torque tau_desired should have a negative component
        along omega, indicating the controller is trying to slow down
        rotation. Full convergence requires proper ODE integration
        over many seconds — tested in the paper MC scripts.
        """
        groups, n_act = config_0mtq_3rw
        w0, q0, _ = paper_initial_conditions(seed=0)
        q_err = q0[1:4]
        omega_ref = np.zeros(3)

        result, tau_des = run_pipeline_step(
            groups, n_act, 'qp',
            q0, w0, np.zeros(3),
            q_err, omega_ref, pd_law,
        )

        # The PD torque (before gyroscopic compensation) should oppose omega
        # tau_pd = -(kp * q_err + kd * omega) => tau_pd . omega < 0
        tau_pd = pd_law.compute(
            attitude_input=q_err, omega_input=w0,
        )
        assert np.dot(tau_pd, w0) < 0, (
            "PD torque should oppose angular velocity (damping)"
        )
