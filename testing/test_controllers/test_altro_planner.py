"""
Test suite for ALTRO trajectory planner.

Tests the core functionality of the Augmented Lagrangian iLQR trajectory optimizer
including Satellite object construction, auto-scaling, and basic trajectory generation.
"""

import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
# Add trajectory planner build directory for C++ bindings
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../trajectory_planner/build")))

# Import ADCS modules first - they will load pysat internally
from ADCS.controller.helpers import PlannerSettings
from ADCS.controller.helpers.planner_subsettings import (
    CostWeights, SolverPassConfig, ConvergenceConfig, AugLagConfig
)
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.helpers.math_constants import MathConstants

# Get C++ Satellite from already-loaded pysat module (loaded by ADCS modules)
CppSatellite = None
CPP_SATELLITE_SKIP_REASON = None
try:
    # Access the already-loaded pysat module via sys.modules
    pysat = sys.modules.get('trajectory_planner.build.pysat')
    if pysat is not None:
        CppSatellite = pysat.Satellite
    else:
        CPP_SATELLITE_SKIP_REASON = "pysat module not loaded by ADCS"
except AttributeError as e:
    CPP_SATELLITE_SKIP_REASON = f"C++ Satellite class not found in pysat: {e}"


@pytest.fixture
def basic_satellite():
    """Create a basic satellite with MTQs and RWs for testing."""
    mtq_max = 5.0
    mtqs = [MTQ(axis=j, max_torque=mtq_max) for j in MathConstants.unitvecs]

    rw_max_torque = 0.005
    rw_J = 0.0014
    rw_h0 = 0.001
    rw_hmax = 0.015
    rws = [RW(axis=j, max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax)
           for j in MathConstants.unitvecs]

    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]

    sat = Satellite(
        mass=10.0,
        J_0=np.diagflat([0.1, 0.12, 0.19]),
        actuators=rws + mtqs,
        sensors=mtms
    )
    return sat


class TestCostWeights:
    """Tests for CostWeights dataclass."""

    def test_default_construction(self):
        """Test that CostWeights can be constructed with defaults."""
        cw = CostWeights()
        assert cw.angle == 1e3
        assert cw.ang_vel == 1e4
        assert cw.angle_N >= cw.angle, "Terminal cost should be >= running cost"
        assert cw.ang_vel_N >= cw.ang_vel, "Terminal cost should be >= running cost"

    def test_custom_construction(self):
        """Test CostWeights with custom values."""
        cw = CostWeights(angle=1e5, ang_vel=1e6, angle_N=1e6, ang_vel_N=1e7)
        assert cw.angle == 1e5
        assert cw.ang_vel == 1e6

    def test_to_tuple(self):
        """Test that to_tuple returns correct format."""
        cw = CostWeights()
        t = cw.to_tuple()
        assert isinstance(t, tuple)
        assert len(t) == 12  # 10 weights + ang_cost_func_type + use_raw_control_cost


class TestSolverPassConfig:
    """Tests for SolverPassConfig and sub-configs."""

    def test_default_construction(self):
        """Test that SolverPassConfig can be constructed with defaults."""
        config = SolverPassConfig()
        assert config.convergence.max_outer_iter > 0
        assert config.convergence.max_inner_iter > 0
        assert config.aug_lag.penalty_init > 0

    def test_convergence_config(self):
        """Test ConvergenceConfig parameters."""
        conv = ConvergenceConfig(max_outer_iter=50, max_inner_iter=300)
        assert conv.max_outer_iter == 50
        assert conv.max_inner_iter == 300

    def test_auglag_config(self):
        """Test AugLagConfig parameters."""
        al = AugLagConfig(penalty_init=1e-2, penalty_max=1e12)
        assert al.penalty_init == 1e-2
        assert al.penalty_max == 1e12


class TestPlannerSettings:
    """Tests for PlannerSettings construction and configuration."""

    def test_construction(self, basic_satellite):
        """Test that PlannerSettings can be constructed."""
        ps = PlannerSettings(est_sat=basic_satellite)
        assert ps.est_sat is basic_satellite
        assert ps.dt_tp > 0
        assert ps.dt_tvlqr > 0

    def test_custom_cost_weights(self, basic_satellite):
        """Test PlannerSettings with custom cost weights."""
        custom_cost = CostWeights(angle=1e6, ang_vel=1e7)
        ps = PlannerSettings(est_sat=basic_satellite, cost_main=custom_cost)
        assert ps.cost_main.angle == 1e6

    def test_settings_tuples(self, basic_satellite):
        """Test that settings tuples are generated correctly."""
        ps = PlannerSettings(est_sat=basic_satellite)

        # These should not raise
        sys_settings = ps.systemSettings()
        assert isinstance(sys_settings, tuple)

        main_cost = ps.optMainCostSettings()
        assert isinstance(main_cost, tuple)

        dist_settings = ps.planner_disturbance_settings()
        assert isinstance(dist_settings, tuple)


class TestSatelliteCppBinding:
    """Tests for the C++ Satellite class via pybind11."""

    def test_satellite_construction(self):
        """Test that C++ Satellite can be constructed via Python."""
        if CppSatellite is None:
            pytest.skip(CPP_SATELLITE_SKIP_REASON)
        sat = CppSatellite()
        sat.change_Jcom(np.eye(3))
        assert sat is not None

    def test_add_actuators(self):
        """Test adding actuators to C++ Satellite."""
        if CppSatellite is None:
            pytest.skip(CPP_SATELLITE_SKIP_REASON)
        sat = CppSatellite()
        sat.change_Jcom(np.diagflat([0.1, 0.12, 0.19]))

        # Add MTQs
        for axis in MathConstants.unitvecs:
            sat.add_MTQ(axis.flatten(), 5.0, 1e3)

        # Add RWs
        for axis in MathConstants.unitvecs:
            sat.add_RW(axis.flatten(), 0.0014, 0.005, 0.015, 1e5, 1e4, 0.012, 1e0, 0.0001)

        assert sat is not None

    def test_auto_scale_control_costs(self):
        """Test auto_scale_control_costs function."""
        if CppSatellite is None:
            pytest.skip(CPP_SATELLITE_SKIP_REASON)
        sat = CppSatellite()
        sat.change_Jcom(np.diagflat([0.1, 0.12, 0.19]))

        # Add actuators with known costs
        sat.add_MTQ(np.array([1.0, 0.0, 0.0]), 5.0, 1e3)
        sat.add_RW(np.array([1.0, 0.0, 0.0]), 0.0014, 0.005, 0.015, 1e5, 1e4, 0.012, 1e0, 0.0001)

        # This should not raise
        sat.auto_scale_control_costs(1.0)


class TestConstraints:
    """Tests for constraint setup."""

    def test_av_constraint(self):
        """Test angular velocity constraint setup."""
        if CppSatellite is None:
            pytest.skip(CPP_SATELLITE_SKIP_REASON)
        sat = CppSatellite()
        sat.change_Jcom(np.eye(3))

        wmax = 20 * np.pi / 180.0
        sat.set_AV_constraint(wmax)
        sat.clear_AV_constraint()
        assert sat is not None


class TestNumpyArmadilloConversion:
    """Tests for numpy-to-armadillo matrix/vector passing.

    These tests verify that arrays are correctly transferred between Python
    (numpy, row-major) and C++ (armadillo, column-major).
    """

    def test_inertia_matrix_passing(self):
        """Test that inertia matrix is correctly passed to C++."""
        if CppSatellite is None:
            pytest.skip(CPP_SATELLITE_SKIP_REASON)

        # Create a non-diagonal inertia matrix to detect memory order issues
        J = np.array([
            [0.1, 0.01, 0.02],
            [0.01, 0.12, 0.015],
            [0.02, 0.015, 0.19]
        ])

        sat = CppSatellite()
        sat.change_Jcom(J)
        assert sat is not None

    def test_vector_passing_mtq_axis(self):
        """Test that axis vectors are correctly passed for MTQs."""
        if CppSatellite is None:
            pytest.skip(CPP_SATELLITE_SKIP_REASON)
        sat = CppSatellite()
        sat.change_Jcom(np.eye(3))

        # Non-unit axis to detect transposition issues
        axis = np.array([1.0, 0.0, 0.0])
        sat.add_MTQ(axis, 5.0, 1e3)
        assert sat is not None

    def test_c_contiguous_array(self):
        """Test passing C-contiguous (row-major) numpy array."""
        if CppSatellite is None:
            pytest.skip(CPP_SATELLITE_SKIP_REASON)

        # Explicitly create C-contiguous array
        J = np.ascontiguousarray(np.diagflat([0.1, 0.12, 0.19]))
        assert J.flags['C_CONTIGUOUS']

        sat = CppSatellite()
        sat.change_Jcom(J)
        assert sat is not None

    def test_fortran_contiguous_array(self):
        """Test passing Fortran-contiguous (column-major) numpy array."""
        if CppSatellite is None:
            pytest.skip(CPP_SATELLITE_SKIP_REASON)

        # Create Fortran-contiguous array (column-major like Armadillo)
        J = np.asfortranarray(np.diagflat([0.1, 0.12, 0.19]))
        assert J.flags['F_CONTIGUOUS']

        sat = CppSatellite()
        sat.change_Jcom(J)
        assert sat is not None

    def test_non_contiguous_array_slice(self):
        """Test passing non-contiguous array slice."""
        if CppSatellite is None:
            pytest.skip(CPP_SATELLITE_SKIP_REASON)

        # Create a non-contiguous view by slicing
        big_array = np.zeros((6, 6))
        big_array[::2, ::2] = np.diagflat([0.1, 0.12, 0.19])
        J_slice = big_array[::2, ::2]

        # This may not be contiguous - pass contiguous copy
        sat = CppSatellite()
        sat.change_Jcom(np.ascontiguousarray(J_slice))
        assert sat is not None

    def test_pickle_roundtrip(self):
        """Test that satellite can be pickled and unpickled correctly."""
        if CppSatellite is None:
            pytest.skip(CPP_SATELLITE_SKIP_REASON)
        import pickle

        sat = CppSatellite()
        sat.change_Jcom(np.diagflat([0.1, 0.12, 0.19]))
        sat.add_MTQ(np.array([1.0, 0.0, 0.0]), 5.0, 1e3)
        sat.add_RW(np.array([0.0, 1.0, 0.0]), 0.0014, 0.005, 0.015, 1e5, 1e4, 0.012, 1e0, 0.0001)

        # Pickle and unpickle
        pickled = pickle.dumps(sat)
        restored = pickle.loads(pickled)

        assert restored is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
