"""Integration tests for the Python-facing C++ simulation adapter."""

import sys
import importlib
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from GenADCS.backends.shared.simulationresult.resultsformat import ResultsFormat

cpp_simulate = importlib.import_module("GenADCS.backends.cpp.simulate")


def test_simulate_passes_native_integrator_samples_to_result_recorder():
    class State:
        orbit = SimpleNamespace(julian_date=2451545.0)

    class CoupledSolver:
        def __init__(self, propagator):
            self.propagator = propagator
            self.tf = 2.5
            self.progress = False
            self.progress_chunk_steps = 500

    class Solver:
        def __init__(self, coupled):
            self.coupled = coupled

    class Propagator:
        dt = 1.0

        def integrate(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            return np.array([0.0, 2.0, 2.5]), np.arange(39.0).reshape(3, 13)

    cpp_state_module = ModuleType("GenADCS.backends.cpp.state")
    cpp_state_module.State = type("CppState", (), {})
    solver_module = ModuleType("GenADCS.backends.cpp.solver")
    solver_module.CoupledSolver = CoupledSolver
    solver_module.Solver = Solver
    python_state_module = ModuleType("GenADCS.backends.python.state")
    python_state_module.State = State
    propagator = Propagator()
    expected = object()

    with patch.dict(sys.modules, {
        "GenADCS.backends.cpp.state": cpp_state_module,
        "GenADCS.backends.cpp.solver": solver_module,
        "GenADCS.backends.python.state": python_state_module,
    }), patch.object(cpp_simulate, "record_samples", return_value=expected) as record:
        state = State()
        satellite, environment, dynamics = object(), object(), object()
        result = cpp_simulate.simulate(
            satellite, environment, dynamics, state,
            Solver(CoupledSolver(propagator)), resultsformat=ResultsFormat(N=2),
        )

    assert result is expected
    assert propagator.args == (state, satellite, environment, dynamics, 2.5)
    assert propagator.kwargs == {
        "progress": False, "progress_chunk_steps": 500, "record_every": 2,
    }
    args = record.call_args.args
    np.testing.assert_array_equal(args[0], [0.0, 2.0])
    np.testing.assert_array_equal(args[1], np.arange(26.0).reshape(2, 13))
    assert args[2] == ResultsFormat(N=2)
    assert args[6:] == (2451545.0, "C")
