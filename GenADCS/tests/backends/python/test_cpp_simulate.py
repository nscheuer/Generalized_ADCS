"""Integration tests for the Python-facing C++ simulation adapter."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from GenADCS.backends.shared.simulationresult.resultsformat import ResultsFormat

cpp_simulate = importlib.import_module("GenADCS.backends.cpp.simulate")
cpp_coupled_rk4 = importlib.import_module(
    "GenADCS.backends.cpp.propagation.coupled_propagators.coupled_rk4_propagator"
)
cpp_coupled_dynamics = importlib.import_module(
    "GenADCS.backends.cpp.propagation.coupled_propagators._coupled_dynamics"
)
cpp_state_layout = importlib.import_module(
    "GenADCS.backends.cpp.propagation.coupled_propagators._state_layout"
)
cpp_separate = importlib.import_module("GenADCS.backends.cpp.propagation._separate")
cpp_solver = importlib.import_module("GenADCS.backends.cpp.solver")
python_state = importlib.import_module("GenADCS.backends.python.state")


def test_simulate_passes_native_integrator_samples_to_result_recorder() -> None:
    state = python_state.State(
        python_state.AttitudeState([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        python_state.OrbitalState.from_cartesian(
            [7e6, 0.0, 0.0], [0.0, 7500.0, 0.0], 2451545.0
        ),
    )
    propagator = cpp_coupled_rk4.CoupledRK4Propagator(dt=1.0)
    solver = cpp_solver.Solver(
        coupled=cpp_solver.CoupledSolver(
            propagator, tf=2.5, progress=False, progress_chunk_steps=500
        )
    )
    satellite = SimpleNamespace(adcs=None)
    environment = SimpleNamespace(ephemeris=None)
    dynamics = object()
    runtime = SimpleNamespace(wheel_commands=np.zeros(0))
    expected = object()

    with (
        patch.object(cpp_coupled_dynamics, "_prepare", return_value=runtime),
        patch.object(
            cpp_state_layout, "pack_state", return_value=np.zeros(13)
        ) as pack_mock,
        patch.object(
            cpp_separate,
            "simulate_coupled",
            return_value=(np.array([0.0, 2.0, 2.5]), np.arange(39.0).reshape(3, 13), None),
        ) as sim_mock,
        patch.object(cpp_simulate, "record_samples", return_value=expected) as record,
    ):
        result = cpp_simulate.simulate(
            satellite,
            environment,
            dynamics,
            state,
            solver,
            resultsformat=ResultsFormat(N=2),
        )

    assert result is expected
    pack_mock.assert_called_once_with(state, 0)
    assert sim_mock.call_args.args[1:5] == (2451545.0, 1.0, 2.5, 2)
    args: tuple[Any, ...] = record.call_args.args
    np.testing.assert_array_equal(args[0], [0.0, 2.0])
    np.testing.assert_array_equal(args[1], np.arange(26.0).reshape(2, 13))
    assert args[2] == ResultsFormat(N=2)
    assert args[6:] == (2451545.0, "C")
