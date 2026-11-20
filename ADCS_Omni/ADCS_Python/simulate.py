"""Simulation entry point for the coupled Numba backend."""

from dataclasses import dataclass

import numpy as np

from .solver import CoupledSolver, Solver
from .state import State
from .propagation.coupled_propagators._state_layout import unpack_state


@dataclass(slots=True)
class SimulationResult:
    """Sampled coupled trajectory with elapsed times in seconds.

    ``state_vectors`` columns are quaternion (0:4), body rates (4:7),
    inertial position (7:10), inertial velocity (10:13), then wheel momenta.
    """

    times_s: np.ndarray
    state_vectors: np.ndarray
    final_state: State


def simulate(satellite, environment, dynamics, state, solver, backend="Python"):
    """Integrate the coupled orbit and attitude from the initial state to tf.

    ``solver.tf`` is elapsed simulation time in seconds. The Python backend
    prepares numeric model data once; the selected integrator runs in Numba.
    """
    if backend != "Python":
        raise NotImplementedError(f"backend {backend!r} is not implemented for ADCS_Omni")
    if isinstance(solver, Solver):
        solver = solver.coupled
    if not isinstance(solver, CoupledSolver):
        raise TypeError("solver must be a CoupledSolver or a Solver containing one")
    if not isinstance(state, State):
        raise TypeError("state must be an ADCS_Omni.ADCS_Python.state.State")

    times_s, state_vectors = solver.propagator.integrate(
        state, satellite, environment, dynamics, solver.tf,
        progress=solver.progress,
        progress_chunk_steps=solver.progress_chunk_steps,
    )
    final_state = unpack_state(state_vectors[-1], state, solver.tf)
    return SimulationResult(times_s, state_vectors, final_state)
