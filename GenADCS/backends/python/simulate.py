"""Simulation entry point for the coupled Numba backend."""

from .helpers.input_output.resultsformat import ResultsFormat
from GenADCS.backends.shared.simulationresult import SimulationResult
from GenADCS.backends.shared.simulationresult.record import record_samples
from .solver import CoupledSolver, Solver
from .state import State


def simulate(satellite, environment, dynamics, state, solver, backend="Python", resultsformat=None):
    """Integrate the coupled orbit and attitude from the initial state to tf.

    ``solver.tf`` is elapsed simulation time in seconds. The Python backend
    prepares numeric model data once; the selected integrator runs in Numba.
    """
    if backend != "Python":
        raise NotImplementedError(f"backend {backend!r} is not implemented for GenADCS")
    if isinstance(solver, Solver):
        solver = solver.coupled
    if not isinstance(solver, CoupledSolver):
        raise TypeError("solver must be a CoupledSolver or a Solver containing one")
    if not isinstance(state, State):
        raise TypeError("state must be an GenADCS.backends.python.state.State")

    if resultsformat is None:
        resultsformat = ResultsFormat()
    if not isinstance(resultsformat, ResultsFormat):
        raise TypeError("resultsformat must be a Python ResultsFormat")
    times_s, state_vectors = solver.propagator.integrate(
        state, satellite, environment, dynamics, solver.tf,
        progress=solver.progress,
        progress_chunk_steps=solver.progress_chunk_steps,
        record_every=resultsformat.N,
    )
    # The integrator includes a terminal row even when its time is not on the
    # requested recording grid; keep only requested samples in the result.
    on_grid = len(times_s) == 1 or abs(times_s[-1] / (solver.propagator.dt * resultsformat.N) - round(times_s[-1] / (solver.propagator.dt * resultsformat.N))) < 1e-9
    samples = slice(None) if on_grid else slice(None, -1)
    vectors = state_vectors[samples]
    return record_samples(times_s[samples], vectors, resultsformat, satellite,
                          environment, dynamics, state.orbit.julian_date, backend)
