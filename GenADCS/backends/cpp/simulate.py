"""Python-facing orchestration for the compiled C++ propagators."""

import numpy as np

from GenADCS.backends.shared.simulationresult import SimulationResult, SatelliteResult
from GenADCS.backends.shared.simulationresult.record import record_samples
from GenADCS.backends.shared.simulationresult.resultsformat import ResultsFormat


def simulate(satellite, environment, dynamics, state, solver, backend="C", resultsformat=None):
    """Integrate with the C++ propagator and assemble the sampled result in Python."""
    if backend != "C":
        raise NotImplementedError(f"backend {backend!r} is not implemented for GenADCS C++")

    from .solver import CoupledSolver, Solver
    from .state import State as CppState
    from GenADCS.backends.python.state import State as PythonState

    if isinstance(solver, Solver):
        solver = solver.coupled
    if not isinstance(solver, CoupledSolver):
        raise TypeError("solver must be a CoupledSolver or a Solver containing one")
    if not isinstance(state, (CppState, PythonState)):
        raise TypeError("state must be a GenADCS State")
    if resultsformat is None:
        resultsformat = ResultsFormat()
    if not isinstance(resultsformat, ResultsFormat):
        raise TypeError("resultsformat must be a ResultsFormat")
    if isinstance(resultsformat.N, bool) or not isinstance(resultsformat.N, int) or resultsformat.N < 1:
        raise ValueError("ResultsFormat.N must be a positive integer")

    propagator = solver.propagator
    times, vectors = propagator.integrate(
        state, satellite, environment, dynamics, solver.tf,
        progress=solver.progress,
        progress_chunk_steps=solver.progress_chunk_steps,
        record_every=resultsformat.N,
    )
    times = np.asarray(times, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    if times.ndim != 1 or times.size < 1:
        raise ValueError("propagator.integrate must return at least one sample")
    if vectors.ndim != 2 or vectors.shape[0] != times.size or vectors.shape[1] < 13:
        raise ValueError("state vectors must have matching rows and at least 13 columns")

    interval = propagator.dt * resultsformat.N
    # Integrators may include a final partial interval. Results are sampled only
    # on the requested output grid, so omit that trailing row if necessary.
    on_grid = abs(times[-1] / interval - round(times[-1] / interval)) < 1e-9
    stop = len(times) if on_grid else len(times) - 1
    initial_jd = state.orbit.julian_date
    return record_samples(times[:stop], vectors[:stop], resultsformat,
                          satellite, environment, dynamics, initial_jd, backend)


__all__ = ["ResultsFormat", "SatelliteResult", "SimulationResult", "simulate"]
