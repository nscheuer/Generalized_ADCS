"""Python-facing orchestration for the compiled C++ propagators."""

import numpy as np
import math

from GenADCS.backends.shared.simulationresult import SimulationResult, SatelliteResult
from GenADCS.backends.shared.simulationresult.record import record_samples
from GenADCS.backends.shared.simulationresult.simulationresult import SENSOR_FIELDS
from GenADCS.backends.shared.simulationresult.resultsformat import ResultsFormat


def simulate(satellite, environment, dynamics, state, solver, backend="C", resultsformat=None, seed=0):
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

    sensors = satellite.adcs.sensors
    sensor_every = satellite.adcs.N
    if sensor_every < 1:
        raise ValueError("ADCSSubsystem.N must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if any(type(s).__name__ in ("CoarseSunSensor", "FineSunSensor", "AutonomousStarTracker") for s in sensors) and environment.ephemeris is None:
        raise ValueError("Sun and star tracker sensors require an ephemeris")
    propagator = solver.propagator
    times, vectors = propagator.integrate(
        state, satellite, environment, dynamics, solver.tf,
        progress=solver.progress,
        progress_chunk_steps=solver.progress_chunk_steps,
        record_every=math.gcd(resultsformat.N, sensor_every) if sensors else resultsformat.N,
    )
    times = np.asarray(times, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    if times.ndim != 1 or times.size < 1:
        raise ValueError("propagator.integrate must return at least one sample")
    if vectors.ndim != 2 or vectors.shape[0] != times.size or vectors.shape[1] < 13:
        raise ValueError("state vectors must have matching rows and at least 13 columns")

    sensor_rows = None
    if sensors or any(getattr(resultsformat, name) for name in SENSOR_FIELDS):
        from .propagation.coupled_propagators._coupled import sample_sensors
        from .propagation.coupled_propagators._coupled_dynamics import _prepare
        sensor_rows = sample_sensors(times, vectors, state.orbit.julian_date,
            propagator.dt * sensor_every, _prepare(satellite, environment, dynamics), satellite.adcs, seed)
    interval = propagator.dt * resultsformat.N
    on_grid = np.abs(times / interval - np.round(times / interval)) < 1e-9
    if sensor_rows is not None:
        sensor_rows = {name: values[on_grid] for name, values in sensor_rows.items()}
    return record_samples(times[on_grid], vectors[on_grid], resultsformat,
                          satellite, environment, dynamics, state.orbit.julian_date, backend, sensor_rows)


__all__ = ["ResultsFormat", "SatelliteResult", "SimulationResult", "simulate"]
