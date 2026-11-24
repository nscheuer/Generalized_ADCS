"""Simulation entry point for the coupled Numba backend."""

import math
import numpy as np

from .helpers.input_output.resultsformat import ResultsFormat
from GenADCS.backends.shared.simulationresult import SimulationResult
from GenADCS.backends.shared.simulationresult.record import record_samples
from .solver import CoupledSolver, Solver
from .state import State
from .hardware.ADCS_subsystem.sensors.sensor_sampling import SENSOR_NAMES, prepare_sensors
from .propagation.coupled_propagators.coupled_rk4_propagator import CoupledRK4Propagator
from .helpers.random.philox import validate_seed


def simulate(satellite, environment, dynamics, state, solver, backend="Python", resultsformat=None, seed=0):
    """Integrate the coupled orbit and attitude, sampling sensors in Numba.

    ``seed`` is a 64-bit Philox key.  A sensor's draw is identified by its
    subsystem tick and position in ``satellite.adcs.sensors``; adaptive solver
    substeps and result recording frequency do not advance its random stream.
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
    if isinstance(satellite.adcs.N, bool) or not isinstance(satellite.adcs.N, int) or satellite.adcs.N < 1:
        raise ValueError("ADCSSubsystem.N must be a positive integer")
    seed = validate_seed(seed)

    sensor_every = satellite.adcs.N if satellite.adcs.sensors else 0
    # RK4 samples sensors directly at completed steps, without storing those
    # states. DOP853 must stop at sensor boundaries, even between result rows.
    record_every = (math.gcd(resultsformat.N, sensor_every)
                    if sensor_every and not isinstance(solver.propagator, CoupledRK4Propagator)
                    else resultsformat.N)
    integration = solver.propagator.integrate(
        state, satellite, environment, dynamics, solver.tf,
        progress=solver.progress, progress_chunk_steps=solver.progress_chunk_steps,
        record_every=record_every, sensor_every=sensor_every,
        seed=seed,
    )
    times_s, state_vectors = integration[:2]
    interval = solver.propagator.dt * resultsformat.N
    on_grid = np.abs(times_s / interval - np.round(times_s / interval)) < 1e-9
    times_s = times_s[on_grid]
    state_vectors = state_vectors[on_grid]

    config = prepare_sensors(satellite.adcs.sensors, environment)
    counts, starts = config[-2:]
    sensor_rows = {}
    for kind, name in enumerate(SENSOR_NAMES):
        width = int(counts[kind])
        for suffix, source in (("_clean", integration[2] if sensor_every else None),
                               ("_reading", integration[3] if sensor_every else None)):
            values = np.full((len(times_s), width), np.nan)
            if source is not None:
                for row, time in enumerate(times_s):
                    tick = int(round(time / (solver.propagator.dt * sensor_every)))
                    if tick < source.shape[0] and abs(time - tick * solver.propagator.dt * sensor_every) < 1e-9:
                        values[row] = source[tick, starts[kind]:starts[kind] + width]
            sensor_rows[name + suffix] = values
    return record_samples(times_s, state_vectors, resultsformat, satellite,
                          environment, dynamics, state.orbit.julian_date, backend, sensor_rows)
