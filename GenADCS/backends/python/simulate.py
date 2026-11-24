"""Simulation entry point for the coupled Numba backend."""

import math
from dataclasses import replace
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
    if satellite.adcs.estimator is not None and satellite.adcs.estimator.record:
        resultsformat = replace(resultsformat,
            **{name: True for name in satellite.adcs.estimator.record})
    if isinstance(satellite.adcs.N, bool) or not isinstance(satellite.adcs.N, int) or satellite.adcs.N < 1:
        raise ValueError("ADCSSubsystem.N must be a positive integer")
    seed = validate_seed(seed)

    sensor_every = satellite.adcs.N if satellite.adcs.sensors or satellite.adcs.estimator is not None else 0
    # RK4 samples sensors directly at completed steps, without storing those
    # states. DOP853 must stop at sensor boundaries, even between result rows.
    record_every = (math.gcd(resultsformat.N, sensor_every)
                    if sensor_every and not isinstance(solver.propagator, CoupledRK4Propagator)
                    else resultsformat.N)
    diagnostic_names = ("estimated_attitude", "estimated_rate", "estimated_attitude_covariance",
                        "estimated_rate_covariance", "measurement_innovation",
                        "measurement_innovation_covariance", "normalized_innovation_squared",
                        "normalized_estimation_error_squared", "measurement_residual", "kalman_gain",
                        "estimated_covariance")
    estimator_flags = sum(1 << i for i, name in enumerate(diagnostic_names)
                          if getattr(resultsformat, name))
    integration = solver.propagator.integrate(
        state, satellite, environment, dynamics, solver.tf,
        progress=solver.progress, progress_chunk_steps=solver.progress_chunk_steps,
        record_every=record_every, sensor_every=sensor_every,
        seed=seed, estimator_flags=estimator_flags,
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
    if satellite.adcs.estimator is not None:
        estimates = integration[4]
        for name, start, stop in (("estimated_attitude", 0, 4),
                                  ("estimated_rate", 4, 7),
                                  ("estimated_covariance", 7, 43),
                                  ("estimated_attitude_covariance", 78, 87),
                                  ("estimated_rate_covariance", 87, 96),
                                  ("measurement_innovation", 43, 46),
                                  ("measurement_innovation_covariance", 46, 55),
                                  ("normalized_innovation_squared", 55, 56),
                                  ("normalized_estimation_error_squared", 56, 57),
                                  ("measurement_residual", 57, 60),
                                  ("kalman_gain", 60, 78)):
            if not getattr(resultsformat, name, False):
                continue
            values = np.full((len(times_s), stop - start), np.nan)
            for row, time in enumerate(times_s):
                tick = int(round(time / (solver.propagator.dt * sensor_every)))
                if tick < estimates.shape[0] and abs(time - tick * solver.propagator.dt * sensor_every) < 1e-9:
                    values[row] = estimates[tick, start:stop]
            sensor_rows[name] = values
    return record_samples(times_s, state_vectors, resultsformat, satellite,
                          environment, dynamics, state.orbit.julian_date, backend, sensor_rows)
