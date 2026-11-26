"""Python-facing orchestration for the compiled C++ propagators."""

import numpy as np
import math
from dataclasses import replace

from GenADCS.backends.shared.simulationresult import SimulationResult, SatelliteResult
from GenADCS.backends.shared.simulationresult.record import record_samples
from GenADCS.backends.shared.simulationresult.simulationresult import SENSOR_FIELDS
from GenADCS.backends.shared.simulationresult.resultsformat import ResultsFormat


def simulate(satellite, environment, dynamics, state, solver, backend="C", resultsformat=None, seed=0):
    """Integrate with the C++ propagator and assemble the sampled result in Python."""
    if backend not in ("C", "cpp"):
        raise NotImplementedError(f"backend {backend!r} is not implemented for GenADCS C++")
    from .solver import CoupledSolver, Solver
    from .state import State as CppState
    from GenADCS.backends.python.state import State as PythonState

    if not isinstance(state, (CppState, PythonState)):
        raise TypeError("state must be a GenADCS State")
    if resultsformat is None:
        resultsformat = ResultsFormat()
    if not isinstance(resultsformat, ResultsFormat):
        raise TypeError("resultsformat must be a ResultsFormat")
    if isinstance(resultsformat.N, bool) or not isinstance(resultsformat.N, int) or resultsformat.N < 1:
        raise ValueError("ResultsFormat.N must be a positive integer")
    adcs = getattr(satellite, "adcs", None)
    estimator = getattr(adcs, "estimator", None)
    if estimator is not None and estimator.record:
        resultsformat = replace(resultsformat, **{name: True for name in estimator.record})
    sensors = getattr(adcs, "sensors", ())
    sensor_every = getattr(adcs, "N", 1)
    if isinstance(sensor_every, bool) or sensor_every < 1:
        raise ValueError("ADCSSubsystem.N must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if any(type(s).__name__ in ("CoarseSunSensor", "FineSunSensor", "AutonomousStarTracker") for s in sensors) and environment.ephemeris is None:
        raise ValueError("Sun and star tracker sensors require an ephemeris")

    separate = isinstance(solver, Solver) and solver.coupled is None
    propagator = None
    native_samples = False
    sensor_rows = None
    record_every = math.gcd(resultsformat.N, sensor_every) if sensors or estimator is not None else resultsformat.N
    diagnostic_fields = {
        name for name in (
            "estimated_attitude", "estimated_rate", "estimated_covariance",
            "estimated_attitude_covariance", "estimated_rate_covariance",
            "measurement_innovation", "measurement_innovation_covariance",
            "normalized_innovation_squared", "normalized_estimation_error_squared",
            "measurement_residual", "kalman_gain",
        ) if getattr(resultsformat, name)
    }
    if separate:
        if getattr(solver, "orbital", None) is None or getattr(solver, "attitude", None) is None:
            raise TypeError("separate Solver requires orbital and attitude solvers")
        orbital_source = solver.orbital.propagator
        attitude_source = solver.attitude.propagator
        tf = float(solver.tf)
        dt = float(orbital_source.dt)
        if not math.isfinite(tf) or tf < 0:
            raise ValueError("separate solvers require a finite nonnegative tf")
        if float(attitude_source.dt) != dt:
            raise ValueError("separate propagators require matching dt")
        from .propagation.orbital_propagators import KeplerianOrbitalPropagator
        from .propagation.attitude_propagators import (
            RK4AttitudePropagator,
            ReferenceAttitudePropagator,
        )

        if type(orbital_source).__name__ == "KeplerianOrbitalPropagator" and not isinstance(orbital_source, KeplerianOrbitalPropagator):
            orbital = KeplerianOrbitalPropagator(dt)
        elif isinstance(orbital_source, KeplerianOrbitalPropagator):
            orbital = orbital_source
        else:
            raise TypeError("C++ separate simulation requires a Keplerian orbit propagator")

        if type(attitude_source).__name__ == "RK4AttitudePropagator":
            attitude = RK4AttitudePropagator(dt)
        elif type(attitude_source).__name__ == "ReferenceAttitudePropagator":
            from .propagation.attitude_propagators.reference_attitude_propagator import ReferenceAttitudePropagator as NativeReference
            mode_names = {0: "nadir", 1: "along-track", 2: "sun"}
            raw_mode = getattr(attitude_source, "mode", "sun")
            if isinstance(raw_mode, str):
                mode = raw_mode
            else:
                mode = mode_names[int(raw_mode)]
            attitude = NativeReference(dt, mode)
        elif isinstance(attitude_source, (RK4AttitudePropagator, ReferenceAttitudePropagator)):
            attitude = attitude_source
        else:
            raise TypeError("C++ separate simulation requires RK4 or reference attitude propagation")

        from .propagation._separate import (
            integrate_separate_reference,
            simulate_separate_rk4,
        )
        from .propagation.coupled_propagators._state_layout import pack_state
        initial = pack_state(state, state.internal.wheel_momentum.size)
        mu = float(environment.planetary_body.gravitational_parameter)
        epoch = state.orbit.julian_date
        if isinstance(attitude, RK4AttitudePropagator):
            from .propagation.coupled_propagators._coupled_dynamics import _prepare
            runtime = _prepare(satellite, environment, dynamics)
            # The native loop propagates, samples, updates the estimator, and
            # records each step before advancing. Python only assembles results.
            times, vectors, sensor_rows = simulate_separate_rk4(
                initial, mu, epoch, dt, tf, resultsformat.N, runtime,
                satellite.adcs, seed, diagnostic_fields,
                solver.progress, solver.progress_chunk_steps,
            )
            native_samples = True
        else:
            if attitude.mode == 2 and environment.ephemeris is None:
                raise ValueError("Sun-pointing reference attitude requires an ephemeris")
            times, vectors = integrate_separate_reference(
                initial, mu, epoch, dt, tf, record_every,
                {0: "nadir", 1: "along-track", 2: "sun"}[attitude.mode],
                solver.progress, solver.progress_chunk_steps,
            )
        dt = orbital.dt
    else:
        solver = solver.coupled if isinstance(solver, Solver) else solver
        if not isinstance(solver, CoupledSolver):
            raise TypeError("solver must be a CoupledSolver or a Solver containing one")
        propagator = solver.propagator
        dt = propagator.dt
        from .propagation._separate import simulate_coupled
        from .propagation.coupled_propagators._coupled_dynamics import _prepare
        from .propagation.coupled_propagators._state_layout import pack_state
        from .propagation.coupled_propagators.coupled_dop853_propagator import CoupledDOP853Propagator
        from .propagation.coupled_propagators.coupled_rk4_propagator import CoupledRK4Propagator
        if not isinstance(propagator, (CoupledRK4Propagator, CoupledDOP853Propagator)):
            raise TypeError("native coupled simulation requires RK4 or DOP853")
        adaptive = isinstance(propagator, CoupledDOP853Propagator)
        runtime = _prepare(satellite, environment, dynamics)
        initial = pack_state(state, runtime.wheel_commands.size)
        atol = (np.array([propagator.atol[0]]) if propagator._scalar_atol else propagator.atol) if adaptive else np.array([1e-12])
        times, vectors, sensor_rows = simulate_coupled(
            initial, state.orbit.julian_date, dt, float(solver.tf), resultsformat.N,
            runtime, satellite.adcs, seed, diagnostic_fields, adaptive,
            propagator.rtol if adaptive else 1e-9, atol,
            propagator.max_step if adaptive else dt,
            propagator.min_step if adaptive else 1e-6,
            solver.progress, solver.progress_chunk_steps,
        )
        native_samples = True
    times = np.asarray(times, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    if times.ndim != 1 or times.size < 1:
        raise ValueError("propagator.integrate must return at least one sample")
    if vectors.ndim != 2 or vectors.shape[0] != times.size or vectors.shape[1] < 13:
        raise ValueError("state vectors must have matching rows and at least 13 columns")

    if not native_samples and (sensors or (not separate and any(getattr(resultsformat, name) for name in SENSOR_FIELDS))):
        from .propagation.coupled_propagators._coupled import sample_sensors
        from .propagation.coupled_propagators._coupled_dynamics import _prepare
        sensor_rows = sample_sensors(times, vectors, state.orbit.julian_date,
            dt * sensor_every, _prepare(satellite, environment, dynamics), satellite.adcs, seed)
    if estimator is not None and not native_samples:
        from .hardware.ADCS_subsystem.estimator.runtime import sample_estimator
        if sensor_rows is None:
            sensor_rows = {}
        sample_estimator(times, vectors, sensor_rows, satellite.adcs,
                         dt * sensor_every, resultsformat)
    interval = dt * resultsformat.N
    on_grid = np.abs(times / interval - np.round(times / interval)) < 1e-9
    if sensor_rows is not None:
        sensor_rows = {name: values[on_grid] for name, values in sensor_rows.items()}
    record_args = (times[on_grid], vectors[on_grid], resultsformat,
                   satellite, environment, dynamics, state.orbit.julian_date, "C")
    if sensor_rows is None:
        return record_samples(*record_args)
    return record_samples(*record_args, sensor_rows)


__all__ = ["ResultsFormat", "SatelliteResult", "SimulationResult", "simulate"]
