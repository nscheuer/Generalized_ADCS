"""Native simulation sampling, estimator cadence, and portable hardware snapshots."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from GenADCS.backends.rust.propagation import (
    CoupledDOP853Propagator,
    CoupledRK4Propagator,
)
from GenADCS.backends.rust.simulate import ResultsFormat, simulate
from GenADCS.backends.rust.solver import CoupledSolver


@pytest.fixture
def setup():
    path = (
        Path(__file__).resolve().parents[4]
        / "examples/plotting/estimators/flagship_leo_all_sensors.py"
    )
    spec = importlib.util.spec_from_file_location("rust_estimator_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = []

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        captured.append(args[:4])
        raise Captured

    module.simulate = capture
    with pytest.raises(Captured):
        module.main()
    return captured[0]


@pytest.mark.parametrize("propagator", [CoupledRK4Propagator, CoupledDOP853Propagator])
def test_native_sampling_and_estimator_cadence(setup, propagator):
    satellite, _environment, _dynamics, _state = setup
    satellite.adcs.N = 2
    fmt = ResultsFormat(
        N=1,
        gyroscope_clean=True,
        gyroscope_reading=True,
        gnss_clean=True,
        estimated_attitude=True,
        estimated_rate=True,
        estimated_covariance=True,
        normalized_estimation_error_squared=True,
        normalized_innovation_squared=True,
    )
    integrator = (
        propagator(dt=0.5, max_step=0.5)
        if propagator is CoupledDOP853Propagator
        else propagator(dt=0.5)
    )
    solver = CoupledSolver(integrator, tf=4.25, progress=False)
    full = simulate(*setup, solver, resultsformat=fmt, seed=12).satellites["sat_000"]
    sparse_result = simulate(*setup, solver, resultsformat=replace(fmt, N=3), seed=12)
    sparse = sparse_result.satellites["sat_000"]
    np.testing.assert_allclose(sparse.times_s, full.times_s[::3])
    for name in full.fields:
        np.testing.assert_allclose(
            sparse.fields[name],
            full.fields[name][::3],
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
        )
    assert np.isnan(full.estimated_attitude[1::2]).all()
    assert np.isfinite(full.estimated_attitude[::2]).all()
    np.testing.assert_allclose(
        np.linalg.norm(full.estimated_attitude[::2], axis=1), 1.0, atol=1e-12
    )
    covariance = full.estimated_covariance[::2].reshape(-1, 6, 6)
    np.testing.assert_allclose(covariance, covariance.transpose(0, 2, 1), atol=1e-15)
    assert (np.linalg.eigvalsh(covariance) > 0).all()
    changed = simulate(*setup, solver, resultsformat=fmt, seed=13).satellites["sat_000"]
    np.testing.assert_array_equal(full.gyroscope_clean, changed.gyroscope_clean)
    assert not np.array_equal(
        full.gyroscope_reading[::2], changed.gyroscope_reading[::2]
    )
    rebuilt = sparse_result.satellites["sat_000"].to_satellite()
    assert rebuilt.adcs.N == 2
    assert len(rebuilt.adcs.sensors) == len(satellite.adcs.sensors)
    np.testing.assert_array_equal(
        rebuilt.adcs.estimator.quaternion, satellite.adcs.estimator.quaternion
    )


def test_record_flags_and_prediction_without_sensors(setup):
    satellite, _, _, _ = setup
    satellite.adcs.sensors = []
    satellite.adcs.estimator.record = ["estimated_attitude", "estimated_rate"]
    solver = CoupledSolver(CoupledRK4Propagator(dt=0.5), tf=2.0, progress=False)
    result = simulate(*setup, solver, resultsformat=ResultsFormat(N=2))
    assert np.isfinite(result.satellites["sat_000"].estimated_attitude).all()
