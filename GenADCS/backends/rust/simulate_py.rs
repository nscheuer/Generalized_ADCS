use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

#[pyfunction]
#[pyo3(signature=(satellite,environment,dynamics,state,solver,backend="Rust",resultsformat=None,seed=0))]
pub fn simulate(
    py: Python<'_>,
    satellite: PyObject,
    environment: PyObject,
    dynamics: PyObject,
    mut state: PyObject,
    mut solver: PyObject,
    backend: &str,
    resultsformat: Option<PyObject>,
    seed: u64,
) -> PyResult<PyObject> {
    if !backend.eq_ignore_ascii_case("rust") {
        return Err(PyValueError::new_err("backend must be 'rust'"));
    }
    let solver_module = py.import("GenADCS.backends.rust.solver._solver")?;
    let is_solver = solver
        .as_ref(py)
        .is_instance(solver_module.getattr("Solver")?)?;
    if is_solver && solver.as_ref(py).getattr("mode")?.extract::<String>()? == "coupled" {
        solver = solver.as_ref(py).getattr("coupled")?.into_py(py);
    }
    let is_coupled = !solver.is_none(py)
        && solver
            .as_ref(py)
            .is_instance(solver_module.getattr("CoupledSolver")?)?;
    let is_split = !solver.is_none(py)
        && solver
            .as_ref(py)
            .is_instance(solver_module.getattr("Solver")?)?
        && solver.as_ref(py).getattr("mode")?.extract::<String>()? == "split";
    if !is_coupled && !is_split {
        return Err(PyTypeError::new_err(
            "solver must be a CoupledSolver or a split Solver",
        ));
    }
    let rust_states = py.import("GenADCS.backends.rust.state")?;
    if !state
        .as_ref(py)
        .is_instance(rust_states.getattr("State")?)?
    {
        state = rust_states
            .getattr("State")?
            .call_method1("from_python", (state,))?
            .into_py(py);
    }
    let solver_ref = solver.as_ref(py);
    let mode = if is_coupled { "coupled" } else { "split" };
    let tf: f64 = solver_ref.getattr("tf")?.extract()?;
    let mut format = match resultsformat {
        Some(value) => value,
        None => py
            .import("GenADCS.backends.shared.simulationresult.resultsformat")?
            .getattr("ResultsFormat")?
            .call0()?
            .into_py(py),
    };
    let adcs = satellite.as_ref(py).getattr("adcs")?;
    let sensor_n: usize = adcs.getattr("N")?.extract()?;
    if sensor_n == 0 {
        return Err(PyValueError::new_err("ADCSSubsystem.N must be positive"));
    }
    let estimator = adcs.getattr("estimator")?.into_py(py);
    if !estimator.is_none(py) {
        let flags = PyDict::new(py);
        for name in estimator.as_ref(py).getattr("record")?.iter()? {
            flags.set_item(name?, true)?;
        }
        format = py
            .import("dataclasses")?
            .getattr("replace")?
            .call((format,), Some(flags))?
            .into_py(py);
    }
    let n: usize = format.as_ref(py).getattr("N")?.extract()?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("progress", solver_ref.getattr("progress")?)?;
    kwargs.set_item(
        "progress_chunk_steps",
        solver_ref.getattr("progress_chunk_steps")?,
    )?;
    if n == 0 {
        return Err(PyValueError::new_err("ResultsFormat.N must be positive"));
    }
    let has_sensor = !adcs.getattr("sensors")?.is_none() && adcs.getattr("sensors")?.len()? > 0;
    let stride = if has_sensor || !estimator.is_none(py) {
        let (mut a, mut b) = (n, sensor_n);
        while b != 0 {
            let r = a % b;
            a = b;
            b = r;
        }
        a
    } else {
        n
    };
    kwargs.set_item("record_every", stride)?;
    let environment_for_record = environment.clone_ref(py);
    let dynamics_for_record = dynamics.clone_ref(py);
    let satellite_for_record = satellite.clone_ref(py);
    let propagator = if mode == "coupled" {
        solver_ref.getattr("propagator")?.into_py(py)
    } else {
        let orbital = solver_ref.getattr("orbital")?;
        let attitude = solver_ref.getattr("attitude")?;
        let orbital_propagator = orbital.getattr("propagator")?;
        if orbital_propagator.get_type().name()? != "KeplerianOrbitalPropagator" {
            return Err(PyTypeError::new_err(
                "Rust split simulation requires KeplerianOrbitalPropagator",
            ));
        }
        let dt: f64 = orbital.getattr("dt")?.extract()?;
        let attitude_propagator = attitude.getattr("propagator")?;
        let attitude_type = attitude_propagator.get_type().name()?;
        let reference_mode: i32 = if attitude_type == "ReferenceAttitudePropagator" {
            let mode = attitude_propagator.getattr("mode")?;
            if let Ok(value) = mode.extract::<i32>() {
                value
            } else {
                match mode.extract::<String>()?.as_str() {
                    "nadir" | "nadir-pointing" => 0,
                    "along-track" => 1,
                    "sun" | "sun-pointing" => 2,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "unsupported reference attitude mode {other:?}"
                        )))
                    }
                }
            }
        } else if attitude_type == "RK4AttitudePropagator" {
            -1
        } else {
            return Err(PyTypeError::new_err(
                "Rust split simulation requires RK4AttitudePropagator or ReferenceAttitudePropagator",
            ));
        };
        py.import("GenADCS.backends.rust.propagation.separate_solver")?
            .getattr("SeparateSolver")?
            .call1((dt, reference_mode))?
            .into_py(py)
    };
    let dt: f64 = propagator.as_ref(py).getattr("dt")?.extract()?;
    let result = propagator.as_ref(py).call_method(
        "integrate",
        (state.clone_ref(py), satellite, environment, dynamics, tf),
        Some(kwargs),
    )?;
    let trajectory = result.downcast::<PyTuple>().map_err(|_| {
        PyValueError::new_err("propagator.integrate must return times and state vectors")
    })?;
    if trajectory.len() != 2 {
        return Err(PyValueError::new_err(
            "propagator.integrate must return times and state vectors",
        ));
    }
    let times = trajectory.get_item(0)?.into_py(py);
    let vectors = trajectory.get_item(1)?;
    let time_array: PyReadonlyArray1<'_, f64> = times.as_ref(py).extract()?;
    let state_array: PyReadonlyArray2<'_, f64> = vectors.extract()?;
    let time_values = time_array.as_array();
    let state_values = state_array.as_array();
    if time_values.is_empty()
        || state_values.nrows() != time_values.len()
        || state_values.ncols() < 13
        || time_values.iter().any(|t| !t.is_finite())
    {
        return Err(PyValueError::new_err(
            "trajectory arrays must be non-empty, rectangular and aligned",
        ));
    }
    let initial_orbit = state.as_ref(py).getattr("orbit")?;
    let epoch_jd: f64 = initial_orbit.getattr("julian_date")?.extract()?;
    let interval: f64 = dt * n as f64;
    let rows: Vec<usize> = time_values
        .iter()
        .enumerate()
        .filter_map(|(i, t)| {
            if (t / interval - (t / interval).round()).abs() < 1e-9 {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    let row_indices = rows.to_pyarray(py);
    let all_rows = rows.len() == time_values.len();
    let sampled_times = if all_rows {
        times.as_ref(py)
    } else {
        times.as_ref(py).get_item(row_indices)?
    };
    let sampled_vectors = if all_rows {
        vectors
    } else {
        vectors.get_item(row_indices)?
    };
    let selected = PyDict::new(py);
    if has_sensor || !estimator.is_none(py) {
        let prepared = py
            .import("GenADCS.backends.rust.propagation.coupled_propagators._coupled_dynamics")?
            .getattr("_prepare")?
            .call1((
                satellite_for_record.as_ref(py),
                environment_for_record.as_ref(py),
                dynamics_for_record.as_ref(py),
            ))?;
        let adcs = satellite_for_record.as_ref(py).getattr("adcs")?;
        let sensor_n: usize = adcs.getattr("N")?.extract()?;
        if sensor_n == 0 {
            return Err(PyValueError::new_err("ADCSSubsystem.N must be positive"));
        }
        let samples = py
            .import("GenADCS.backends.rust.propagation.coupled_propagators._coupled")?
            .getattr("sample_sensors")?
            .call1((
                times.as_ref(py),
                vectors,
                epoch_jd,
                dt * sensor_n as f64,
                prepared,
                adcs.getattr("sensors")?,
                environment_for_record.as_ref(py),
                seed,
                estimator,
                format
                    .as_ref(py)
                    .getattr("normalized_estimation_error_squared")?,
            ))?;
        for (key, value) in samples.downcast::<PyDict>()? {
            selected.set_item(
                key,
                if all_rows {
                    value
                } else {
                    value.get_item(row_indices)?
                },
            )?;
        }
    } else {
        let empty = numpy::ndarray::Array2::<f64>::zeros((rows.len(), 0)).into_pyarray(py);
        for name in [
            "gyroscope",
            "magnetometer",
            "autonomous_star_tracker",
            "coarse_sun_sensor",
            "fine_sun_sensor",
            "fine_earth_sensor",
            "gnss",
            "accelerometer",
        ] {
            for suffix in ["clean", "reading"] {
                selected.set_item(format!("{name}_{suffix}"), empty)?;
            }
        }
    }
    let result = py
        .import("GenADCS.backends.shared.simulationresult.record")?
        .getattr("record_samples")?
        .call1((
            sampled_times,
            sampled_vectors,
            format,
            satellite_for_record,
            environment_for_record,
            dynamics_for_record,
            epoch_jd,
            backend,
            selected,
        ))?;
    Ok(result.into_py(py))
}
