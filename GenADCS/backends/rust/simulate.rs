#[pyfunction]
#[pyo3(signature=(satellite,environment,dynamics,state,solver,backend="Rust",resultsformat=None))]
fn simulate(
    py: Python<'_>,
    satellite: PyObject,
    environment: PyObject,
    dynamics: PyObject,
    mut state: PyObject,
    mut solver: PyObject,
    backend: &str,
    resultsformat: Option<PyObject>,
) -> PyResult<PyObject> {
    if backend != "Rust" {
        return Err(PyValueError::new_err("backend must be Rust"));
    }
    let solver_module = py.import("GenADCS.backends.rust.solver._solver")?;
    if solver
        .as_ref(py)
        .is_instance(solver_module.getattr("Solver")?)?
    {
        solver = solver.as_ref(py).getattr("coupled")?.into_py(py);
    }
    if solver.is_none(py)
        || !solver
            .as_ref(py)
            .is_instance(solver_module.getattr("CoupledSolver")?)?
    {
        return Err(PyTypeError::new_err(
            "solver must be a CoupledSolver or a Solver containing one",
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
    let tf: f64 = solver_ref.getattr("tf")?.extract()?;
    let format = match resultsformat {
        Some(value) => value,
        None => py.import("GenADCS.backends.shared.simulationresult.resultsformat")?
            .getattr("ResultsFormat")?.call0()?.into_py(py),
    };
    let n: usize = format.as_ref(py).getattr("N")?.extract()?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("progress", solver_ref.getattr("progress")?)?;
    kwargs.set_item(
        "progress_chunk_steps",
        solver_ref.getattr("progress_chunk_steps")?,
    )?;
    kwargs.set_item("record_every", n)?;
    let environment_for_record = environment.clone_ref(py);
    let dynamics_for_record = dynamics.clone_ref(py);
    let satellite_for_record = satellite.clone_ref(py);
    let result = solver_ref.getattr("propagator")?.call_method(
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
    let numpy = py.import("numpy")?;
    let vectors = numpy
        .call_method1("asarray", (trajectory.get_item(1)?,))?
        .call_method1("astype", ("float64",))?;
    let shape: Vec<usize> = vectors.getattr("shape")?.extract()?;
    if shape.len() != 2 || shape[0] < 1 || shape[1] < 13 {
        return Err(PyValueError::new_err(
            "state vectors must have at least one row and 13 columns",
        ));
    }
    let initial_orbit = state.as_ref(py).getattr("orbit")?;
    let epoch_jd: f64 = initial_orbit.getattr("julian_date")?.extract()?;
    let interval: f64 = solver_ref.getattr("propagator")?.getattr("dt")?.extract::<f64>()? * n as f64;
    let last_time: f64 = times.as_ref(py).get_item(-1)?.extract()?;
    let on_grid = (last_time / interval - (last_time / interval).round()).abs() < 1e-9;
    let end = if on_grid { shape[0] } else { shape[0] - 1 };
    let rows = PySlice::new(py, 0, end as isize, 1);
    let sampled_times = times.as_ref(py).get_item(rows)?;
    let sampled_vectors = vectors.get_item((rows, PySlice::new(py, 0, shape[1] as isize, 1)))?;
    let result = py.import("GenADCS.backends.shared.simulationresult.record")?
        .getattr("record_samples")?.call1((sampled_times, sampled_vectors, format,
            satellite_for_record, environment_for_record, dynamics_for_record, epoch_jd, backend))?;
    Ok(result.into_py(py))
}
