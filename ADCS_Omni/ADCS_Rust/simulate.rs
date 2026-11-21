#[pyclass]
struct SimulationResult {
    #[pyo3(get)]
    times_s: PyObject,
    #[pyo3(get)]
    state_vectors: PyObject,
    #[pyo3(get)]
    final_state: PyObject,
}

fn copied_slice(
    py: Python<'_>,
    numpy: &PyAny,
    row: &PyAny,
    start: isize,
    end: isize,
) -> PyResult<PyObject> {
    let values = row.get_item(PySlice::new(py, start, end, 1))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("copy", true)?;
    Ok(numpy
        .call_method("array", (values,), Some(kwargs))?
        .into_py(py))
}

#[pyfunction]
#[pyo3(signature=(satellite,environment,dynamics,state,solver,backend="Rust"))]
fn simulate(
    py: Python<'_>,
    satellite: PyObject,
    environment: PyObject,
    dynamics: PyObject,
    mut state: PyObject,
    mut solver: PyObject,
    backend: &str,
) -> PyResult<SimulationResult> {
    if backend != "Rust" {
        return Err(PyValueError::new_err("backend must be Rust"));
    }
    let solver_module = py.import("ADCS_Omni.ADCS_Rust.solver._solver")?;
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
    let rust_states = py.import("ADCS_Omni.ADCS_Rust.state")?;
    if !state
        .as_ref(py)
        .is_instance(rust_states.getattr("State")?)?
    {
        state = rust_states
            .getattr("State")?
            .call_method1("from_python", (state,))?
            .into_py(py);
    }
    let state_module = rust_states;
    let solver_ref = solver.as_ref(py);
    let tf: f64 = solver_ref.getattr("tf")?.extract()?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("progress", solver_ref.getattr("progress")?)?;
    kwargs.set_item(
        "progress_chunk_steps",
        solver_ref.getattr("progress_chunk_steps")?,
    )?;
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
    let row = vectors.get_item(-1)?;
    let q = copied_slice(py, numpy, row, 0, 4)?;
    let rates = copied_slice(py, numpy, row, 4, 7)?;
    let position = copied_slice(py, numpy, row, 7, 10)?;
    let velocity = copied_slice(py, numpy, row, 10, 13)?;
    let momentum = copied_slice(py, numpy, row, 13, shape[1] as isize)?;
    let initial_orbit = state.as_ref(py).getattr("orbit")?;
    let jd: f64 = initial_orbit.getattr("julian_date")?.extract()?;
    let attitude = state_module.getattr("AttitudeState")?.call1((q, rates))?;
    let orbit = state_module
        .getattr("OrbitalState")?
        .getattr("from_cartesian")?
        .call1((
            position,
            velocity,
            jd + tf / 86400.,
            initial_orbit.getattr("ephemeris")?,
        ))?;
    let wheel_kwargs = PyDict::new(py);
    wheel_kwargs.set_item("wheel_momentum", momentum)?;
    let internal = state_module
        .getattr("WheelMomentumState")?
        .call((), Some(wheel_kwargs))?;
    let final_state = state_module
        .getattr("State")?
        .call1((attitude, orbit, internal))?;
    Ok(SimulationResult {
        times_s: times,
        state_vectors: vectors.into_py(py),
        final_state: final_state.into_py(py),
    })
}
