#[pyclass]
pub struct SimulationConfig {
    satellite: PyObject,
    environment: PyObject,
    state: PyObject,
    propagator: PyObject,
    solver: PyObject,
    backend: String,
    state_record: PyObject,
    julian_date: f64,
}

#[pymethods]
impl SimulationConfig {
    #[new]
    #[pyo3(signature = (satellite, environment, state, propagator, solver, backend="Python"))]
    fn new(
        py: Python<'_>,
        satellite: PyObject,
        environment: PyObject,
        state: PyObject,
        propagator: PyObject,
        solver: PyObject,
        backend: &str,
    ) -> PyResult<Self> {
        if !["Python", "C", "Rust", "JAX"].contains(&backend) {
            return Err(PyValueError::new_err(
                "backend must be Python, C, Rust, or JAX",
            ));
        }

        let wheel_count = satellite
            .as_ref(py)
            .getattr("adcs")?
            .getattr("reaction_wheels")?
            .len()?;
        let state_record = state
            .as_ref(py)
            .call_method1("to_record", (wheel_count,))?
            .into_py(py);
        let julian_date = state
            .as_ref(py)
            .getattr("orbit")?
            .getattr("julian_date")?
            .extract::<f64>()?;

        Ok(Self {
            satellite,
            environment,
            state,
            propagator,
            solver,
            backend: backend.to_owned(),
            state_record,
            julian_date,
        })
    }

    #[getter]
    fn satellite(&self, py: Python<'_>) -> PyObject {
        self.satellite.clone_ref(py)
    }
    #[setter]
    fn set_satellite(&mut self, value: PyObject) {
        self.satellite = value;
    }

    #[getter]
    fn environment(&self, py: Python<'_>) -> PyObject {
        self.environment.clone_ref(py)
    }
    #[setter]
    fn set_environment(&mut self, value: PyObject) {
        self.environment = value;
    }

    #[getter]
    fn state(&self, py: Python<'_>) -> PyObject {
        self.state.clone_ref(py)
    }
    #[setter]
    fn set_state(&mut self, value: PyObject) {
        self.state = value;
    }

    #[getter]
    fn propagator(&self, py: Python<'_>) -> PyObject {
        self.propagator.clone_ref(py)
    }
    #[setter]
    fn set_propagator(&mut self, value: PyObject) {
        self.propagator = value;
    }

    #[getter]
    fn solver(&self, py: Python<'_>) -> PyObject {
        self.solver.clone_ref(py)
    }
    #[setter]
    fn set_solver(&mut self, value: PyObject) {
        self.solver = value;
    }

    #[getter]
    fn backend(&self) -> &str {
        &self.backend
    }
    #[setter]
    fn set_backend(&mut self, value: String) {
        self.backend = value;
    }

    #[getter]
    fn state_record(&self, py: Python<'_>) -> PyObject {
        self.state_record.clone_ref(py)
    }

    #[getter]
    fn julian_date(&self) -> f64 {
        self.julian_date
    }
}
