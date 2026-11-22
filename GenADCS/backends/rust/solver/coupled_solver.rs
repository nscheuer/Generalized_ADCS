#[pyclass]
struct CoupledSolver {
    #[pyo3(get, set)]
    propagator: PyObject,
    #[pyo3(get, set)]
    tf: f64,
    #[pyo3(get, set)]
    progress: bool,
    #[pyo3(get, set)]
    progress_chunk_steps: i32,
}
#[pymethods]
impl CoupledSolver {
    #[new]
    #[pyo3(signature=(propagator,tf,progress=false,progress_chunk_steps=500))]
    fn new(
        propagator: PyObject,
        tf: f64,
        progress: bool,
        progress_chunk_steps: i32,
    ) -> PyResult<Self> {
        if !tf.is_finite() || tf < 0. {
            return Err(PyValueError::new_err(
                "tf must be finite and nonnegative seconds",
            ));
        }
        if progress_chunk_steps <= 0 {
            return Err(PyValueError::new_err(
                "progress_chunk_steps must be a positive integer",
            ));
        }
        Ok(Self {
            propagator,
            tf,
            progress,
            progress_chunk_steps,
        })
    }
    #[getter]
    fn dt(&self, py: Python<'_>) -> PyResult<f64> {
        self.propagator.as_ref(py).getattr("dt")?.extract()
    }
    fn step(
        &self,
        py: Python<'_>,
        state: PyObject,
        satellite: PyObject,
        environment: PyObject,
        dynamics: PyObject,
    ) -> PyResult<PyObject> {
        Ok(self
            .propagator
            .as_ref(py)
            .call_method1("step", (state, satellite, environment, dynamics))?
            .into_py(py))
    }
}
