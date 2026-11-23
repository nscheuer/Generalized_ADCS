use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;

#[pyclass]
pub struct Bias;

#[pymethods]
impl Bias {
    #[new]
    fn new() -> Self { Self }

    #[pyo3(signature = (value, dt, rng=None))]
    fn apply(&self, _value: &PyAny, _dt: f64, _rng: Option<&PyAny>) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(""))
    }
}
