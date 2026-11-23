use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;

#[pyclass]
pub struct FixedBias { #[pyo3(get, set)] pub value: f64 }

#[pymethods]
impl FixedBias {
    #[new]
    fn new(value: f64) -> Self { Self { value } }
    #[pyo3(signature = (value, dt, rng=None))]
    fn apply(&self, value: &PyAny, dt: f64, rng: Option<&PyAny>) -> PyResult<()> {
        let _ = (value, dt, rng);
        Err(PyNotImplementedError::new_err(""))
    }
}
