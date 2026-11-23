use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;

#[pyclass]
pub struct RandomWalkBias { #[pyo3(get, set)] pub standard_deviation: f64, #[pyo3(get, set)] pub initial_value: f64 }

#[pymethods]
impl RandomWalkBias {
    #[new]
    #[pyo3(signature = (standard_deviation, initial_value=0.0))]
    fn new(standard_deviation: f64, initial_value: f64) -> Self { Self { standard_deviation, initial_value } }
    #[pyo3(signature = (value, dt, rng=None))]
    fn apply(&self, value: &PyAny, dt: f64, rng: Option<&PyAny>) -> PyResult<()> {
        let _ = (value, dt, rng);
        Err(PyNotImplementedError::new_err(""))
    }
}
