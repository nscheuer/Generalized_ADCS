use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{
    bias_py::{array_values, numpy_array},
    random_walk_bias::RandomWalkBias as NativeRandomWalkBias,
};

#[pyclass(name = "RandomWalkBias")]
pub struct RandomWalkBias {
    inner: NativeRandomWalkBias,
}

#[pymethods]
impl RandomWalkBias {
    #[new]
    #[pyo3(signature = (standard_deviation, initial_value=0.0))]
    fn new(standard_deviation: f64, initial_value: f64) -> PyResult<Self> {
        Ok(Self {
            inner: NativeRandomWalkBias::new(standard_deviation, initial_value)
                .map_err(PyValueError::new_err)?,
        })
    }
    #[getter]
    fn standard_deviation(&self) -> f64 {
        self.inner.standard_deviation
    }
    #[setter]
    fn set_standard_deviation(&mut self, value: f64) -> PyResult<()> {
        self.inner = NativeRandomWalkBias::new(value, self.inner.initial_value)
            .map_err(PyValueError::new_err)?;
        Ok(())
    }
    #[getter]
    fn initial_value(&self) -> f64 {
        self.inner.initial_value
    }
    #[setter]
    fn set_initial_value(&mut self, value: f64) -> PyResult<()> {
        self.inner = NativeRandomWalkBias::new(self.inner.standard_deviation, value)
            .map_err(PyValueError::new_err)?;
        Ok(())
    }
    #[pyo3(signature = (value, dt, rng=None))]
    fn apply(
        &self,
        py: Python<'_>,
        value: &PyAny,
        dt: f64,
        rng: Option<&PyAny>,
    ) -> PyResult<PyObject> {
        let (values, shape) = array_values(py, value)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("size", values.len())?;
        let normals: Vec<f64> = match rng {
            Some(rng) => rng.call_method("normal", (0.0, 1.0), Some(kwargs))?,
            None => py.import("numpy")?.getattr("random")?.call_method(
                "standard_normal",
                (),
                Some(kwargs),
            )?,
        }
        .call_method0("ravel")?
        .call_method0("tolist")?
        .extract()?;
        let result = self
            .inner
            .apply(&values, dt, &normals)
            .map_err(PyValueError::new_err)?;
        numpy_array(py, &result, &shape)
    }
}
