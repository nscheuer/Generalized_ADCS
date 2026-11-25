use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::quantization_noise::QuantizationNoise as NativeQuantizationNoise;

#[pyclass(name = "QuantizationNoise")]
pub struct QuantizationNoise {
    inner: NativeQuantizationNoise,
}

#[pymethods]
impl QuantizationNoise {
    #[new]
    fn new(step_size: f64) -> PyResult<Self> {
        Ok(Self {
            inner: NativeQuantizationNoise::new(step_size).map_err(PyValueError::new_err)?,
        })
    }

    #[getter]
    fn step_size(&self) -> f64 {
        self.inner.step_size
    }

    #[setter]
    fn set_step_size(&mut self, step_size: f64) -> PyResult<()> {
        self.inner = NativeQuantizationNoise::new(step_size).map_err(PyValueError::new_err)?;
        Ok(())
    }
}
