use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::uniform_noise::UniformNoise as NativeUniformNoise;

#[pyclass(name = "UniformNoise")]
pub struct UniformNoise {
    inner: NativeUniformNoise,
}

#[pymethods]
impl UniformNoise {
    #[new]
    fn new(lower: f64, upper: f64) -> PyResult<Self> {
        Ok(Self {
            inner: NativeUniformNoise::new(lower, upper).map_err(PyValueError::new_err)?,
        })
    }

    #[getter]
    fn lower(&self) -> f64 {
        self.inner.lower
    }

    #[setter]
    fn set_lower(&mut self, lower: f64) -> PyResult<()> {
        self.inner =
            NativeUniformNoise::new(lower, self.inner.upper).map_err(PyValueError::new_err)?;
        Ok(())
    }

    #[getter]
    fn upper(&self) -> f64 {
        self.inner.upper
    }

    #[setter]
    fn set_upper(&mut self, upper: f64) -> PyResult<()> {
        self.inner =
            NativeUniformNoise::new(self.inner.lower, upper).map_err(PyValueError::new_err)?;
        Ok(())
    }
}
