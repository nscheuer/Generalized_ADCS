use super::gaussian_white_noise::GaussianWhiteNoise as NativeGaussianWhiteNoise;
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass(name = "GaussianWhiteNoise")]
pub struct GaussianWhiteNoise {
    inner: NativeGaussianWhiteNoise,
}
#[pymethods]
impl GaussianWhiteNoise {
    #[new]
    fn new(standard_deviation: f64) -> PyResult<Self> {
        Ok(Self {
            inner: NativeGaussianWhiteNoise::new(standard_deviation)
                .map_err(PyValueError::new_err)?,
        })
    }
    #[getter]
    fn standard_deviation(&self) -> f64 {
        self.inner.standard_deviation
    }
    #[setter]
    fn set_standard_deviation(&mut self, value: f64) -> PyResult<()> {
        self.inner = NativeGaussianWhiteNoise::new(value).map_err(PyValueError::new_err)?;
        Ok(())
    }
    #[pyo3(signature=(value,dt,rng=None))]
    fn apply(
        &self,
        py: Python<'_>,
        value: &PyAny,
        dt: f64,
        rng: Option<&PyAny>,
    ) -> PyResult<PyObject> {
        let values: Vec<f64> = py
            .import("numpy")?
            .call_method1("asarray", (value, "float64"))?
            .call_method0("ravel")?
            .extract()?;
        let out = if let Some(rng) = rng {
            let z: Vec<f64> = rng
                .call_method1("standard_normal", (values.len(),))?
                .extract()?;
            self.inner
                .apply_with_draw(&values, &z)
                .map_err(PyValueError::new_err)?
        } else {
            self.inner.apply(&values, dt)
        };
        Ok(py.import("numpy")?.call_method1("asarray", (out,))?.into())
    }
}
