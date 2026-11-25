#[pyclass(name = "GaussMarkovNoise")]
pub struct PyGaussMarkovNoise {
    inner: GaussMarkovNoise,
}

#[pymethods]
impl PyGaussMarkovNoise {
    #[new]
    fn new(standard_deviation: f64, correlation_time: f64) -> Self {
        Self {
            inner: GaussMarkovNoise::new(standard_deviation, correlation_time),
        }
    }

    #[getter]
    fn standard_deviation(&self) -> f64 {
        self.inner.standard_deviation
    }

    #[setter]
    fn set_standard_deviation(&mut self, value: f64) {
        self.inner.standard_deviation = value;
    }

    #[getter]
    fn correlation_time(&self) -> f64 {
        self.inner.correlation_time
    }

    #[setter]
    fn set_correlation_time(&mut self, value: f64) {
        self.inner.correlation_time = value;
    }
}
use super::gauss_markov_noise::GaussMarkovNoise;
use pyo3::prelude::*;
