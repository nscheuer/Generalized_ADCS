#[pyclass(name = "FlickerNoise")]
pub struct PyFlickerNoise {
    inner: FlickerNoise,
}

#[pymethods]
impl PyFlickerNoise {
    #[new]
    fn new(amplitude: f64) -> Self {
        Self {
            inner: FlickerNoise::new(amplitude),
        }
    }

    #[getter]
    fn amplitude(&self) -> f64 {
        self.inner.amplitude
    }

    #[setter]
    fn set_amplitude(&mut self, amplitude: f64) {
        self.inner.amplitude = amplitude;
    }
}
use super::flicker_noise::FlickerNoise;
use pyo3::prelude::*;
