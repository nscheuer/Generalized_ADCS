use super::{
    gyroscope::Gyroscope as Native,
    sensor_py::{array, matrix3, vector},
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(name = "Gyroscope")]
pub struct PyGyroscope {
    inner: Native,
}
#[pymethods]
impl PyGyroscope {
    #[new]
    fn new(
        py: Python<'_>,
        position: &PyAny,
        orientation: &PyAny,
        misalignment: &PyAny,
        saturation_value: f64,
        noise: &PyAny,
        bias: &PyAny,
    ) -> PyResult<Self> {
        let p = vector::<3>(py, position, "position")?;
        let o = vector::<4>(py, orientation, "orientation")?;
        let m = vector::<4>(py, misalignment, "misalignment")?;
        let b = vector::<3>(py, bias, "bias")?;
        let n = matrix3(py, noise)?;
        Ok(Self {
            inner: Native::new(&p, &o, &m, saturation_value, &n, &b)
                .map_err(PyValueError::new_err)?,
        })
    }
    #[getter]
    fn position(&self, py: Python<'_>) -> PyResult<PyObject> {
        array(py, self.inner.sensor.position.to_vec())
    }
    #[setter]
    fn set_position(&mut self, py: Python<'_>, v: &PyAny) -> PyResult<()> {
        self.inner.sensor.position = vector(py, v, "position")?;
        Ok(())
    }
    #[getter]
    fn orientation(&self, py: Python<'_>) -> PyResult<PyObject> {
        array(py, self.inner.sensor.orientation.to_vec())
    }
    #[setter]
    fn set_orientation(&mut self, py: Python<'_>, v: &PyAny) -> PyResult<()> {
        self.inner.sensor.orientation = vector(py, v, "orientation")?;
        Ok(())
    }
    #[getter]
    fn misalignment(&self, py: Python<'_>) -> PyResult<PyObject> {
        array(py, self.inner.sensor.misalignment.to_vec())
    }
    #[setter]
    fn set_misalignment(&mut self, py: Python<'_>, v: &PyAny) -> PyResult<()> {
        self.inner.sensor.misalignment = vector(py, v, "misalignment")?;
        Ok(())
    }
    #[getter]
    fn saturation_value(&self) -> f64 {
        self.inner.sensor.saturation_value
    }
    #[setter]
    fn set_saturation_value(&mut self, v: f64) {
        self.inner.sensor.saturation_value = v;
    }
    #[getter]
    fn noise(&self, py: Python<'_>) -> PyResult<PyObject> {
        array(py, self.inner.sensor.noise.to_vec())
    }
    #[setter]
    fn set_noise(&mut self, py: Python<'_>, v: &PyAny) -> PyResult<()> {
        let n = matrix3(py, v)?;
        self.inner.sensor.noise = std::array::from_fn(|i| std::array::from_fn(|j| n[i][j]));
        Ok(())
    }
    #[getter]
    fn noise_cholesky(&self, py: Python<'_>) -> PyResult<PyObject> {
        array(py, self.inner.noise_cholesky.to_vec())
    }
    #[setter]
    fn set_noise_cholesky(&mut self, py: Python<'_>, v: &PyAny) -> PyResult<()> {
        let n = matrix3(py, v)?;
        self.inner.noise_cholesky = std::array::from_fn(|i| std::array::from_fn(|j| n[i][j]));
        Ok(())
    }
    #[getter]
    fn bias(&self, py: Python<'_>) -> PyResult<PyObject> {
        array(py, self.inner.sensor.bias.to_vec())
    }
    #[setter]
    fn set_bias(&mut self, py: Python<'_>, v: &PyAny) -> PyResult<()> {
        self.inner.sensor.bias = vector(py, v, "bias")?;
        Ok(())
    }
    #[getter]
    fn enabled(&self) -> bool {
        self.inner.sensor.enabled
    }
    #[setter]
    fn set_enabled(&mut self, v: bool) {
        self.inner.sensor.enabled = v;
    }
    fn noiseless_biasfree_measurement(&self, py: Python<'_>, w: &PyAny) -> PyResult<PyObject> {
        array(
            py,
            self.inner
                .noiseless_biasfree_measurement(vector(py, w, "angular velocity")?)
                .to_vec(),
        )
    }
    fn noiseless_biased_measurement(&self, py: Python<'_>, w: &PyAny) -> PyResult<PyObject> {
        array(
            py,
            self.inner
                .noiseless_biased_measurement(vector(py, w, "angular velocity")?)
                .to_vec(),
        )
    }
    fn noisy_biasfree_measurement(&self, py: Python<'_>, w: &PyAny) -> PyResult<PyObject> {
        array(
            py,
            self.inner
                .noisy_biasfree_measurement(vector(py, w, "angular velocity")?)
                .to_vec(),
        )
    }
    fn noisy_biased_measurement(&self, py: Python<'_>, w: &PyAny) -> PyResult<PyObject> {
        array(
            py,
            self.inner
                .noisy_biased_measurement(vector(py, w, "angular velocity")?)
                .to_vec(),
        )
    }
    fn jacobian_angular_velocity(&self, py: Python<'_>) -> PyResult<PyObject> {
        array(py, self.inner.jacobian_angular_velocity().to_vec())
    }
    fn jacobian_bias(&self, py: Python<'_>) -> PyResult<PyObject> {
        array(py, self.inner.jacobian_bias().to_vec())
    }
    fn availability(&self) -> bool {
        self.inner.availability()
    }
}
