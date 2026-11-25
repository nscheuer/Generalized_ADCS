use super::{
    magnetometer::Magnetometer as Native,
    sensor_py::{array, matrix3, vector},
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(name = "Magnetometer")]
pub struct PyMagnetometer {
    inner: Native,
}
#[pymethods]
impl PyMagnetometer {
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
    fn noiseless_biasfree_measurement(
        &self,
        py: Python<'_>,
        field: &PyAny,
        q: &PyAny,
    ) -> PyResult<PyObject> {
        array(
            py,
            self.inner
                .noiseless_biasfree_measurement(
                    vector(py, field, "field")?,
                    vector(py, q, "attitude")?,
                )
                .to_vec(),
        )
    }
    fn noiseless_biased_measurement(
        &self,
        py: Python<'_>,
        field: &PyAny,
        q: &PyAny,
    ) -> PyResult<PyObject> {
        array(
            py,
            self.inner
                .noiseless_biased_measurement(
                    vector(py, field, "field")?,
                    vector(py, q, "attitude")?,
                )
                .to_vec(),
        )
    }
    fn noisy_biasfree_measurement(
        &self,
        py: Python<'_>,
        field: &PyAny,
        q: &PyAny,
    ) -> PyResult<PyObject> {
        array(
            py,
            self.inner
                .noisy_biasfree_measurement(vector(py, field, "field")?, vector(py, q, "attitude")?)
                .to_vec(),
        )
    }
    fn noisy_biased_measurement(
        &self,
        py: Python<'_>,
        field: &PyAny,
        q: &PyAny,
    ) -> PyResult<PyObject> {
        array(
            py,
            self.inner
                .noisy_biased_measurement(vector(py, field, "field")?, vector(py, q, "attitude")?)
                .to_vec(),
        )
    }
    fn jacobian_bias(&self, py: Python<'_>) -> PyResult<PyObject> {
        array(py, self.inner.jacobian_bias().to_vec())
    }
    fn jacobian_relative_quaternion(
        &self,
        py: Python<'_>,
        field: &PyAny,
        q: &PyAny,
        d: &PyAny,
    ) -> PyResult<PyObject> {
        array(
            py,
            self.inner
                .jacobian_relative_quaternion(
                    vector(py, field, "field")?,
                    vector(py, q, "attitude")?,
                    vector(py, d, "delta quaternion")?,
                )
                .to_vec(),
        )
    }
    fn jacobian_relative_mrp(
        &self,
        py: Python<'_>,
        field: &PyAny,
        q: &PyAny,
        d: &PyAny,
    ) -> PyResult<PyObject> {
        array(
            py,
            self.inner
                .jacobian_relative_mrp(
                    vector(py, field, "field")?,
                    vector(py, q, "attitude")?,
                    vector(py, d, "delta MRP")?,
                )
                .to_vec(),
        )
    }
    fn jacobian_relative_angle_vector(
        &self,
        py: Python<'_>,
        field: &PyAny,
        q: &PyAny,
        d: &PyAny,
    ) -> PyResult<PyObject> {
        array(
            py,
            self.inner
                .jacobian_relative_angle_vector(
                    vector(py, field, "field")?,
                    vector(py, q, "attitude")?,
                    vector(py, d, "delta angle")?,
                )
                .to_vec(),
        )
    }
    fn availability(&self) -> bool {
        self.inner.availability()
    }
    fn fisher_information_matrix(
        &self,
        py: Python<'_>,
        field: &PyAny,
        q: &PyAny,
        cov: &PyAny,
    ) -> PyResult<PyObject> {
        let cov = matrix3(py, cov)?;
        let cov = std::array::from_fn(|i| std::array::from_fn(|j| cov[i][j]));
        array(
            py,
            self.inner
                .fisher_information_matrix(
                    vector(py, field, "field")?,
                    vector(py, q, "attitude")?,
                    cov,
                )
                .to_vec(),
        )
    }
}
