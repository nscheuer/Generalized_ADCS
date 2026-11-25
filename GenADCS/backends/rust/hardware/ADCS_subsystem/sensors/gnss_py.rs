use super::GNSS;
use pyo3::{exceptions::PyValueError, prelude::*};

fn floats(py: Python<'_>, x: &PyAny) -> PyResult<Vec<f64>> {
    py.import("numpy")?
        .call_method1("asarray", (x, "float64"))?
        .call_method0("ravel")?
        .extract()
}
fn arr<const N: usize>(py: Python<'_>, x: &PyAny, label: &str) -> PyResult<[f64; N]> {
    floats(py, x)?
        .try_into()
        .map_err(|_| PyValueError::new_err(format!("{label} must have {N} components")))
}
fn matrix(py: Python<'_>, x: impl IntoPy<PyObject>) -> PyResult<PyObject> {
    Ok(py.import("numpy")?.call_method1("asarray", (x,))?.into())
}
#[pyclass(name = "GNSS")]
pub struct PyGNSS {
    inner: GNSS,
}
#[pymethods]
impl PyGNSS {
    #[new]
    fn new(py: Python<'_>, noise: &PyAny, bias: &PyAny) -> PyResult<Self> {
        let noise: Vec<Vec<f64>> = py
            .import("numpy")?
            .call_method1("asarray", (noise, "float64"))?
            .call_method0("tolist")?
            .extract()
            .map_err(|_| PyValueError::new_err("noise must be a 6x6 covariance"))?;
        let bias = floats(py, bias)?;
        let inner = GNSS::new(&noise, &bias).map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }
    #[getter]
    fn noise(&self, py: Python<'_>) -> PyResult<PyObject> {
        matrix(py, self.inner.noise.clone())
    }
    #[getter]
    fn noise_cholesky(&self, py: Python<'_>) -> PyResult<PyObject> {
        matrix(py, self.inner.noise_cholesky.clone())
    }
    #[getter]
    fn bias(&self, py: Python<'_>) -> PyResult<PyObject> {
        matrix(py, self.inner.bias.to_vec())
    }
    #[getter]
    fn enabled(&self) -> bool {
        self.inner.enabled
    }
    #[setter]
    fn set_enabled(&mut self, value: bool) {
        self.inner.enabled = value
    }
    fn availability(&self) -> bool {
        self.inner.availability()
    }
    fn noiseless_biasfree_measurement(
        &self,
        py: Python<'_>,
        position: &PyAny,
        velocity: &PyAny,
    ) -> PyResult<PyObject> {
        matrix(
            py,
            self.inner
                .noiseless_biasfree_measurement(
                    arr(py, position, "position")?,
                    arr(py, velocity, "velocity")?,
                )
                .to_vec(),
        )
    }
    fn noiseless_biased_measurement(
        &self,
        py: Python<'_>,
        position: &PyAny,
        velocity: &PyAny,
    ) -> PyResult<PyObject> {
        matrix(
            py,
            self.inner
                .noiseless_biased_measurement(
                    arr(py, position, "position")?,
                    arr(py, velocity, "velocity")?,
                )
                .to_vec(),
        )
    }
    fn noisy_biasfree_measurement(
        &self,
        py: Python<'_>,
        position: &PyAny,
        velocity: &PyAny,
    ) -> PyResult<PyObject> {
        matrix(
            py,
            self.inner
                .noisy_biasfree_measurement(
                    arr(py, position, "position")?,
                    arr(py, velocity, "velocity")?,
                )
                .to_vec(),
        )
    }
    fn noisy_biased_measurement(
        &self,
        py: Python<'_>,
        position: &PyAny,
        velocity: &PyAny,
    ) -> PyResult<PyObject> {
        matrix(
            py,
            self.inner
                .noisy_biased_measurement(
                    arr(py, position, "position")?,
                    arr(py, velocity, "velocity")?,
                )
                .to_vec(),
        )
    }
    fn jacobian_relative_quaternion(&self, py: Python<'_>) -> PyResult<PyObject> {
        matrix(py, self.inner.jacobian_relative_quaternion().to_vec())
    }
    fn jacobian_relative_mrp(&self, py: Python<'_>) -> PyResult<PyObject> {
        matrix(py, self.inner.jacobian_relative_mrp().to_vec())
    }
    fn jacobian_relative_angle_vector(&self, py: Python<'_>) -> PyResult<PyObject> {
        matrix(py, self.inner.jacobian_relative_angle_vector().to_vec())
    }
    fn jacobian_position(&self, py: Python<'_>) -> PyResult<PyObject> {
        matrix(py, self.inner.jacobian_position().to_vec())
    }
    fn jacobian_velocity(&self, py: Python<'_>) -> PyResult<PyObject> {
        matrix(py, self.inner.jacobian_velocity().to_vec())
    }
    fn jacobian_bias(&self, py: Python<'_>) -> PyResult<PyObject> {
        matrix(py, self.inner.jacobian_bias().to_vec())
    }
}
