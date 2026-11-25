use super::AutonomousStarTracker;
use pyo3::{exceptions::PyValueError, prelude::*};

fn floats(py: Python<'_>, x: &PyAny) -> PyResult<Vec<f64>> {
    py.import("numpy")?
        .call_method1("asarray", (x, "float64"))?
        .call_method0("ravel")?
        .extract()
}
fn arr<const N: usize>(py: Python<'_>, x: &PyAny, name: &str) -> PyResult<[f64; N]> {
    floats(py, x)?
        .try_into()
        .map_err(|_| PyValueError::new_err(format!("{name} must have {N} values")))
}
fn result(py: Python<'_>, x: impl IntoPy<PyObject>) -> PyResult<PyObject> {
    Ok(py.import("numpy")?.call_method1("asarray", (x,))?.into())
}
#[pyclass(name = "AutonomousStarTracker")]
pub struct PyAutonomousStarTracker {
    inner: AutonomousStarTracker,
}
#[pymethods]
impl PyAutonomousStarTracker {
    #[new]
    #[pyo3(signature=(position,orientation,misalignment,saturation_value,noise,bias,fov_half_angle=None))]
    fn new(
        py: Python<'_>,
        position: &PyAny,
        orientation: &PyAny,
        misalignment: &PyAny,
        saturation_value: f64,
        noise: &PyAny,
        bias: &PyAny,
        fov_half_angle: Option<f64>,
    ) -> PyResult<Self> {
        let noise: Vec<Vec<f64>> = py
            .import("numpy")?
            .call_method1("asarray", (noise, "float64"))?
            .call_method0("tolist")?
            .extract()?;
        let inner = AutonomousStarTracker::new(
            &arr::<3>(py, position, "position")?,
            &arr::<4>(py, orientation, "orientation")?,
            &arr::<4>(py, misalignment, "misalignment")?,
            saturation_value,
            &noise,
            &arr::<3>(py, bias, "bias")?,
            fov_half_angle.unwrap_or(std::f64::consts::PI),
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }
    #[getter]
    fn position(&self, py: Python<'_>) -> PyResult<PyObject> {
        result(py, self.inner.sensor.position.to_vec())
    }
    #[getter]
    fn orientation(&self, py: Python<'_>) -> PyResult<PyObject> {
        result(py, self.inner.sensor.orientation.to_vec())
    }
    #[getter]
    fn misalignment(&self, py: Python<'_>) -> PyResult<PyObject> {
        result(py, self.inner.sensor.misalignment.to_vec())
    }
    #[getter]
    fn noise(&self, py: Python<'_>) -> PyResult<PyObject> {
        result(py, self.inner.sensor.noise.to_vec())
    }
    #[getter]
    fn noise_cholesky(&self, py: Python<'_>) -> PyResult<PyObject> {
        result(py, self.inner.noise_cholesky.to_vec())
    }
    #[getter]
    fn bias(&self, py: Python<'_>) -> PyResult<PyObject> {
        result(py, self.inner.sensor.bias.to_vec())
    }
    #[getter]
    fn saturation_value(&self) -> f64 {
        self.inner.sensor.saturation_value
    }
    #[getter]
    fn fov_half_angle(&self) -> f64 {
        self.inner.fov_half_angle
    }
    #[setter]
    fn set_fov_half_angle(&mut self, x: f64) -> PyResult<()> {
        if !x.is_finite() || x <= 0. || x > std::f64::consts::PI {
            return Err(PyValueError::new_err("fov_half_angle must be in (0, pi]"));
        }
        self.inner.fov_half_angle = x;
        Ok(())
    }
    #[getter]
    fn enabled(&self) -> bool {
        self.inner.sensor.enabled
    }
    #[setter]
    fn set_enabled(&mut self, x: bool) {
        self.inner.sensor.enabled = x
    }
    fn noiseless_biasfree_measurement(
        &self,
        py: Python<'_>,
        attitude: &PyAny,
    ) -> PyResult<PyObject> {
        result(
            py,
            self.inner
                .noiseless_biasfree_measurement(arr(py, attitude, "attitude")?),
        )
    }
    fn noiseless_biased_measurement(&self, py: Python<'_>, attitude: &PyAny) -> PyResult<PyObject> {
        result(
            py,
            self.inner
                .noiseless_biased_measurement(arr(py, attitude, "attitude")?),
        )
    }
    fn noisy_biasfree_measurement(&self, py: Python<'_>, attitude: &PyAny) -> PyResult<PyObject> {
        result(
            py,
            self.inner
                .noisy_biasfree_measurement(arr(py, attitude, "attitude")?),
        )
    }
    fn noisy_biased_measurement(&self, py: Python<'_>, attitude: &PyAny) -> PyResult<PyObject> {
        result(
            py,
            self.inner
                .noisy_biased_measurement(arr(py, attitude, "attitude")?),
        )
    }
    fn residual(&self, py: Python<'_>, measured: &PyAny, estimated: &PyAny) -> PyResult<PyObject> {
        result(
            py,
            self.inner.residual(
                arr(py, measured, "measured attitude")?,
                arr(py, estimated, "estimated attitude")?,
            ),
        )
    }
    fn jacobian_relative_quaternion(&self, py: Python<'_>) -> PyResult<PyObject> {
        result(py, self.inner.jacobian_relative_quaternion().to_vec())
    }
    fn jacobian_relative_mrp(&self, py: Python<'_>, d: &PyAny) -> PyResult<PyObject> {
        result(
            py,
            self.inner
                .jacobian_relative_mrp(arr(py, d, "delta MRP")?)
                .to_vec(),
        )
    }
    fn jacobian_relative_angle_vector(&self, py: Python<'_>, d: &PyAny) -> PyResult<PyObject> {
        result(
            py,
            self.inner
                .jacobian_relative_angle_vector(arr(py, d, "delta angle")?)
                .to_vec(),
        )
    }
    #[pyo3(signature=(attitude=None,earth_inertial=None,sun_inertial=None))]
    fn availability(
        &self,
        py: Python<'_>,
        attitude: Option<&PyAny>,
        earth_inertial: Option<&PyAny>,
        sun_inertial: Option<&PyAny>,
    ) -> PyResult<bool> {
        Ok(self.inner.availability(
            attitude.map(|x| arr(py, x, "attitude")).transpose()?,
            earth_inertial
                .map(|x| arr(py, x, "earth direction"))
                .transpose()?,
            sun_inertial
                .map(|x| arr(py, x, "sun direction"))
                .transpose()?,
        ))
    }
    fn fisher_information_matrix(&self, py: Python<'_>) -> PyResult<PyObject> {
        result(py, self.inner.fisher_information_matrix().to_vec())
    }
}
