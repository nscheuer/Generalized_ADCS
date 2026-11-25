use super::coarse_sun_sensor::CoarseSunSensor as Native;
use pyo3::{exceptions::PyValueError, prelude::*};

fn arr<const N: usize>(py: Python<'_>, x: &PyAny, label: &str) -> PyResult<[f64; N]> {
    let a = py
        .import("numpy")?
        .call_method1("asarray", (x, "float64"))?;
    let shape: Vec<usize> = a.getattr("shape")?.extract()?;
    if shape != [N] {
        return Err(PyValueError::new_err(format!(
            "{label} must have shape ({N},)"
        )));
    }
    let v: Vec<f64> = a.call_method0("tolist")?.extract()?;
    v.try_into()
        .map_err(|_| PyValueError::new_err(format!("{label} must have {N} values")))
}
fn out(py: Python<'_>, x: impl IntoPy<PyObject>) -> PyResult<PyObject> {
    Ok(py.import("numpy")?.call_method1("asarray", (x,))?.into())
}

#[pyclass(name = "CoarseSunSensor")]
pub struct PyCoarseSunSensor {
    inner: Native,
}
#[pymethods]
impl PyCoarseSunSensor {
    #[new]
    fn new(
        py: Python<'_>,
        position: &PyAny,
        orientation: &PyAny,
        misalignment: &PyAny,
        normal: &PyAny,
        sensitivity: f64,
        saturation_value: f64,
        noise_variance: f64,
        bias: f64,
        fov_half_angle: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Native::new(
                arr(py, position, "position")?,
                arr(py, orientation, "orientation")?,
                arr(py, misalignment, "misalignment")?,
                arr(py, normal, "normal")?,
                sensitivity,
                saturation_value,
                noise_variance,
                bias,
                fov_half_angle,
            )
            .map_err(PyValueError::new_err)?,
        })
    }
    #[getter]
    fn position(&self, py: Python<'_>) -> PyResult<PyObject> {
        out(py, self.inner.position)
    }
    #[getter]
    fn orientation(&self, py: Python<'_>) -> PyResult<PyObject> {
        out(py, self.inner.orientation)
    }
    #[getter]
    fn misalignment(&self, py: Python<'_>) -> PyResult<PyObject> {
        out(py, self.inner.misalignment)
    }
    #[getter]
    fn normal(&self, py: Python<'_>) -> PyResult<PyObject> {
        out(py, self.inner.normal)
    }
    #[getter]
    fn sensitivity(&self) -> f64 {
        self.inner.sensitivity
    }
    #[getter]
    fn saturation_value(&self) -> f64 {
        self.inner.saturation_value
    }
    #[getter]
    fn noise_variance(&self) -> f64 {
        self.inner.noise_variance
    }
    #[getter]
    fn noise_std(&self) -> f64 {
        self.inner.noise_std
    }
    #[getter]
    fn bias(&self) -> f64 {
        self.inner.bias
    }
    #[getter]
    fn fov_half_angle(&self) -> f64 {
        self.inner.fov_half_angle
    }
    #[getter]
    fn enabled(&self) -> bool {
        self.inner.enabled
    }
    #[setter]
    fn set_enabled(&mut self, x: bool) {
        self.inner.enabled = x
    }
    fn noiseless_biasfree_measurement(
        &self,
        py: Python<'_>,
        sun: &PyAny,
        attitude: &PyAny,
    ) -> PyResult<f64> {
        Ok(self.inner.noiseless_biasfree_measurement(
            arr(py, sun, "sun_inertial")?,
            arr(py, attitude, "attitude")?,
        ))
    }
    fn noiseless_biased_measurement(
        &self,
        py: Python<'_>,
        sun: &PyAny,
        attitude: &PyAny,
    ) -> PyResult<f64> {
        Ok(self.inner.noiseless_biased_measurement(
            arr(py, sun, "sun_inertial")?,
            arr(py, attitude, "attitude")?,
        ))
    }
    fn noisy_biasfree_measurement(
        &self,
        py: Python<'_>,
        sun: &PyAny,
        attitude: &PyAny,
    ) -> PyResult<f64> {
        Ok(self.inner.noisy_biasfree_measurement(
            arr(py, sun, "sun_inertial")?,
            arr(py, attitude, "attitude")?,
        ))
    }
    fn noisy_biased_measurement(
        &self,
        py: Python<'_>,
        sun: &PyAny,
        attitude: &PyAny,
    ) -> PyResult<f64> {
        Ok(self.inner.noisy_biased_measurement(
            arr(py, sun, "sun_inertial")?,
            arr(py, attitude, "attitude")?,
        ))
    }
    fn jacobian_bias(&self) -> f64 {
        self.inner.jacobian_bias()
    }
    fn jacobian_relative_quaternion(
        &self,
        py: Python<'_>,
        sun: &PyAny,
        attitude: &PyAny,
        delta: &PyAny,
    ) -> PyResult<PyObject> {
        out(
            py,
            self.inner.jacobian_relative_quaternion(
                arr(py, sun, "sun_inertial")?,
                arr(py, attitude, "attitude")?,
                arr(py, delta, "delta_quaternion")?,
            ),
        )
    }
    fn jacobian_relative_mrp(
        &self,
        py: Python<'_>,
        sun: &PyAny,
        attitude: &PyAny,
        delta: &PyAny,
    ) -> PyResult<PyObject> {
        out(
            py,
            self.inner.jacobian_relative_mrp(
                arr(py, sun, "sun_inertial")?,
                arr(py, attitude, "attitude")?,
                arr(py, delta, "delta_mrp")?,
            ),
        )
    }
    fn jacobian_relative_angle_vector(
        &self,
        py: Python<'_>,
        sun: &PyAny,
        attitude: &PyAny,
        delta: &PyAny,
    ) -> PyResult<PyObject> {
        out(
            py,
            self.inner.jacobian_relative_angle_vector(
                arr(py, sun, "sun_inertial")?,
                arr(py, attitude, "attitude")?,
                arr(py, delta, "delta_angle")?,
            ),
        )
    }
    #[pyo3(signature=(sun_inertial,attitude,sun_visible=true))]
    fn availability(
        &self,
        py: Python<'_>,
        sun_inertial: &PyAny,
        attitude: &PyAny,
        sun_visible: bool,
    ) -> PyResult<bool> {
        Ok(self.inner.availability(
            arr(py, sun_inertial, "sun_inertial")?,
            arr(py, attitude, "attitude")?,
            sun_visible,
        ))
    }
    #[pyo3(signature=(sun_inertial,attitude,sun_visible=true))]
    fn fisher_information_matrix(
        &self,
        py: Python<'_>,
        sun_inertial: &PyAny,
        attitude: &PyAny,
        sun_visible: bool,
    ) -> PyResult<PyObject> {
        out(
            py,
            self.inner.fisher_information_matrix(
                arr(py, sun_inertial, "sun_inertial")?,
                arr(py, attitude, "attitude")?,
                sun_visible,
            ),
        )
    }
}
