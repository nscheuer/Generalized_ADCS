use super::fine_earth_sensor::FineEarthSensor as Native;
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
fn mat2(py: Python<'_>, x: &PyAny) -> PyResult<[[f64; 2]; 2]> {
    let a = py
        .import("numpy")?
        .call_method1("asarray", (x, "float64"))?;
    let shape: Vec<usize> = a.getattr("shape")?.extract()?;
    if shape != [2, 2] {
        return Err(PyValueError::new_err("noise must have shape (2, 2)"));
    }
    let v: Vec<Vec<f64>> = a.call_method0("tolist")?.extract()?;
    Ok(std::array::from_fn(|i| std::array::from_fn(|j| v[i][j])))
}
fn out(py: Python<'_>, x: impl IntoPy<PyObject>) -> PyResult<PyObject> {
    Ok(py.import("numpy")?.call_method1("asarray", (x,))?.into())
}

#[pyclass(name = "FineEarthSensor")]
pub struct PyFineEarthSensor {
    inner: Native,
}
#[pymethods]
impl PyFineEarthSensor {
    #[new]
    fn new(
        py: Python<'_>,
        position: &PyAny,
        orientation: &PyAny,
        misalignment: &PyAny,
        noise: &PyAny,
        bias: &PyAny,
        fov_half_angles: &PyAny,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Native::new(
                arr(py, position, "position")?,
                arr(py, orientation, "orientation")?,
                arr(py, misalignment, "misalignment")?,
                mat2(py, noise)?,
                arr(py, bias, "bias")?,
                arr(py, fov_half_angles, "fov_half_angles")?,
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
    fn noise(&self, py: Python<'_>) -> PyResult<PyObject> {
        out(py, self.inner.noise)
    }
    #[getter]
    fn noise_cholesky(&self, py: Python<'_>) -> PyResult<PyObject> {
        out(py, self.inner.noise_cholesky)
    }
    #[getter]
    fn bias(&self, py: Python<'_>) -> PyResult<PyObject> {
        out(py, self.inner.bias)
    }
    #[getter]
    fn fov_half_angles(&self, py: Python<'_>) -> PyResult<PyObject> {
        out(py, self.inner.fov_half_angles)
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
        earth: &PyAny,
        attitude: &PyAny,
    ) -> PyResult<PyObject> {
        out(
            py,
            self.inner.noiseless_biasfree_measurement(
                arr(py, earth, "earth_inertial")?,
                arr(py, attitude, "attitude")?,
            ),
        )
    }
    fn noiseless_biased_measurement(
        &self,
        py: Python<'_>,
        earth: &PyAny,
        attitude: &PyAny,
    ) -> PyResult<PyObject> {
        out(
            py,
            self.inner.noiseless_biased_measurement(
                arr(py, earth, "earth_inertial")?,
                arr(py, attitude, "attitude")?,
            ),
        )
    }
    fn noisy_biasfree_measurement(
        &self,
        py: Python<'_>,
        earth: &PyAny,
        attitude: &PyAny,
    ) -> PyResult<PyObject> {
        out(
            py,
            self.inner.noisy_biasfree_measurement(
                arr(py, earth, "earth_inertial")?,
                arr(py, attitude, "attitude")?,
            ),
        )
    }
    fn noisy_biased_measurement(
        &self,
        py: Python<'_>,
        earth: &PyAny,
        attitude: &PyAny,
    ) -> PyResult<PyObject> {
        out(
            py,
            self.inner.noisy_biased_measurement(
                arr(py, earth, "earth_inertial")?,
                arr(py, attitude, "attitude")?,
            ),
        )
    }
    fn jacobian_bias(&self, py: Python<'_>) -> PyResult<PyObject> {
        out(py, self.inner.jacobian_bias())
    }
    fn jacobian_relative_quaternion(
        &self,
        py: Python<'_>,
        earth: &PyAny,
        attitude: &PyAny,
        delta: &PyAny,
    ) -> PyResult<PyObject> {
        out(
            py,
            self.inner.jacobian_relative_quaternion(
                arr(py, earth, "earth_inertial")?,
                arr(py, attitude, "attitude")?,
                arr(py, delta, "delta_quaternion")?,
            ),
        )
    }
    fn jacobian_relative_mrp(
        &self,
        py: Python<'_>,
        earth: &PyAny,
        attitude: &PyAny,
        delta: &PyAny,
    ) -> PyResult<PyObject> {
        out(
            py,
            self.inner.jacobian_relative_mrp(
                arr(py, earth, "earth_inertial")?,
                arr(py, attitude, "attitude")?,
                arr(py, delta, "delta_mrp")?,
            ),
        )
    }
    fn jacobian_relative_angle_vector(
        &self,
        py: Python<'_>,
        earth: &PyAny,
        attitude: &PyAny,
        delta: &PyAny,
    ) -> PyResult<PyObject> {
        out(
            py,
            self.inner.jacobian_relative_angle_vector(
                arr(py, earth, "earth_inertial")?,
                arr(py, attitude, "attitude")?,
                arr(py, delta, "delta_angle")?,
            ),
        )
    }
    #[pyo3(signature=(earth_inertial,attitude,earth_visible=true))]
    fn availability(
        &self,
        py: Python<'_>,
        earth_inertial: &PyAny,
        attitude: &PyAny,
        earth_visible: bool,
    ) -> PyResult<bool> {
        Ok(self.inner.availability(
            arr(py, earth_inertial, "earth_inertial")?,
            arr(py, attitude, "attitude")?,
            earth_visible,
        ))
    }
    #[pyo3(signature=(earth_inertial,attitude,earth_visible=true))]
    fn fisher_information_matrix(
        &self,
        py: Python<'_>,
        earth_inertial: &PyAny,
        attitude: &PyAny,
        earth_visible: bool,
    ) -> PyResult<PyObject> {
        out(
            py,
            self.inner.fisher_information_matrix(
                arr(py, earth_inertial, "earth_inertial")?,
                arr(py, attitude, "attitude")?,
                earth_visible,
            ),
        )
    }
}
