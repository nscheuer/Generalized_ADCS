use pyo3::exceptions::PyValueError;
use pyo3::types::PyTuple;

fn to_py_error(error: GravityError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn vector3(py: Python<'_>, value: &PyAny, name: &str) -> PyResult<[f64; 3]> {
    let values: Vec<f64> = py
        .import("numpy")?
        .call_method1("asarray", (value,))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    if values.len() != 3 {
        return Err(PyValueError::new_err(format!("{name} must be a 3-vector")));
    }
    Ok([values[0], values[1], values[2]])
}

#[pyfunction]
fn gravity_acceleration_inertial_m_s2(
    py: Python<'_>,
    position_inertial_m: &PyAny,
    epoch_jd: f64,
    gravitational_parameter_m3_s2: f64,
    reference_radius_m: f64,
    zonal_coefficients: &PyAny,
) -> PyResult<PyObject> {
    let position = vector3(py, position_inertial_m, "position_inertial_m")?;
    let coefficients: Vec<f64> = py
        .import("numpy")?
        .call_method1("asarray", (zonal_coefficients,))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    let _ = epoch_jd;
    let acceleration = native_acceleration(
        position,
        gravitational_parameter_m3_s2,
        reference_radius_m,
        &coefficients,
    )
    .map_err(to_py_error)?;
    Ok(acceleration.to_pyarray(py).into_py(py))
}

#[pyclass(name = "JGravityModel", subclass)]
struct JGravityModelPy {
    inner: JGravityModel,
}

impl JGravityModelPy {
    fn configured(
        degrees: Vec<i32>,
        coefficients: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: JGravityModel::configured(degrees, coefficients).map_err(to_py_error)?,
        })
    }
}

#[pymethods]
impl JGravityModelPy {
    #[new]
    #[pyo3(signature = (degrees=None, coefficients=None))]
    fn new(degrees: Option<Vec<i32>>, coefficients: Option<Vec<f64>>) -> PyResult<Self> {
        let model = JGravityModel::new(degrees, coefficients).map_err(to_py_error)?;
        Ok(Self { inner: model })
    }

    #[getter]
    fn degrees(&self, py: Python<'_>) -> PyObject {
        PyTuple::new(py, self.inner.degrees().iter()).into()
    }

    #[getter]
    fn zonal_coefficients(&self, py: Python<'_>) -> PyObject {
        self.inner.zonal_coefficients().to_pyarray(py).into_py(py)
    }

    #[classattr]
    fn degree() -> Option<i32> {
        None
    }

    #[pyo3(signature = (position_inertial_m, epoch_jd=0.0))]
    fn evaluate(
        &self,
        py: Python<'_>,
        position_inertial_m: &PyAny,
        epoch_jd: f64,
    ) -> PyResult<PyObject> {
        let position = vector3(py, position_inertial_m, "position_inertial_m")?;
        let _ = epoch_jd;
        let acceleration = self.inner.evaluate(position).map_err(to_py_error)?;
        Ok(acceleration.to_pyarray(py).into_py(py))
    }

    #[pyo3(signature = (position_inertial_m, epoch_jd=0.0))]
    fn acceleration(
        &self,
        py: Python<'_>,
        position_inertial_m: &PyAny,
        epoch_jd: f64,
    ) -> PyResult<PyObject> {
        self.evaluate(py, position_inertial_m, epoch_jd)
    }

    #[staticmethod]
    fn acceleration_inertial_m_s2(
        py: Python<'_>,
        position_inertial_m: &PyAny,
        epoch_jd: f64,
        gravitational_parameter_m3_s2: f64,
        reference_radius_m: f64,
        zonal_coefficients: &PyAny,
    ) -> PyResult<PyObject> {
        gravity_acceleration_inertial_m_s2(
            py,
            position_inertial_m,
            epoch_jd,
            gravitational_parameter_m3_s2,
            reference_radius_m,
            zonal_coefficients,
        )
    }
}
