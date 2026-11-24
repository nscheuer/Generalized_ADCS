#[pyfunction]
fn atmospheric_density(
    _py: Python<'_>,
    position_inertial_m: &PyAny,
    epoch_jd: f64,
) -> PyResult<f64> {
    let _ = (position_inertial_m, epoch_jd);
    Err(PyNotImplementedError::new_err(
        generic_density([0.0; 3], 0.0).unwrap_err(),
    ))
}

#[pyfunction]
#[pyo3(signature = (position_inertial_m, epoch_jd, angular_velocity_inertial_rad_s=None))]
fn co_rotating_atmosphere_velocity(
    py: Python<'_>,
    position_inertial_m: &PyAny,
    epoch_jd: f64,
    angular_velocity_inertial_rad_s: Option<[f64; 3]>,
) -> PyResult<PyObject> {
    let values: Vec<f64> = py
        .import("numpy")?
        .call_method1("asarray", (position_inertial_m,))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    let position: [f64; 3] = values
        .try_into()
        .map_err(|_| PyValueError::new_err("position_inertial_m must be a finite 3-vector"))?;
    let velocity = co_rotating_velocity(position, epoch_jd, angular_velocity_inertial_rad_s)
        .map_err(PyValueError::new_err)?;
    Ok(py
        .import("numpy")?
        .call_method1("asarray", (velocity,))?
        .into())
}

#[pyclass(name = "AtmosphericModel")]
struct AtmosphericModel;

#[pymethods]
impl AtmosphericModel {
    #[staticmethod]
    fn density(py: Python<'_>, position_inertial_m: &PyAny, epoch_jd: f64) -> PyResult<f64> {
        atmospheric_density(py, position_inertial_m, epoch_jd)
    }

    #[staticmethod]
    #[pyo3(signature = (position_inertial_m, epoch_jd, angular_velocity_inertial_rad_s=None))]
    fn velocity(
        py: Python<'_>,
        position_inertial_m: &PyAny,
        epoch_jd: f64,
        angular_velocity_inertial_rad_s: Option<[f64; 3]>,
    ) -> PyResult<PyObject> {
        co_rotating_atmosphere_velocity(
            py,
            position_inertial_m,
            epoch_jd,
            angular_velocity_inertial_rad_s,
        )
    }
}
