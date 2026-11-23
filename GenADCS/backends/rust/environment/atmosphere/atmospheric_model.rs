#[pyfunction]
fn atmospheric_density(py: Python<'_>, position_inertial_m: &PyAny, epoch_jd: f64) -> PyResult<f64> {
    let _ = (py, position_inertial_m, epoch_jd);
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "Choose an atmospheric density model",
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
    let _ = epoch_jd;
    let position: Vec<f64> = py
        .import("numpy")?
        .call_method1("asarray", (position_inertial_m,))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    if position.len() != 3 || position.iter().any(|value| !value.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "position_inertial_m must be a finite 3-vector",
        ));
    }
    let p = [position[0], position[1], position[2]];
    let omega = angular_velocity_inertial_rad_s.unwrap_or([0.0, 0.0, 7.2921150e-5]);
    let velocity = [
        omega[1] * p[2] - omega[2] * p[1],
        omega[2] * p[0] - omega[0] * p[2],
        omega[0] * p[1] - omega[1] * p[0],
    ];
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
