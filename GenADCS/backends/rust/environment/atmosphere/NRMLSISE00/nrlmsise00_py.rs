#[pyfunction]
#[pyo3(signature = (day_of_year_utc, seconds_utc, altitude_km, latitude_deg, longitude_deg, f107=150.0, f107a=150.0, ap=4.0, effective_density=true))]
fn nrlmsise00_density_geodetic(
    day_of_year_utc: i32,
    seconds_utc: f64,
    altitude_km: f64,
    latitude_deg: f64,
    longitude_deg: f64,
    f107: f64,
    f107a: f64,
    ap: f64,
    effective_density: bool,
) -> f64 {
    density_geodetic(
        day_of_year_utc,
        seconds_utc,
        altitude_km,
        latitude_deg,
        longitude_deg,
        f107,
        f107a,
        ap,
        effective_density,
    )
}

#[pyfunction]
#[pyo3(signature = (position_inertial_m, epoch_jd, f107=150.0, f107a=150.0, ap=4.0, effective_density=true, ap_array=None))]
fn nrlmsise00_density(
    py: Python<'_>,
    position_inertial_m: &PyAny,
    epoch_jd: f64,
    f107: f64,
    f107a: f64,
    ap: f64,
    effective_density: bool,
    ap_array: Option<[f64; 7]>,
) -> PyResult<f64> {
    let values: Vec<f64> = py
        .import("numpy")?
        .call_method1("asarray", (position_inertial_m,))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    let position: [f64; 3] = values
        .try_into()
        .map_err(|_| PyValueError::new_err("position_inertial_m must be a finite 3-vector"))?;
    if position.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "position_inertial_m must be a finite 3-vector",
        ));
    }
    let ap_history = ap_array.unwrap_or([-1.0; 7]);
    if ap_history.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("ap_array must contain finite values"));
    }
    density_inertial(
        position,
        epoch_jd,
        f107,
        f107a,
        ap,
        ap_history,
        effective_density,
    )
    .map_err(PyValueError::new_err)
}

#[pyclass(name = "NRLMSISE00")]
struct NRLMSISE00;

#[pymethods]
impl NRLMSISE00 {
    #[new]
    fn new() -> Self {
        Self
    }

    #[staticmethod]
    #[pyo3(signature = (position_inertial_m, epoch_jd, f107=150.0, f107a=150.0, ap=4.0, effective_density=true, ap_array=None))]
    fn density(
        py: Python<'_>,
        position_inertial_m: &PyAny,
        epoch_jd: f64,
        f107: f64,
        f107a: f64,
        ap: f64,
        effective_density: bool,
        ap_array: Option<[f64; 7]>,
    ) -> PyResult<f64> {
        nrlmsise00_density(
            py,
            position_inertial_m,
            epoch_jd,
            f107,
            f107a,
            ap,
            effective_density,
            ap_array,
        )
    }

    #[staticmethod]
    #[pyo3(signature = (day_of_year_utc, seconds_utc, altitude_km, latitude_deg, longitude_deg, f107=150.0, f107a=150.0, ap=4.0, effective_density=true))]
    fn density_geodetic(
        day_of_year_utc: i32,
        seconds_utc: f64,
        altitude_km: f64,
        latitude_deg: f64,
        longitude_deg: f64,
        f107: f64,
        f107a: f64,
        ap: f64,
        effective_density: bool,
    ) -> PyResult<f64> {
        Ok(nrlmsise00_density_geodetic(
            day_of_year_utc,
            seconds_utc,
            altitude_km,
            latitude_deg,
            longitude_deg,
            f107,
            f107a,
            ap,
            effective_density,
        ))
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
