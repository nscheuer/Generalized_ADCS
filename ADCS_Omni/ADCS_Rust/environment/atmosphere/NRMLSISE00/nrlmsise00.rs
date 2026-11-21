use pyo3::exceptions::PyValueError;
fn density_geodetic_impl(
    day_of_year: i32,
    seconds_utc: f64,
    altitude_km: f64,
    latitude_deg: f64,
    longitude_deg: f64,
    f107: f64,
    f107a: f64,
    ap: f64,
    ap_history: [f64; 7],
    effective_density: bool,
) -> f64 {
    evaluate_geodetic(
        day_of_year,
        seconds_utc,
        altitude_km,
        latitude_deg,
        longitude_deg,
        f107,
        f107a,
        ap,
        ap_history,
        effective_density,
    )
}

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
    density_geodetic_impl(
        day_of_year_utc,
        seconds_utc,
        altitude_km,
        latitude_deg,
        longitude_deg,
        f107,
        f107a,
        ap,
        [-1.0; 7],
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
    if !epoch_jd.is_finite()
        || !f107.is_finite()
        || !f107a.is_finite()
        || !ap.is_finite()
    {
        return Err(PyValueError::new_err("atmosphere inputs must be finite"));
    }
    let values: Vec<f64> = py.import("numpy")?.call_method1("asarray",(position_inertial_m,))?
        .call_method1("astype",("float64",))?.call_method0("tolist")?.extract()?;
    if values.len()!=3 || values.iter().any(|v|!v.is_finite()) {
        return Err(PyValueError::new_err("position_inertial_m must be a finite 3-vector"));
    }
    let history=ap_array.unwrap_or([-1.;7]);
    if history.iter().any(|v|!v.is_finite()) {
        return Err(PyValueError::new_err("ap_array must contain finite values"));
    }
    native_density_with([values[0],values[1],values[2]],epoch_jd,f107,f107a,ap,history,effective_density)
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
    ) -> f64 {
        density_geodetic_impl(
            day_of_year_utc,
            seconds_utc,
            altitude_km,
            latitude_deg,
            longitude_deg,
            f107,
            f107a,
            ap,
            [-1.0; 7],
            effective_density,
        )
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
