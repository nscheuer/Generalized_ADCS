use pyo3::exceptions::PyValueError;
fn utc_fields(py: Python<'_>, jd: f64) -> PyResult<(i32, i32, i32, f64)> {
    py.import("ADCS_Omni.ADCS_Rust.helpers.math.time")?
        .getattr("jd_to_utc_calendar_fields")?
        .call1((jd,))?
        .extract()
}

fn doy(py: Python<'_>, year: i32, month: i32, day: i32) -> PyResult<i32> {
    py.import("ADCS_Omni.ADCS_Rust.helpers.math.time")?
        .getattr("day_of_year")?
        .call1((year, month, day))?
        .extract()
}

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
    let (year, month, day, seconds) = utc_fields(py, epoch_jd)?;
    let day_of_year = doy(py, year, month, day)?;
    let jd_tt = if epoch_jd >= 2_441_317.5 {
        py.import("ADCS_Omni.ADCS_Rust.helpers.math.time")?
            .getattr("utc_jd_to_tt_jd")?
            .call1((epoch_jd,))?
            .extract::<f64>()?
    } else {
        epoch_jd
    };
    let geodetic = py
        .import("ADCS_Omni.ADCS_Rust.helpers.math.frames")?
        .getattr("gcrf_position_to_geodetic")?
        .call1((position_inertial_m, epoch_jd, jd_tt))?
        .extract::<(f64, f64, f64)>()?;
    let history = match ap_array {
        Some(values) => values,
        None => [-1.0; 7],
    };
    if history.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("ap_array must contain finite values"));
    }
    Ok(density_geodetic_impl(
        day_of_year,
        seconds,
        geodetic.2,
        geodetic.0,
        geodetic.1,
        f107,
        f107a,
        ap,
        history,
        effective_density,
    ))
}

#[pyclass(name = "NRLMSISE00")]
struct NRLMSISE00;

#[pymethods]
impl NRLMSISE00 {
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
