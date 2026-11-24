/// Density from geodetic position and UTC time using NRLMSISE-00.
pub fn density_geodetic(
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
    evaluate_geodetic(
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

/// Density at an inertial position, with optional seven-element Ap history.
pub fn density_inertial(
    position_inertial_m: [f64; 3],
    epoch_jd: f64,
    f107: f64,
    f107a: f64,
    ap: f64,
    ap_history: [f64; 7],
    effective_density: bool,
) -> Result<f64, String> {
    validate_finite(&[
        position_inertial_m[0],
        position_inertial_m[1],
        position_inertial_m[2],
        epoch_jd,
        f107,
        f107a,
        ap,
        ap_history[0],
        ap_history[1],
        ap_history[2],
        ap_history[3],
        ap_history[4],
        ap_history[5],
        ap_history[6],
    ])
    .map_err(str::to_owned)?;
    native_density_with(
        position_inertial_m,
        epoch_jd,
        f107,
        f107a,
        ap,
        ap_history,
        effective_density,
    )
}

fn validate_finite(values: &[f64]) -> Result<(), &'static str> {
    if values.iter().any(|value| !value.is_finite()) {
        Err("atmosphere inputs must be finite")
    } else {
        Ok(())
    }
}
