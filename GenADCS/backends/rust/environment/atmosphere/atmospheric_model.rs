/// Generic density selection is intentionally left to the caller.
pub fn generic_density(
    _position_inertial_m: [f64; 3],
    _epoch_jd: f64,
) -> Result<f64, &'static str> {
    Err("Choose an atmospheric density model")
}

/// Velocity of a co-rotating atmosphere at an inertial position.
pub fn co_rotating_velocity(
    position_inertial_m: [f64; 3],
    _epoch_jd: f64,
    angular_velocity_inertial_rad_s: Option<[f64; 3]>,
) -> Result<[f64; 3], &'static str> {
    if position_inertial_m.iter().any(|value| !value.is_finite()) {
        return Err("position_inertial_m must be a finite 3-vector");
    }
    let position = position_inertial_m;
    let omega = angular_velocity_inertial_rad_s.unwrap_or([0.0, 0.0, 7.2921150e-5]);
    Ok([
        omega[1] * position[2] - omega[2] * position[1],
        omega[2] * position[0] - omega[0] * position[2],
        omega[0] * position[1] - omega[1] * position[0],
    ])
}
