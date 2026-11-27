use frames_native::ecef_to_gcrf_matrix_native;
use math_native::dcm_to_quaternion_native;
use nalgebra::{Matrix3, Vector3};

use super::goal::{
    dcm_from_rows, validate_reference_state, vec_add, vec_cross, vec_dot, vec_norm, vec_scale,
    vec_sub, Goal, GoalReference,
};

#[inline]
pub(crate) fn ecef_target_to_gcrf(epoch_jd: f64, target_ecef: [f64; 3]) -> [f64; 3] {
    let rot = ecef_to_gcrf_matrix_native(epoch_jd, epoch_jd, 0.0, 0.0);
    let v = rot * Vector3::new(target_ecef[0], target_ecef[1], target_ecef[2]);
    [v[0], v[1], v[2]]
}

#[inline]
pub(crate) fn finish_dcm(
    z_axis: [f64; 3],
    x_raw: [f64; 3],
    x_norm: f64,
) -> Matrix3<f64> {
    let x_axis = vec_scale(x_raw, 1.0 / x_norm);
    let y_raw = vec_cross(z_axis, x_axis);
    let y_axis = vec_scale(y_raw, 1.0 / vec_norm(y_raw));
    let x_axis = vec_cross(y_axis, z_axis);
    dcm_from_rows(x_axis, y_axis, z_axis)
}

pub(crate) fn central_difference_reference<F>(
    position: [f64; 3],
    velocity: [f64; 3],
    epoch_jd: f64,
    mu: f64,
    zero_pos_msg: &str,
    dcm_fn: F,
) -> Result<GoalReference, String>
where
    F: Fn([f64; 3], [f64; 3], f64) -> Result<Matrix3<f64>, String>,
{
    let dcm = dcm_fn(position, velocity, epoch_jd)?;
    let quaternion = dcm_to_quaternion_native(dcm).map_err(|e| e.to_string())?;

    let radius = vec_norm(position);
    if radius == 0.0 {
        return Err(zero_pos_msg.into());
    }
    let acceleration = vec_scale(position, -mu / radius.powi(3));
    let step = 1.0;
    let half_step_sq = 0.5 * step * step;
    let plus_position = vec_add(
        vec_add(position, vec_scale(velocity, step)),
        vec_scale(acceleration, half_step_sq),
    );
    let minus_position = vec_add(
        vec_sub(position, vec_scale(velocity, step)),
        vec_scale(acceleration, half_step_sq),
    );
    let plus_velocity = vec_add(velocity, vec_scale(acceleration, step));
    let minus_velocity = vec_sub(velocity, vec_scale(acceleration, step));
    let day_step = step / 86400.0;

    let dcm_plus = dcm_fn(plus_position, plus_velocity, epoch_jd + day_step)?;
    let dcm_minus = dcm_fn(minus_position, minus_velocity, epoch_jd - day_step)?;
    let dcm_dot = (dcm_plus - dcm_minus) / (2.0 * step);
    let omega_skew_body = -dcm_dot * dcm.transpose();
    let body_rate = [
        0.5 * (omega_skew_body[(2, 1)] - omega_skew_body[(1, 2)]),
        0.5 * (omega_skew_body[(0, 2)] - omega_skew_body[(2, 0)]),
        0.5 * (omega_skew_body[(1, 0)] - omega_skew_body[(0, 1)]),
    ];
    Ok((Some(quaternion), Some(body_rate)))
}

fn coordinate_dcm(
    position: [f64; 3],
    velocity: [f64; 3],
    epoch_jd: f64,
    target_ecef: [f64; 3],
) -> Result<Matrix3<f64>, String> {
    let target = ecef_target_to_gcrf(epoch_jd, target_ecef);
    let relative = vec_sub(target, position);
    let distance = vec_norm(relative);
    if distance == 0.0 {
        return Err("spacecraft position coincides with coordinate target".into());
    }
    let z_axis = vec_scale(relative, 1.0 / distance);

    let mut x_raw = vec_sub(velocity, vec_scale(z_axis, vec_dot(velocity, z_axis)));
    let mut x_norm = vec_norm(x_raw);
    if x_norm < 1e-12 {
        let position_norm = vec_norm(position);
        if position_norm == 0.0 {
            return Err("coordinate goal roll reference is degenerate".into());
        }
        let radial = vec_scale(position, 1.0 / position_norm);
        x_raw = vec_sub(radial, vec_scale(z_axis, vec_dot(radial, z_axis)));
        x_norm = vec_norm(x_raw);
    }
    if x_norm < 1e-12 {
        return Err("coordinate goal roll reference is degenerate".into());
    }
    Ok(finish_dcm(z_axis, x_raw, x_norm))
}

/// Track a fixed WGS84 latitude/longitude/altitude coordinate.
#[derive(Debug, Clone, Copy)]
pub struct CoordinateGoal {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub alt_km: f64,
    pub target_ecef_m: [f64; 3],
}

impl CoordinateGoal {
    pub fn new(lat: f64, lon: f64, alt: Option<f64>) -> Result<Self, String> {
        let alt = alt.unwrap_or(0.0);
        if !lat.is_finite() || !lon.is_finite() || !alt.is_finite() {
            return Err("latitude, longitude, and altitude must be finite".into());
        }
        if !(-90.0..=90.0).contains(&lat) {
            return Err("latitude must be between -90 and 90 degrees".into());
        }
        if !(-180.0..=180.0).contains(&lon) {
            return Err("longitude must be between -180 and 180 degrees".into());
        }
        if alt < -6378.137 {
            return Err("altitude is below the WGS84 ellipsoid center".into());
        }
        let target_ecef_m = Self::geodetic_to_ecef(lat, lon, alt);
        Ok(Self {
            lat_deg: lat,
            lon_deg: lon,
            alt_km: alt,
            target_ecef_m,
        })
    }

    pub fn geodetic_to_ecef(lat_deg: f64, lon_deg: f64, alt_km: f64) -> [f64; 3] {
        let lat = lat_deg.to_radians();
        let lon = lon_deg.to_radians();
        let semi_major = 6378137.0;
        let flattening = 1.0 / 298.257223563;
        let eccentricity_sq = flattening * (2.0 - flattening);
        let altitude_m = alt_km * 1000.0;
        let sin_lat = lat.sin();
        let cos_lat = lat.cos();
        let prime_vertical = semi_major / (1.0 - eccentricity_sq * sin_lat * sin_lat).sqrt();
        [
            (prime_vertical + altitude_m) * cos_lat * lon.cos(),
            (prime_vertical + altitude_m) * cos_lat * lon.sin(),
            (prime_vertical * (1.0 - eccentricity_sq) + altitude_m) * sin_lat,
        ]
    }
}

impl Goal for CoordinateGoal {
    fn reference(&self, state: &[f64], epoch_jd: f64, mu: f64) -> Result<GoalReference, String> {
        validate_reference_state(state)?;
        if !epoch_jd.is_finite() {
            return Err("epoch_jd must be finite".into());
        }
        let position = [state[7], state[8], state[9]];
        let velocity = [state[10], state[11], state[12]];
        central_difference_reference(
            position,
            velocity,
            epoch_jd,
            mu,
            "coordinate goal is undefined at zero position",
            |pos, vel, jd| coordinate_dcm(pos, vel, jd, self.target_ecef_m),
        )
    }
}
