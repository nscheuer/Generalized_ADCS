use math_native::dcm_to_quaternion_native;
use nalgebra::{Matrix3, RowVector3};

/// Default Earth gravitational parameter (m^3 / s^2).
pub const DEFAULT_MU: f64 = 3.986004418e14;

/// Result of evaluating an attitude-reference goal: `(quaternion, body_rate)`.
pub type GoalReference = (Option<[f64; 4]>, Option<[f64; 3]>);

/// Contract for pure-Rust attitude-reference goals.
pub trait Goal {
    /// Return desired `(q_body_from_inertial, omega_body)`; `None` means unset.
    fn reference(&self, state: &[f64], epoch_jd: f64, mu: f64) -> Result<GoalReference, String>;
}

/// Validate that `state` has a finite 13-element prefix with nonzero quaternion.
pub fn validate_reference_state(state: &[f64]) -> Result<(), String> {
    if state.len() < 13 || !state.iter().all(|v| v.is_finite()) {
        return Err("goal state must have a finite 13-element spacecraft-state prefix".into());
    }
    let q_norm_sq =
        state[0] * state[0] + state[1] * state[1] + state[2] * state[2] + state[3] * state[3];
    if q_norm_sq == 0.0 {
        return Err("goal state quaternion must be nonzero".into());
    }
    Ok(())
}

/// Validate and normalize a scalar-first Hamilton quaternion.
pub fn validate_quaternion(quaternion: [f64; 4]) -> Result<[f64; 4], String> {
    if !quaternion.iter().all(|v| v.is_finite()) {
        return Err("quaternion must be a finite 4-element vector".into());
    }
    let norm = (quaternion[0] * quaternion[0]
        + quaternion[1] * quaternion[1]
        + quaternion[2] * quaternion[2]
        + quaternion[3] * quaternion[3])
        .sqrt();
    if norm == 0.0 {
        return Err("quaternion must be nonzero".into());
    }
    Ok([
        quaternion[0] / norm,
        quaternion[1] / norm,
        quaternion[2] / norm,
        quaternion[3] / norm,
    ])
}

/// Validate and normalize a finite nonzero 3-vector.
pub fn validate_vector(vector: [f64; 3], name: &str) -> Result<[f64; 3], String> {
    if !vector.iter().all(|v| v.is_finite()) {
        return Err(format!("{name} must be a finite 3-element vector"));
    }
    let norm = vec_norm(vector);
    if norm == 0.0 {
        return Err(format!("{name} must be nonzero"));
    }
    Ok([vector[0] / norm, vector[1] / norm, vector[2] / norm])
}

#[inline]
pub(crate) fn vec_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub(crate) fn vec_norm(a: [f64; 3]) -> f64 {
    vec_dot(a, a).sqrt()
}

#[inline]
pub(crate) fn vec_cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub(crate) fn vec_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub(crate) fn vec_add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub(crate) fn vec_scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub(crate) fn dcm_from_rows(row0: [f64; 3], row1: [f64; 3], row2: [f64; 3]) -> Matrix3<f64> {
    Matrix3::from_rows(&[
        RowVector3::from(row0),
        RowVector3::from(row1),
        RowVector3::from(row2),
    ])
}

/// Build right-handed triads from `(primary, secondary)` in inertial and body frames
/// and return `dcm_to_quaternion(body_basis * inertial_basis.T)`.
pub(crate) fn triad_quaternion(
    inertial_primary: [f64; 3],
    inertial_secondary: [f64; 3],
    body_primary: [f64; 3],
    body_secondary: [f64; 3],
) -> Result<[f64; 4], String> {
    let i0 = inertial_primary;
    let i1_raw = vec_sub(
        inertial_secondary,
        vec_scale(i0, vec_dot(inertial_secondary, i0)),
    );
    let i1_norm = vec_norm(i1_raw);
    let i1 = vec_scale(i1_raw, 1.0 / i1_norm);
    let i2 = vec_cross(i0, i1);

    let b0 = body_primary;
    let b1_raw = vec_sub(body_secondary, vec_scale(b0, vec_dot(body_secondary, b0)));
    let b1_norm = vec_norm(b1_raw);
    let b1 = vec_scale(b1_raw, 1.0 / b1_norm);
    let b2 = vec_cross(b0, b1);

    // basis = [v0 | v1 | v2] (columns), so body_basis * inertial_basis.T
    let inertial_basis = Matrix3::from_columns(&[i0.into(), i1.into(), i2.into()]);
    let body_basis = Matrix3::from_columns(&[b0.into(), b1.into(), b2.into()]);
    let dcm = body_basis * inertial_basis.transpose();
    dcm_to_quaternion_native(dcm).map_err(|e| e.to_string())
}
