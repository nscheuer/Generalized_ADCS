use ephemeris_native::NativeEphemeris;
use math_native::{dcm_to_quaternion_native, utc_jd_to_tdb_jd_native};

use super::goal::{
    dcm_from_rows, validate_reference_state, vec_add, vec_cross, vec_dot, vec_norm, vec_scale,
    vec_sub, GoalReference,
};

#[derive(Debug, Clone, Copy)]
pub(super) enum OrbitMode {
    Nadir,
    Zenith,
    AlongTrack,
    CrossTrack,
    Velocity,
    Sun,
}

#[inline]
fn unit_rate(vector: [f64; 3], derivative: [f64; 3]) -> Result<([f64; 3], [f64; 3]), String> {
    let length = vec_norm(vector);
    if length <= 0.0 {
        return Err("reference direction is undefined".into());
    }
    let inv = 1.0 / length;
    let unit = vec_scale(vector, inv);
    let proj = vec_dot(unit, derivative);
    let unit_dot = vec_scale(vec_sub(derivative, vec_scale(unit, proj)), inv);
    Ok((unit, unit_dot))
}

pub(super) fn orbit_reference(
    state: &[f64],
    epoch_jd: f64,
    mu: f64,
    mode: OrbitMode,
    ephemeris: Option<&NativeEphemeris>,
) -> Result<GoalReference, String> {
    if !epoch_jd.is_finite() {
        return Err("epoch_jd must be finite".into());
    }
    validate_reference_state(state)?;

    let position = [state[7], state[8], state[9]];
    let velocity = [state[10], state[11], state[12]];
    let radius = vec_norm(position);
    if radius <= 0.0 {
        return Err("orbital frame is undefined at zero position".into());
    }
    let acceleration = vec_scale(position, -mu / radius.powi(3));
    let (radial, radial_dot) = unit_rate(position, velocity)?;

    let v_dot_r = vec_dot(velocity, radial);
    let tangent_raw = vec_sub(velocity, vec_scale(radial, v_dot_r));
    let tangent_dot_raw = vec_sub(
        vec_sub(
            acceleration,
            vec_scale(
                radial,
                vec_dot(acceleration, radial) + vec_dot(velocity, radial_dot),
            ),
        ),
        vec_scale(radial_dot, v_dot_r),
    );
    let (tangent, tangent_dot) = unit_rate(tangent_raw, tangent_dot_raw)?;

    let orbit_normal_raw = vec_cross(position, velocity);
    let orbit_normal_dot_raw = vec_cross(position, acceleration);
    let (orbit_normal, orbit_normal_dot) = unit_rate(orbit_normal_raw, orbit_normal_dot_raw)?;

    let (z, z_dot, x_raw, x_dot_raw) = match mode {
        OrbitMode::Nadir => (
            vec_scale(radial, -1.0),
            vec_scale(radial_dot, -1.0),
            tangent,
            tangent_dot,
        ),
        OrbitMode::Zenith => (radial, radial_dot, tangent, tangent_dot),
        OrbitMode::AlongTrack => (tangent, tangent_dot, radial, radial_dot),
        OrbitMode::CrossTrack => (orbit_normal, orbit_normal_dot, radial, radial_dot),
        OrbitMode::Velocity => {
            let (z, z_dot) = unit_rate(velocity, acceleration)?;
            (z, z_dot, radial, radial_dot)
        }
        OrbitMode::Sun => {
            let eph = ephemeris.ok_or_else(|| "sun goal requires ephemeris".to_string())?;
            let tdb_jd = utc_jd_to_tdb_jd_native(epoch_jd).map_err(|e| e.to_string())?;
            let sun = vec_scale(eph.position_km(0, tdb_jd)?, 1000.0);
            let sun_velocity = vec_scale(eph.velocity_km_s(0, tdb_jd)?, 1000.0);
            let (z, z_dot) = unit_rate(vec_sub(sun, position), vec_sub(sun_velocity, velocity))?;
            let projected = vec_sub(tangent, vec_scale(z, vec_dot(tangent, z)));
            let (xr, xdr) = if vec_norm(projected) < 1e-10 {
                (radial, radial_dot)
            } else {
                (tangent, tangent_dot)
            };
            (z, z_dot, xr, xdr)
        }
    };

    // Project the secondary axis into the plane normal to body +Z and include
    // the derivative of the projection so the reference rate tracks the frame.
    let scalar = vec_dot(x_raw, z);
    let scalar_dot = vec_dot(x_dot_raw, z) + vec_dot(x_raw, z_dot);
    let x_projected = vec_sub(x_raw, vec_scale(z, scalar));
    let x_projected_dot = vec_sub(
        vec_sub(x_dot_raw, vec_scale(z, scalar_dot)),
        vec_scale(z_dot, scalar),
    );

    let (x, x_dot) = unit_rate(x_projected, x_projected_dot)?;
    let y_raw = vec_cross(z, x);
    let y_dot_raw = vec_add(vec_cross(z_dot, x), vec_cross(z, x_dot));
    let (body_y, body_y_dot) = unit_rate(y_raw, y_dot_raw)?;
    let body_x = vec_cross(body_y, z);
    let body_x_dot = vec_add(vec_cross(body_y_dot, z), vec_cross(body_y, z_dot));

    let dcm = dcm_from_rows(body_x, body_y, z);
    let quaternion = dcm_to_quaternion_native(dcm).map_err(|e| e.to_string())?;
    let rate = [
        -vec_dot(body_y_dot, z),
        -vec_dot(z_dot, body_x),
        -vec_dot(body_x_dot, body_y),
    ];
    Ok((Some(quaternion), Some(rate)))
}

macro_rules! define_orbit_goal {
    ($name:ident, $mode:expr, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $name;

        impl $name {
            pub fn new() -> Self {
                Self
            }
        }

        impl super::goal::Goal for $name {
            fn reference(
                &self,
                state: &[f64],
                epoch_jd: f64,
                mu: f64,
            ) -> Result<super::goal::GoalReference, String> {
                super::orbit_goal_base::orbit_reference(state, epoch_jd, mu, $mode, None)
            }
        }
    };
}

pub(super) use define_orbit_goal;
