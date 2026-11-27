use nalgebra::Matrix3;

use super::coordinate_goal::{
    central_difference_reference, ecef_target_to_gcrf, finish_dcm, CoordinateGoal,
};
use super::goal::{
    validate_reference_state, vec_add, vec_cross, vec_dot, vec_norm, vec_scale, vec_sub, Goal,
    GoalReference,
};

const EARTH_RATE: [f64; 3] = [0.0, 0.0, 7.2921159e-5];

fn coordinate_fmc_dcm(
    position: [f64; 3],
    velocity: [f64; 3],
    epoch_jd: f64,
    target_ecef: [f64; 3],
    fmc_level: i64,
) -> Result<Matrix3<f64>, String> {
    let target = ecef_target_to_gcrf(epoch_jd, target_ecef);
    let radius = vec_norm(position);
    if radius == 0.0 {
        return Err("coordinate FMC goal is undefined at zero position".into());
    }
    let radial = vec_scale(position, 1.0 / radius);
    let nadir = vec_scale(radial, -1.0);

    let target_direction = vec_sub(target, position);
    let mut azimuth = vec_sub(
        target_direction,
        vec_scale(radial, vec_dot(target_direction, radial)),
    );
    let mut azimuth_norm = vec_norm(azimuth);
    let ground_velocity = vec_sub(velocity, vec_cross(EARTH_RATE, position));
    if azimuth_norm < 1e-12 {
        azimuth = vec_sub(
            ground_velocity,
            vec_scale(radial, vec_dot(ground_velocity, radial)),
        );
        azimuth_norm = vec_norm(azimuth);
    }
    if azimuth_norm < 1e-12 {
        return Err("coordinate FMC tilt azimuth is degenerate".into());
    }
    azimuth = vec_scale(azimuth, 1.0 / azimuth_norm);

    let tilt = (1.0 / (fmc_level as f64)).acos();
    let z_axis = vec_add(
        vec_scale(nadir, tilt.cos()),
        vec_scale(azimuth, tilt.sin()),
    );

    let mut x_raw = vec_sub(
        ground_velocity,
        vec_scale(z_axis, vec_dot(ground_velocity, z_axis)),
    );
    let mut x_norm = vec_norm(x_raw);
    if x_norm < 1e-12 {
        x_raw = vec_sub(radial, vec_scale(z_axis, vec_dot(radial, z_axis)));
        x_norm = vec_norm(x_raw);
    }
    if x_norm < 1e-12 {
        return Err("coordinate FMC roll reference is degenerate".into());
    }
    Ok(finish_dcm(z_axis, x_raw, x_norm))
}

/// Tilt toward a geodetic coordinate with a selected FMC reduction level (1..=10).
#[derive(Debug, Clone, Copy)]
pub struct CoordinateFMCGoal {
    pub coordinate: CoordinateGoal,
    pub fmc_level: i64,
}

impl CoordinateFMCGoal {
    pub fn new(
        lat: f64,
        lon: f64,
        alt: Option<f64>,
        fmc_level: Option<i64>,
    ) -> Result<Self, String> {
        let coordinate = CoordinateGoal::new(lat, lon, alt)?;
        let fmc_level = fmc_level.unwrap_or(1);
        if !(1..=10).contains(&fmc_level) {
            return Err("fmc_level must be an integer from 1 through 10".into());
        }
        Ok(Self {
            coordinate,
            fmc_level,
        })
    }
}

impl Goal for CoordinateFMCGoal {
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
            "coordinate FMC goal is undefined at zero position",
            |pos, vel, jd| {
                coordinate_fmc_dcm(pos, vel, jd, self.coordinate.target_ecef_m, self.fmc_level)
            },
        )
    }
}
