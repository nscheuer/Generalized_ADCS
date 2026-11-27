use std::fmt;

pub const EARTH_ZONAL_COEFFICIENTS: [f64; 5] = [
    1.08262668e-3,
    -2.5324105e-6,
    -1.6198976e-6,
    -2.2729608e-7,
    5.406812e-7,
];
pub const EARTH_MU_M3_S2: f64 = 3.986004418e14;
pub const EARTH_REFERENCE_RADIUS_M: f64 = 6_378_137.0;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GravityError {
    AtBodyCentre,
    InvalidDegree(i32),
    InsufficientCoefficients { provided: usize, required: usize },
}

impl fmt::Display for GravityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AtBodyCentre => write!(f, "Gravity is undefined at the body's centre"),
            Self::InvalidDegree(degree) => {
                write!(f, "J-gravity degree {degree} is outside the supported range 2 through 6")
            }
            Self::InsufficientCoefficients { provided, required } => write!(
                f,
                "Provide {required} coefficients ordered as J2, J3, J4, J5, J6 (got {provided})"
            ),
        }
    }
}

impl std::error::Error for GravityError {}

pub fn native_acceleration(
    position: [f64; 3],
    mu: f64,
    radius: f64,
    coefficients: &[f64],
) -> Result<[f64; 3], GravityError> {
    let [x, y, z] = position;
    let r2 = x * x + y * y + z * z;
    let r = r2.sqrt();
    if r == 0.0 {
        return Err(GravityError::AtBodyCentre);
    }
    let inv_r = 1.0 / r;
    let s = z * inv_r;
    let ratio = radius * inv_r;
    let mut ratio_n = ratio * ratio;
    let p = [
        0.5 * (3.0 * s * s - 1.0),
        0.5 * (5.0 * s.powi(3) - 3.0 * s),
        (35.0 * s.powi(4) - 30.0 * s * s + 3.0) / 8.0,
        (63.0 * s.powi(5) - 70.0 * s.powi(3) + 15.0 * s) / 8.0,
        (231.0 * s.powi(6) - 315.0 * s.powi(4) + 105.0 * s * s - 5.0) / 16.0,
    ];
    let dp = [
        3.0 * s,
        0.5 * (15.0 * s * s - 3.0),
        (140.0 * s.powi(3) - 60.0 * s) / 8.0,
        (315.0 * s.powi(4) - 210.0 * s * s + 15.0) / 8.0,
        (1386.0 * s.powi(5) - 1260.0 * s.powi(3) + 210.0 * s) / 16.0,
    ];
    let mut radial = 1.0;
    let mut z_gradient = 0.0;
    for k in 0..coefficients.len().min(5) {
        let jn = coefficients[k];
        let n = (k + 2) as f64;
        radial -= (n + 1.0) * jn * ratio_n * p[k];
        z_gradient += jn * ratio_n * dp[k];
        ratio_n *= ratio;
    }
    let mu_r3 = mu / (r2 * r);
    let base = -mu_r3 * radial;
    // The latitude term enters with a negative sign:
    // -mu/r^3 * sum(Jn (R/r)^n Pn'(s)) * (r z_hat - s r_vec).
    Ok([
        base * x - mu_r3 * z_gradient * (-s * x),
        base * y - mu_r3 * z_gradient * (-s * y),
        base * z - mu_r3 * z_gradient * (r - s * z),
    ])
}

pub fn acceleration(
    position: [f64; 3],
    mu: f64,
    radius: f64,
    coefficients: &[f64],
) -> Result<[f64; 3], GravityError> {
    native_acceleration(position, mu, radius, coefficients)
}

#[derive(Clone, Debug, PartialEq)]
pub struct JGravityModel {
    degrees: Vec<i32>,
    zonal_coefficients: [f64; 5],
}

impl JGravityModel {
    pub fn configured(
        degrees: Vec<i32>,
        coefficients: Option<Vec<f64>>,
    ) -> Result<Self, GravityError> {
        Self::new(Some(degrees), coefficients)
    }

    pub fn new(
        degrees: Option<Vec<i32>>,
        coefficients: Option<Vec<f64>>,
    ) -> Result<Self, GravityError> {
        let mut degrees = degrees.unwrap_or_else(|| vec![2, 3, 4, 5, 6]);
        degrees.sort_unstable();
        degrees.dedup();
        if let Some(degree) = degrees.iter().find(|degree| !(2..=6).contains(*degree)) {
            return Err(GravityError::InvalidDegree(*degree));
        }
        if let Some(values) = &coefficients {
            if values.len() < 5 {
                return Err(GravityError::InsufficientCoefficients {
                    provided: values.len(),
                    required: 5,
                });
            }
        }
        let source = coefficients.unwrap_or_else(|| EARTH_ZONAL_COEFFICIENTS.to_vec());
        let mut zonal_coefficients = [0.0; 5];
        for degree in &degrees {
            zonal_coefficients[(degree - 2) as usize] = source[(degree - 2) as usize];
        }
        Ok(Self {
            degrees,
            zonal_coefficients,
        })
    }

    pub fn degrees(&self) -> &[i32] {
        &self.degrees
    }

    pub fn zonal_coefficients(&self) -> &[f64; 5] {
        &self.zonal_coefficients
    }

    pub fn evaluate(&self, position_inertial_m: [f64; 3]) -> Result<[f64; 3], GravityError> {
        native_acceleration(
            position_inertial_m,
            EARTH_MU_M3_S2,
            EARTH_REFERENCE_RADIUS_M,
            &self.zonal_coefficients,
        )
    }

    pub fn acceleration_inertial_m_s2(
        position_inertial_m: [f64; 3],
        gravitational_parameter_m3_s2: f64,
        reference_radius_m: f64,
        zonal_coefficients: &[f64],
    ) -> Result<[f64; 3], GravityError> {
        native_acceleration(
            position_inertial_m,
            gravitational_parameter_m3_s2,
            reference_radius_m,
            zonal_coefficients,
        )
    }
}
