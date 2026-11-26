use crate::state_types::{KeplerianElements, Vector3};

pub const EARTH_MU: f64 = 3.986_004_418e14;

pub fn keplerian_to_cartesian(
    elements: KeplerianElements,
    mu: f64,
) -> Result<(Vector3, Vector3), &'static str> {
    if !mu.is_finite() || mu <= 0.0 {
        return Err("gravitational parameter must be positive and finite");
    }
    if elements.iter().any(|v| !v.is_finite()) {
        return Err("Keplerian elements must be finite");
    }
    let [a, e, i, raan, argp, nu] = elements;
    if e < 0.0 {
        return Err("eccentricity must be non-negative");
    }
    let p = a * (1.0 - e * e);
    if !p.is_finite() || p <= 0.0 {
        return Err("semi_major_axis and eccentricity must yield a positive semi-latus rectum");
    }
    let (sin_nu, cos_nu) = nu.sin_cos();
    let denom = 1.0 + e * cos_nu;
    if !denom.is_finite() || denom <= 0.0 {
        return Err("true_anomaly is out of bounds for orbit");
    }
    let r = p / denom;
    let sqrt_mu_p = (mu / p).sqrt();
    let r_pqw = [r * cos_nu, r * sin_nu];
    let v_pqw = [-sqrt_mu_p * sin_nu, sqrt_mu_p * (e + cos_nu)];

    let (sin_i, cos_i) = i.sin_cos();
    let (sin_o, cos_o) = raan.sin_cos();
    let (sin_w, cos_w) = argp.sin_cos();

    let p_vec = [
        cos_o * cos_w - sin_o * sin_w * cos_i,
        sin_o * cos_w + cos_o * sin_w * cos_i,
        sin_w * sin_i,
    ];
    let q_vec = [
        -cos_o * sin_w - sin_o * cos_w * cos_i,
        -sin_o * sin_w + cos_o * cos_w * cos_i,
        cos_w * sin_i,
    ];

    let position = [
        p_vec[0] * r_pqw[0] + q_vec[0] * r_pqw[1],
        p_vec[1] * r_pqw[0] + q_vec[1] * r_pqw[1],
        p_vec[2] * r_pqw[0] + q_vec[2] * r_pqw[1],
    ];
    let velocity = [
        p_vec[0] * v_pqw[0] + q_vec[0] * v_pqw[1],
        p_vec[1] * v_pqw[0] + q_vec[1] * v_pqw[1],
        p_vec[2] * v_pqw[0] + q_vec[2] * v_pqw[1],
    ];
    Ok((position, velocity))
}

#[derive(Clone, Debug, PartialEq)]
pub struct OrbitalState {
    pub position: Option<Vector3>,
    pub velocity: Option<Vector3>,
    pub julian_date: f64,
    pub keplerian_elements: Option<KeplerianElements>,
}

impl OrbitalState {
    pub fn new_cartesian(
        position: Option<Vector3>,
        velocity: Option<Vector3>,
        julian_date: f64,
    ) -> Result<Self, &'static str> {
        if !julian_date.is_finite() {
            return Err("julian_date must be finite");
        }
        if position.map_or(false, |values| values.iter().any(|v| !v.is_finite())) {
            return Err("position must be a finite 3-vector");
        }
        if velocity.map_or(false, |values| values.iter().any(|v| !v.is_finite())) {
            return Err("velocity must be a finite 3-vector");
        }
        Ok(Self {
            position,
            velocity,
            julian_date,
            keplerian_elements: None,
        })
    }

    pub fn new_keplerian(
        elements: KeplerianElements,
        julian_date: f64,
    ) -> Result<Self, &'static str> {
        if !julian_date.is_finite() {
            return Err("julian_date must be finite");
        }
        if elements.iter().any(|v| !v.is_finite()) {
            return Err("Keplerian elements must be finite");
        }
        Ok(Self {
            position: None,
            velocity: None,
            julian_date,
            keplerian_elements: Some(elements),
        })
    }

    pub fn cartesian(&self) -> Result<(Vector3, Vector3), &'static str> {
        match (self.position, self.velocity) {
            (Some(position), Some(velocity)) => Ok((position, velocity)),
            _ => match self.keplerian_elements {
                Some(elements) => keplerian_to_cartesian(elements, EARTH_MU),
                None => Err("orbital state has neither Cartesian vectors nor Keplerian elements"),
            },
        }
    }
}
