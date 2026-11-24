use crate::state_types::{KeplerianElements, Vector3};

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
            _ => Err("Keplerian to Cartesian conversion is not implemented"),
        }
    }
}
