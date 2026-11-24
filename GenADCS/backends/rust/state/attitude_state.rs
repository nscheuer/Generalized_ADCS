use crate::state_types::{Quaternion, Vector3};

#[derive(Clone, Debug, PartialEq)]
pub struct AttitudeState {
    pub orientation: Quaternion,
    pub rates: Vector3,
}

impl AttitudeState {
    pub fn new(orientation: Quaternion, rates: Vector3) -> Result<Self, &'static str> {
        if orientation.iter().any(|value| !value.is_finite()) {
            return Err("orientation must be finite");
        }
        if rates.iter().any(|value| !value.is_finite()) {
            return Err("rates must be finite");
        }
        if orientation.iter().all(|value| *value == 0.0) {
            return Err("orientation quaternion must be finite and nonzero");
        }
        Ok(Self { orientation, rates })
    }

    pub fn normalized_quaternion(&self) -> Result<Quaternion, &'static str> {
        let norm = self
            .orientation
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if norm == 0.0 || !norm.is_finite() {
            return Err("orientation quaternion must be finite and nonzero");
        }
        Ok(self.orientation.map(|value| value / norm))
    }
}
