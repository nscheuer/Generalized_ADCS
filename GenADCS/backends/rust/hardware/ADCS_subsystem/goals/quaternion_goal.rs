use super::goal::{validate_quaternion, validate_reference_state, Goal, GoalReference};

/// Track a fixed inertial-to-body quaternion and optional body rate.
#[derive(Debug, Clone, Copy)]
pub struct QuaternionGoal {
    pub quaternion: [f64; 4],
    pub rate: [f64; 3],
}

impl QuaternionGoal {
    pub fn new(quaternion: [f64; 4], rate: Option<[f64; 3]>) -> Result<Self, String> {
        let quaternion = validate_quaternion(quaternion)?;
        let rate = match rate {
            Some(r) => {
                if !r.iter().all(|v| v.is_finite()) {
                    return Err("rate must be a finite 3-element vector".into());
                }
                r
            }
            None => [0.0; 3],
        };
        Ok(Self { quaternion, rate })
    }
}

impl Goal for QuaternionGoal {
    fn reference(&self, state: &[f64], _epoch_jd: f64, _mu: f64) -> Result<GoalReference, String> {
        validate_reference_state(state)?;
        Ok((Some(self.quaternion), Some(self.rate)))
    }
}
