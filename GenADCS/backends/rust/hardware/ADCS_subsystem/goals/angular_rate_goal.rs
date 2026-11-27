use super::goal::{validate_reference_state, Goal, GoalReference};

/// Track a constant body-frame rate without specifying an attitude target.
#[derive(Debug, Clone, Copy)]
pub struct AngularRateGoal {
    pub rate: [f64; 3],
}

impl AngularRateGoal {
    pub fn new(rate: [f64; 3]) -> Result<Self, String> {
        if !rate.iter().all(|v| v.is_finite()) {
            return Err("rate must be a finite 3-element vector".into());
        }
        Ok(Self { rate })
    }
}

impl Goal for AngularRateGoal {
    fn reference(&self, state: &[f64], _epoch_jd: f64, _mu: f64) -> Result<GoalReference, String> {
        validate_reference_state(state)?;
        Ok((None, Some(self.rate)))
    }
}
