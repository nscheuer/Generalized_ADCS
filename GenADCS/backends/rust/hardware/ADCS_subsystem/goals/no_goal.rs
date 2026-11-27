use super::goal::{validate_reference_state, Goal, GoalReference};

/// Return no references, indicating that attitude control is inactive.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoGoal;

impl NoGoal {
    pub fn new() -> Self {
        Self
    }
}

impl Goal for NoGoal {
    fn reference(&self, state: &[f64], _epoch_jd: f64, _mu: f64) -> Result<GoalReference, String> {
        validate_reference_state(state)?;
        Ok((None, None))
    }
}
