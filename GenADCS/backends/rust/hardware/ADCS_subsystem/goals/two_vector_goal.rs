use super::goal::{
    triad_quaternion, validate_reference_state, validate_vector, vec_dot, Goal, GoalReference,
};

/// Track two nonparallel inertial-to-body vector correspondences.
#[derive(Debug, Clone, Copy)]
pub struct TwoVectorGoal {
    pub inertial_first: [f64; 3],
    pub inertial_second: [f64; 3],
    pub body_first: [f64; 3],
    pub body_second: [f64; 3],
}

impl TwoVectorGoal {
    pub fn new(
        inertial_vectors: [[f64; 3]; 2],
        body_vectors: [[f64; 3]; 2],
    ) -> Result<Self, String> {
        let inertial_first = validate_vector(inertial_vectors[0], "inertial_vectors[0]")?;
        let inertial_second = validate_vector(inertial_vectors[1], "inertial_vectors[1]")?;
        let body_first = validate_vector(body_vectors[0], "body_vectors[0]")?;
        let body_second = validate_vector(body_vectors[1], "body_vectors[1]")?;

        if vec_dot(inertial_first, inertial_second).abs() > 1.0 - 1e-10 {
            return Err("inertial vectors must not be parallel".into());
        }
        if vec_dot(body_first, body_second).abs() > 1.0 - 1e-10 {
            return Err("body vectors must not be parallel".into());
        }

        Ok(Self {
            inertial_first,
            inertial_second,
            body_first,
            body_second,
        })
    }
}

impl Goal for TwoVectorGoal {
    fn reference(&self, state: &[f64], _epoch_jd: f64, _mu: f64) -> Result<GoalReference, String> {
        validate_reference_state(state)?;
        let q = triad_quaternion(
            self.inertial_first,
            self.inertial_second,
            self.body_first,
            self.body_second,
        )?;
        Ok((Some(q), Some([0.0; 3])))
    }
}
