use super::goal::{
    triad_quaternion, validate_reference_state, validate_vector, vec_dot, Goal, GoalReference,
};

/// Align `body_axis` to `inertial_vector` and resolve roll with references.
#[derive(Debug, Clone, Copy)]
pub struct VectorGoal {
    pub inertial_vector: [f64; 3],
    pub body_axis: [f64; 3],
    pub inertial_roll: [f64; 3],
    pub body_roll: [f64; 3],
}

impl VectorGoal {
    pub fn new(
        inertial_vector: [f64; 3],
        body_axis: Option<[f64; 3]>,
        inertial_roll: Option<[f64; 3]>,
        body_roll: Option<[f64; 3]>,
    ) -> Result<Self, String> {
        let inertial_vector = validate_vector(inertial_vector, "inertial_vector")?;
        let body_axis = validate_vector(body_axis.unwrap_or([0.0, 0.0, 1.0]), "body_axis")?;
        let inertial_roll =
            validate_vector(inertial_roll.unwrap_or([1.0, 0.0, 0.0]), "inertial_roll")?;
        let body_roll = validate_vector(body_roll.unwrap_or([1.0, 0.0, 0.0]), "body_roll")?;

        if vec_dot(inertial_vector, inertial_roll).abs() > 1.0 - 1e-10 {
            return Err("inertial_roll must not be parallel to inertial_vector".into());
        }
        if vec_dot(body_axis, body_roll).abs() > 1.0 - 1e-10 {
            return Err("body_roll must not be parallel to body_axis".into());
        }

        Ok(Self {
            inertial_vector,
            body_axis,
            inertial_roll,
            body_roll,
        })
    }
}

impl Goal for VectorGoal {
    fn reference(&self, state: &[f64], _epoch_jd: f64, _mu: f64) -> Result<GoalReference, String> {
        validate_reference_state(state)?;
        let q = triad_quaternion(
            self.inertial_vector,
            self.inertial_roll,
            self.body_axis,
            self.body_roll,
        )?;
        Ok((Some(q), Some([0.0; 3])))
    }
}
