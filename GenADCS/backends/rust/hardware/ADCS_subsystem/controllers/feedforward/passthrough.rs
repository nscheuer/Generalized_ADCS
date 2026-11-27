use super::feedforward::{Feedforward, FF_PASSTHROUGH};

pub fn passthrough_step(
    q_goal: [f64; 4],
    w_goal: [f64; 3],
    q_current: [f64; 4],
) -> ([f64; 4], [f64; 3], [f64; 3], [f64; 3]) {
    let q_d = if q_goal[0].is_nan() {
        q_current
    } else {
        q_goal
    };
    let w_d = if w_goal[0].is_nan() {
        [0.0; 3]
    } else {
        w_goal
    };
    (q_d, w_d, [0.0; 3], [0.0; 3])
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Passthrough;

impl Feedforward for Passthrough {
    fn kind(&self) -> i32 {
        FF_PASSTHROUGH
    }

    fn to_params(&self) -> Vec<f64> {
        vec![0.0]
    }
}
