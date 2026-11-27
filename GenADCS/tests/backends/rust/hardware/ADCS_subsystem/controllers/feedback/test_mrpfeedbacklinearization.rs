use std::f64::consts::PI;

use crate::controllers::feedback::{
    mrp_feedback_linearized_step, Feedback, MRPFeedbackLinearized,
};

#[test]
fn test_mrp_feedback_linearized_gain_validation() {
    let fb = MRPFeedbackLinearized::new(0.5, [1.0, 1.2, 1.4]).unwrap();
    let (kp, kd) = fb.to_matrices();
    assert_eq!(kp.len(), 3);
    assert_eq!(kd.len(), 3);
    assert!(MRPFeedbackLinearized::new(0.0, 1.0).is_err());
    assert!(MRPFeedbackLinearized::new(0.5, -0.1).is_err());
}

#[test]
fn test_mrp_feedback_linearized_step_produces_restoring_torque() {
    let q = [(PI / 6.0).cos(), (PI / 6.0).sin(), 0.0, 0.0];
    let w = [0.0; 3];
    let qd = [1.0, 0.0, 0.0, 0.0];
    let wd = [0.0; 3];
    let inertia = [[0.15, 0.0, 0.0], [0.0, 0.12, 0.0], [0.0, 0.0, 0.08]];
    let eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let kp = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]];
    let kd = [[1.4, 0.0, 0.0], [0.0, 1.4, 0.0], [0.0, 0.0, 1.4]];
    let tau = mrp_feedback_linearized_step(
        q,
        w,
        qd,
        wd,
        wd,
        wd,
        &[0.0; 3],
        &inertia,
        &inertia,
        &eye,
        &kp,
        &kd,
    );
    assert!(tau[0] > 0.0);
    assert!(tau[1].abs() < 1e-12);
    assert!(tau[2].abs() < 1e-12);
}
