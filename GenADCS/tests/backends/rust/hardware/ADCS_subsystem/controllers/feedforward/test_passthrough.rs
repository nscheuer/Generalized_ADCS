use crate::controllers::feedforward::{passthrough_step, Feedforward, Passthrough};

#[test]
fn test_passthrough_forwards_goal_and_handles_nan() {
    let ff = Passthrough;
    assert_eq!(ff.to_params().len(), 1);

    let q_cur = [1.0, 0.0, 0.0, 0.0];
    let q_g = [0.0, 1.0, 0.0, 0.0];
    let w_g = [0.01, -0.02, 0.03];
    let (qd, wd, wdd, fft) = passthrough_step(q_g, w_g, q_cur);
    for i in 0..4 {
        assert!((qd[i] - q_g[i]).abs() < 1e-12);
    }
    for i in 0..3 {
        assert!((wd[i] - w_g[i]).abs() < 1e-12);
        assert!(wdd[i].abs() < 1e-12);
        assert!(fft[i].abs() < 1e-12);
    }

    let nan_q = [f64::NAN; 4];
    let nan_w = [f64::NAN; 3];
    let (qd2, wd2, _, _) = passthrough_step(nan_q, nan_w, q_cur);
    for i in 0..4 {
        assert!((qd2[i] - q_cur[i]).abs() < 1e-12);
    }
    for v in wd2 {
        assert!(v.abs() < 1e-12);
    }
}
