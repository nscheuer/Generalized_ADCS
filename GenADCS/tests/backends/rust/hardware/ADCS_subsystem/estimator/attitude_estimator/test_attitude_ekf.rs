use super::attitude_ekf::{
    correct_gyro, correct_star_tracker, predict, step, step_diagnostics, AttitudeEKF, Mat7,
};
use super::attitude_mekf::SensorConfig;

fn eye7(scale: f64) -> Mat7 {
    std::array::from_fn(|i| std::array::from_fn(|j| if i == j { scale } else { 0.0 }))
}

fn tangent_eye7(q: [f64; 4], scale: f64) -> Mat7 {
    let mut p = eye7(scale);
    for i in 0..4 {
        for j in 0..4 {
            p[i][j] -= scale * q[i] * q[j];
        }
    }
    p
}

#[test]
fn ekf_constructor_validates_and_normalizes_state() {
    let psd = [[1e-5, 0.0, 0.0], [0.0, 1e-5, 0.0], [0.0, 0.0, 1e-5]];
    let ekf = AttitudeEKF::new([2.0, 0.0, 0.0, 0.0], [0.1, -0.2, 0.3], eye7(0.1), psd, vec![])
        .unwrap();
    assert_eq!(ekf.quaternion, [1.0, 0.0, 0.0, 0.0]);
    let (q, w, p, a) = ekf.prepare();
    assert_eq!(q, ekf.quaternion);
    assert_eq!(w, ekf.rate);
    assert_eq!(p, ekf.covariance);
    assert_eq!(a, ekf.acceleration_psd);

    let mut bad_p = eye7(0.1);
    bad_p[0][0] = -1.0;
    assert!(AttitudeEKF::new([1.0, 0.0, 0.0, 0.0], [0.0; 3], bad_p, psd, vec![]).is_err());
    assert!(AttitudeEKF::new([0.0; 4], [0.0; 3], eye7(0.1), psd, vec![]).is_err());
    assert!(
        AttitudeEKF::new([1.0, 0.0, 0.0, 0.0], [0.0; 3], eye7(0.1), psd, vec!["bad".into()])
            .is_err()
    );
}

#[test]
fn ekf_prediction_propagates_quaternion_and_projects_tangent_covariance() {
    let q = [1.0, 0.0, 0.0, 0.0];
    let w = [0.2, -0.1, 0.05];
    let p = tangent_eye7(q, 0.01);
    let psd = [[2e-5, 0.0, 0.0], [0.0, 2e-5, 0.0], [0.0, 0.0, 2e-5]];
    let (q1, w1, p1) = predict(q, w, p, psd, 1.0).unwrap();
    assert_eq!(w1, w);
    let qnorm = q1.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((qnorm - 1.0).abs() < 1e-14);
    // Tangent projection: P * q == 0 in quaternion block
    for i in 0..4 {
        let pq: f64 = (0..4).map(|j| p1[i][j] * q1[j]).sum();
        assert!(pq.abs() < 1e-12);
    }
    for i in 0..7 {
        for j in 0..7 {
            assert!((p1[i][j] - p1[j][i]).abs() < 1e-14);
        }
    }
    assert!(predict(q, w, p, psd, -0.1).is_err());
}

#[test]
fn ekf_gyro_and_star_tracker_corrections_reduce_error() {
    let q = [0.9984379, 0.04997396, -0.02498698, 0.00999479];
    let p = tangent_eye7(q, 0.1);
    let r = [[1e-5, 0.0, 0.0], [0.0, 1e-5, 0.0], [0.0, 0.0, 1e-5]];
    let eye3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    let (_, w_corr, p_gyro) = correct_gyro(q, [0.5, -0.4, 0.3], p, [0.0, 0.0, 0.0], eye3, r).unwrap();
    assert!(w_corr.iter().all(|x| x.abs() < 1e-3));
    assert!(p_gyro[4][4] < p[4][4]);

    let (q_corr, _, p_st) = correct_star_tracker(
        q,
        [0.0; 3],
        p,
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        r,
    )
    .unwrap();
    let initial_err = q[1..].iter().map(|x| x * x).sum::<f64>().sqrt();
    let updated_err = q_corr[1..].iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(updated_err < initial_err / 50.0);
    assert!(p_st[1][1] < p[1][1]);
}

#[test]
fn ekf_step_and_diagnostics_fuse_gyro_and_tracker_and_skip_nan() {
    let mut chol = [[0.0; 6]; 6];
    for i in 0..3 {
        chol[i][i] = 1e-3;
    }
    let config = SensorConfig {
        code: vec![0, 2],
        orientation: vec![[1.0, 0.0, 0.0, 0.0]; 2],
        misalignment: vec![[1.0, 0.0, 0.0, 0.0]; 2],
        bias: vec![[0.0; 6]; 2],
        noise_cholesky: vec![chol; 2],
        enabled: vec![true, true],
        offsets: vec![0, 3],
        ..Default::default()
    };
    let psd = [[1e-8, 0.0, 0.0], [0.0, 1e-8, 0.0], [0.0, 0.0, 1e-8]];
    let readings = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let (q, w, p) = step(
        [0.9984379, 0.04997396, -0.02498698, 0.00999479],
        [0.5, -0.3, 0.2],
        eye7(0.1),
        psd,
        0.1,
        &readings,
        &config,
    )
    .unwrap();
    assert!(w.iter().all(|x| x.abs() < 0.01));
    assert!(q[1..].iter().all(|x| x.abs() < 0.01));
    assert!(p.iter().flatten().all(|x| x.is_finite()));

    let mut ekf = AttitudeEKF::new(
        [0.9984379, 0.04997396, -0.02498698, 0.00999479],
        [0.5, -0.3, 0.2],
        eye7(0.1),
        psd,
        vec![],
    )
    .unwrap();
    let truth = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let diag = step_diagnostics(&mut ekf, 0.1, &readings, &config, &truth, true).unwrap();
    assert!(diag[68].is_finite() && diag[68] >= 0.0); // NIS
    assert!(diag[69].is_finite() && diag[69] >= 0.0); // NEES
    assert!(diag[91..109].iter().all(|x| x.is_finite()));

    let nan_readings = [f64::NAN; 7];
    let (q2, w2, _) = step(q, w, p, [[0.0; 3]; 3], 0.1, &nan_readings, &config).unwrap();
    assert_eq!(w2, w);
    assert!(q2.iter().all(|x| x.is_finite()));
}
