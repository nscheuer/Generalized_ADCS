use super::attitude_mekf::{
    correct, correct_gyro, correct_star_tracker, predict, step, AttitudeMEKF, SensorConfig,
};
use super::base::AttitudeEstimator;

fn eye6(scale: f64) -> [[f64; 6]; 6] {
    std::array::from_fn(|i| std::array::from_fn(|j| if i == j { scale } else { 0.0 }))
}

#[test]
fn prediction_propagates_rate_and_continuous_acceleration_noise() {
    let q = [1.0, 0.0, 0.0, 0.0];
    let w = [0.0; 3];
    let p = [[0.0; 6]; 6];
    let psd = [[2e-5, 0.0, 0.0], [0.0, 2e-5, 0.0], [0.0, 0.0, 2e-5]];
    let (q1, w1, propagated) = predict(q, w, p, psd, 2.0).unwrap();
    assert_eq!(q1, q);
    assert_eq!(w1, w);
    assert!((propagated[0][0] - 2e-5 * 8.0 / 3.0).abs() < 1e-16);
    assert!((propagated[3][3] - 2e-5 * 2.0).abs() < 1e-16);
    assert!((propagated[0][3] + 2e-5 * 2.0).abs() < 1e-16);
    assert!((propagated[0][3] - propagated[3][0]).abs() < 1e-16);
}

#[test]
fn tracker_correction_reduces_error_and_covariance() {
    let q = [0.9984379, 0.04997396, -0.02498698, 0.00999479];
    let p = eye6(0.1);
    let r = [[1e-5, 0.0, 0.0], [0.0, 1e-5, 0.0], [0.0, 0.0, 1e-5]];
    let (updated, _, posterior) = correct_star_tracker(
        q,
        [0.0; 3],
        p,
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        r,
    )
    .unwrap();
    let initial_error = q[1..].iter().map(|x| x * x).sum::<f64>().sqrt();
    let updated_error = updated[1..].iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(updated_error < initial_error / 100.0);
    assert!(posterior[0][0] < p[0][0]);
    for i in 0..6 {
        for j in 0..6 {
            assert!((posterior[i][j] - posterior[j][i]).abs() < 1e-14);
        }
    }
}

#[test]
fn mekf_uses_shared_attitude_estimator_state() {
    let base = AttitudeEstimator::new(
        [1.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        eye6(0.1),
        [[1e-8, 0.0, 0.0], [0.0, 1e-8, 0.0], [0.0, 0.0, 1e-8]],
        vec![],
    )
    .unwrap();
    let mekf = AttitudeMEKF::new(base.clone());
    assert_eq!(mekf.base, base);
}

#[test]
fn step_predicts_then_consumes_gyro_readings_and_skips_nan_values() {
    let config = SensorConfig {
        code: vec![0],
        orientation: vec![[1.0, 0.0, 0.0, 0.0]],
        misalignment: vec![[1.0, 0.0, 0.0, 0.0]],
        bias: vec![[0.0; 6]],
        noise_cholesky: vec![std::array::from_fn(|i| {
            std::array::from_fn(|j| if i == j { 1e-3 } else { 0.0 })
        })],
        enabled: vec![true],
        offsets: vec![0],
        ..Default::default()
    };
    let (q, rate, p) = step(
        [1.0, 0.0, 0.0, 0.0],
        [1.0; 3],
        eye6(0.1),
        [[1e-8, 0.0, 0.0], [0.0, 1e-8, 0.0], [0.0, 0.0, 1e-8]],
        0.1,
        &[0.0; 3],
        &config,
    )
    .unwrap();
    assert!(rate.iter().all(|x| x.abs() < 0.01));
    assert!(q.iter().all(|x| x.is_finite()));
    assert!(p.iter().flatten().all(|x| x.is_finite()));
    let (q2, rate2, _) = step(q, rate, p, [[0.0; 3]; 3], 0.1, &[f64::NAN; 3], &config).unwrap();
    assert!(q2.iter().all(|x| x.is_finite()));
    assert_eq!(rate2, rate);
}

#[test]
fn constructor_normalizes_prepares_and_rejects_invalid_inputs() {
    let base = AttitudeEstimator::new(
        [2.0, 0.0, 0.0, 0.0],
        [0.1, -0.2, 0.3],
        eye6(0.25),
        [[1e-4, 0.0, 0.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 1e-4]],
        vec!["estimated_attitude".to_string()],
    )
    .unwrap();
    assert!((base.quaternion[0] - 1.0).abs() < 1e-14);
    let (pq, pw, pp, _) = base.prepare();
    assert_eq!(pq, base.quaternion);
    assert_eq!(pw, base.rate);
    assert_eq!(pp, base.covariance);

    assert!(AttitudeEstimator::new(
        [0.0; 4],
        [0.0; 3],
        eye6(1.0),
        [[1e-4, 0.0, 0.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 1e-4]],
        vec![],
    )
    .is_err());
    assert!(AttitudeEstimator::new(
        [1.0, 0.0, 0.0, 0.0],
        [f64::NAN, 0.0, 0.0],
        eye6(1.0),
        [[1e-4, 0.0, 0.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 1e-4]],
        vec![],
    )
    .is_err());
    let mut asym = eye6(1.0);
    asym[0][1] = 0.5;
    assert!(AttitudeEstimator::new(
        [1.0, 0.0, 0.0, 0.0],
        [0.0; 3],
        asym,
        [[1e-4, 0.0, 0.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 1e-4]],
        vec![],
    )
    .is_err());
    assert!(AttitudeEstimator::new(
        [1.0, 0.0, 0.0, 0.0],
        [0.0; 3],
        eye6(-1.0),
        [[1e-4, 0.0, 0.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 1e-4]],
        vec![],
    )
    .is_err());
    assert!(AttitudeEstimator::new(
        [1.0, 0.0, 0.0, 0.0],
        [0.0; 3],
        eye6(1.0),
        [[1e-4, 0.0, 0.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 1e-4]],
        vec!["unknown_field".to_string()],
    )
    .is_err());
}

#[test]
fn predict_handles_nonzero_rate_zero_dt_and_rejects_negative_dt() {
    let psd = [[1e-3, 0.0, 0.0], [0.0, 1e-3, 0.0], [0.0, 0.0, 1e-3]];
    let p0 = eye6(0.2);
    let w0 = [0.0, 0.0, 0.2];
    let (q0, w_out, p_out) = predict([1.0, 0.0, 0.0, 0.0], w0, p0, psd, 0.0).unwrap();
    assert_eq!(q0, [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(w_out, w0);
    assert_eq!(p_out, p0);

    let (q_turn, _, _) = predict([1.0, 0.0, 0.0, 0.0], w0, p0, psd, 1.0).unwrap();
    assert!((q_turn[0] - 0.1_f64.cos()).abs() < 1e-12);
    assert!((q_turn[3] + 0.1_f64.sin()).abs() < 1e-12);
    assert!(predict([1.0, 0.0, 0.0, 0.0], w0, p0, psd, -0.1).is_err());
}

#[test]
fn direct_correct_and_gyro_with_non_identity_mounting() {
    let mut h = [[0.0; 6]; 3];
    for i in 0..3 {
        h[i][i] = 1.0;
    }
    let r = [[1e-3, 0.0, 0.0], [0.0, 1e-3, 0.0], [0.0, 0.0, 1e-3]];
    let (q1, w1, p1) = correct(
        [1.0, 0.0, 0.0, 0.0],
        [0.0; 3],
        eye6(0.1),
        [0.02, -0.01, 0.03],
        h,
        r,
    )
    .unwrap();
    assert!(q1[1..].iter().map(|x| x * x).sum::<f64>().sqrt() > 0.0);
    assert!((q1.iter().map(|x| x * x).sum::<f64>().sqrt() - 1.0).abs() < 1e-14);
    assert_eq!(w1, [0.0; 3]);
    for i in 0..6 {
        for j in 0..6 {
            assert!((p1[i][j] - p1[j][i]).abs() < 1e-14);
        }
    }
    assert!(correct([1.0, 0.0, 0.0, 0.0], [0.0; 3], eye6(0.0), [0.0; 3], h, [[0.0; 3]; 3]).is_err());

    let mounting = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let r_gyro = [[1e-4, 0.0, 0.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 1e-4]];
    let (qg, wg, pg) = correct_gyro(
        [1.0, 0.0, 0.0, 0.0],
        [0.0; 3],
        eye6(0.5),
        [0.0, 0.4, 0.0],
        mounting,
        r_gyro,
    )
    .unwrap();
    assert_eq!(qg, [1.0, 0.0, 0.0, 0.0]);
    assert!((wg[0] - 0.4).abs() < 1e-3);
    assert!(wg[1].abs() < 1e-12);
    assert!(pg[3][3] < 0.5);
}

#[test]
fn step_skips_disabled_sensors_and_consumes_direction_sensors() {
    let disabled = SensorConfig {
        code: vec![0],
        orientation: vec![[1.0, 0.0, 0.0, 0.0]],
        misalignment: vec![[1.0, 0.0, 0.0, 0.0]],
        bias: vec![[0.0; 6]],
        noise_cholesky: vec![eye6(1e-3)],
        normal: vec![[0.0, 0.0, 1.0]],
        fov: vec![[0.0; 2]],
        scalars: vec![[0.0; 3]],
        enabled: vec![false],
        offsets: vec![0],
        reference: vec![[0.0; 3]],
    };
    let (q_skip, w_skip, _) = step(
        [1.0, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        eye6(0.1),
        [[0.0; 3]; 3],
        0.0,
        &[0.0; 3],
        &disabled,
    )
    .unwrap();
    assert_eq!(q_skip, [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(w_skip, [0.2, 0.0, 0.0]);

    let dir_config = SensorConfig {
        code: vec![1, 3, 4],
        orientation: vec![[1.0, 0.0, 0.0, 0.0]; 3],
        misalignment: vec![[1.0, 0.0, 0.0, 0.0]; 3],
        bias: vec![[0.0; 6]; 3],
        noise_cholesky: vec![eye6(1e-3); 3],
        normal: vec![[0.0, 0.0, 1.0]; 3],
        fov: vec![[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
        scalars: vec![[10.0, 0.0, 0.0], [10.0, 1.0, 1e-3], [0.0; 3]],
        enabled: vec![true; 3],
        offsets: vec![0, 3, 4],
        reference: vec![[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
    };
    let q_init = [0.05_f64.cos(), 0.05_f64.sin(), 0.0, 0.0];
    let (q_dir, _, p_dir) = step(
        q_init,
        [0.0; 3],
        eye6(0.1),
        [[0.0; 3]; 3],
        0.0,
        &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        &dir_config,
    )
    .unwrap();
    let init_err = q_init[1..].iter().map(|x| x * x).sum::<f64>().sqrt();
    let final_err = q_dir[1..].iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(final_err < init_err);
    for i in 0..6 {
        for j in 0..6 {
            assert!((p_dir[i][j] - p_dir[j][i]).abs() < 1e-14);
        }
    }
}
