use super::attitude_mekf::{correct_star_tracker, predict, step, AttitudeMEKF, SensorConfig};
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
