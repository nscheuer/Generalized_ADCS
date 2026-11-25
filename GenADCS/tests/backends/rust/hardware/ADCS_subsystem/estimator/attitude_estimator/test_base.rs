use super::AttitudeEstimator;

fn base() -> AttitudeEstimator {
    AttitudeEstimator::new(
        [2.0, 0.0, 0.0, 0.0],
        [0.1, -0.2, 0.3],
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 0.1 } else { 0.0 })),
        [[1e-5, 0.0, 0.0], [0.0, 2e-5, 0.0], [0.0, 0.0, 3e-5]],
        vec!["estimated_attitude".to_owned()],
    )
    .unwrap()
}

#[test]
fn normalizes_quaternion_and_preserves_initial_values() {
    let estimator = base();
    assert_eq!(estimator.quaternion, [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(estimator.rate, [0.1, -0.2, 0.3]);
    assert_eq!(estimator.covariance[0][0], 0.1);
    assert_eq!(estimator.acceleration_psd[2][2], 3e-5);
    assert_eq!(estimator.record, vec!["estimated_attitude"]);
}

#[test]
fn prepare_returns_independent_copyable_numeric_state() {
    let estimator = base();
    let (q, w, p, a) = estimator.prepare();
    assert_eq!(q, estimator.quaternion);
    assert_eq!(w, estimator.rate);
    assert_eq!(p, estimator.covariance);
    assert_eq!(a, estimator.acceleration_psd);
}

#[test]
fn rejects_invalid_state_and_unrecognized_diagnostics() {
    let p = std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1.0 } else { 0.0 }));
    let a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    assert!(AttitudeEstimator::new([0.0; 4], [0.0; 3], p, a, vec![]).is_err());
    assert!(AttitudeEstimator::new([f64::NAN, 0.0, 0.0, 1.0], [0.0; 3], p, a, vec![]).is_err());
    assert!(
        AttitudeEstimator::new([1.0, 0.0, 0.0, 0.0], [0.0; 3], p, a, vec!["unknown".into()])
            .is_err()
    );
    let bad_p = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if i == j {
                if i == 0 {
                    -1.0
                } else {
                    1.0
                }
            } else {
                0.0
            }
        })
    });
    assert!(AttitudeEstimator::new([1.0, 0.0, 0.0, 0.0], [0.0; 3], bad_p, a, vec![]).is_err());
}
