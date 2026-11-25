use crate::sensors::kernels;
use crate::sensors::Gyroscope;
fn sensor(limit: f64, orientation: [f64; 4], misalignment: [f64; 4], bias: [f64; 3]) -> Gyroscope {
    Gyroscope::new(
        &[0.; 3],
        &orientation,
        &misalignment,
        limit,
        &[vec![1e-8, 0., 0.], vec![0., 1e-8, 0.], vec![0., 0., 1e-8]],
        &bias,
    )
    .unwrap()
}
#[test]
fn noiseless_rate_mounts_and_biases() {
    let s = sensor(
        100.,
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [0.01, -0.02, 0.03],
    );
    let w = [0.1, 0.2, -0.3];
    assert_eq!(s.noiseless_biasfree_measurement(w), w);
    let biased = s.noiseless_biased_measurement(w);
    for (actual, expected) in biased.into_iter().zip([0.11, 0.18, -0.27]) {
        assert!((actual - expected).abs() < 1e-12);
    }
}
#[test]
fn saturation_is_componentwise() {
    let s = sensor(0.25, [1., 0., 0., 0.], [1., 0., 0., 0.], [0.; 3]);
    assert_eq!(
        s.noiseless_biasfree_measurement([0.5, -0.4, 0.1]),
        [0.25, -0.25, 0.1]
    );
}
#[test]
fn noisy_variants_are_finite_and_jacobians_identity() {
    let s = sensor(100., [1., 0., 0., 0.], [1., 0., 0., 0.], [0.; 3]);
    for x in [
        s.noisy_biasfree_measurement([0.1, 0.2, 0.3]),
        s.noisy_biased_measurement([0.1, 0.2, 0.3]),
    ] {
        assert!(x.iter().all(|x| x.is_finite()))
    }
    assert_eq!(
        s.jacobian_angular_velocity(),
        [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    );
    assert_eq!(
        s.jacobian_bias(),
        [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    );
    assert!(s.availability());
}
#[test]
fn measurement_applies_orientation_and_misalignment() {
    let qz = [
        std::f64::consts::FRAC_1_SQRT_2,
        0.,
        0.,
        std::f64::consts::FRAC_1_SQRT_2,
    ];
    let qx = [
        std::f64::consts::FRAC_1_SQRT_2,
        std::f64::consts::FRAC_1_SQRT_2,
        0.,
        0.,
    ];
    let s = sensor(100., qz, qx, [0.; 3]);
    let w = [0.1, -0.2, 0.3];
    let expected = kernels::rotate_body_to_sensor(w, s.sensor.orientation, s.sensor.misalignment);
    let actual = s.noiseless_biasfree_measurement(w);
    for i in 0..3 {
        assert!((actual[i] - expected[i]).abs() < 1e-12);
    }
}
