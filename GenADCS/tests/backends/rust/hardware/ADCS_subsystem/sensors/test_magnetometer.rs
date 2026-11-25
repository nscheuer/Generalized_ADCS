use crate::sensors::Magnetometer;
fn sensor(limit: f64, bias: [f64; 3]) -> Magnetometer {
    Magnetometer::new(
        &[0.; 3],
        &[1., 0., 0., 0.],
        &[1., 0., 0., 0.],
        limit,
        &[vec![2., 0., 0.], vec![0., 3., 0.], vec![0., 0., 4.]],
        &bias,
    )
    .unwrap()
}
#[test]
fn inertial_measurements_bias_and_saturation() {
    let s = sensor(3e-5, [0.; 3]);
    let field = [5e-5, -4e-5, 1e-5];
    assert_eq!(
        s.noiseless_biasfree_measurement(field, [1., 0., 0., 0.]),
        [3e-5, -3e-5, 1e-5]
    );
}
#[test]
fn noisy_measurements_are_finite_and_bias_jacobian_identity() {
    let s = sensor(1., [1e-6, 2e-6, 3e-6]);
    for x in [
        s.noisy_biasfree_measurement([0.; 3], [1., 0., 0., 0.]),
        s.noisy_biased_measurement([0.; 3], [1., 0., 0., 0.]),
    ] {
        assert!(x.iter().all(|x| x.is_finite()))
    }
    assert_eq!(
        s.jacobian_bias(),
        [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    );
}
#[test]
fn zero_error_jacobians_have_parameterization_scaling() {
    let s = sensor(1., [0.; 3]);
    let field = [2e-5, -3e-5, 4e-5];
    let q = [1., 0., 0., 0.];
    let jq = s.jacobian_relative_quaternion(field, q, q);
    let jm = s.jacobian_relative_mrp(field, q, [0.; 3]);
    let ja = s.jacobian_relative_angle_vector(field, q, [0.; 3]);
    for i in 0..3 {
        for j in 0..3 {
            assert!((jm[i][j] - 2. * jq[i][j]).abs() < 1e-14);
            assert!((ja[i][j] - 0.5 * jq[i][j]).abs() < 1e-14)
        }
    }
}
#[test]
fn disabled_sensor_has_zero_information() {
    let mut s = sensor(1., [0.; 3]);
    s.sensor.enabled = false;
    assert_eq!(
        s.fisher_information_matrix(
            [1., 2., 3.],
            [1., 0., 0., 0.],
            [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        ),
        [[0.; 3]; 3]
    );
}
#[test]
fn enabled_fisher_information_uses_measurement_and_field_covariance() {
    let s = sensor(100., [0.; 3]);
    let info = s.fisher_information_matrix(
        [1., 2., 3.],
        [1., 0., 0., 0.],
        [[0.1, 0., 0.], [0., 0.2, 0.], [0., 0., 0.3]],
    );
    let r1 = 1. / 2.1;
    let r2 = 1. / 3.2;
    let r3 = 1. / 4.3;
    let expected = [
        [9. * r2 + 4. * r3, -2. * r3, -3. * r2],
        [-2. * r3, 9. * r1 + r3, -6. * r1],
        [-3. * r2, -6. * r1, 4. * r1 + r2],
    ];
    for i in 0..3 {
        for j in 0..3 {
            assert!((info[i][j] - expected[i][j]).abs() < 1e-12)
        }
    }
}
