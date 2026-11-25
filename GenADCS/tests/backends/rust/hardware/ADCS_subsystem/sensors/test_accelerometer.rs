use crate::sensors::kernels;
use crate::sensors::Accelerometer;

fn sensor(limit: f64, bias: [f64; 3]) -> Accelerometer {
    Accelerometer::new(
        &[0.; 3],
        &[1., 0., 0., 0.],
        &[1., 0., 0., 0.],
        limit,
        &[vec![1e-6, 0., 0.], vec![0., 1e-6, 0.], vec![0., 0., 1e-6]],
        &bias,
    )
    .unwrap()
}
fn close(a: [f64; 3], b: [f64; 3]) {
    for i in 0..3 {
        assert!((a[i] - b[i]).abs() < 1e-10, "{a:?} != {b:?}")
    }
}

#[test]
fn specific_force_subtracts_gravity() {
    let s = sensor(100., [0.; 3]);
    close(
        s.noiseless_biasfree_measurement([2., 3., 4.], [0.5, 1., -1.], [1., 0., 0., 0.]),
        [1.5, 2., 5.],
    );
}
#[test]
fn specific_force_is_rotated_through_attitude_and_mounting() {
    let qm = [0.9689124217, 0., 0., 0.2474039593];
    let qmis = [0.9950041653, -0.0998334166, 0., 0.];
    let qa = [0.9887710779, 0., 0.1494381325, 0.];
    let a = [2., 3., 4.];
    let g = [0.5, 1., -1.];
    let s = Accelerometer::new(
        &[0.; 3],
        &qm,
        &qmis,
        100.,
        &[vec![1e-6, 0., 0.], vec![0., 1e-6, 0.], vec![0., 0., 1e-6]],
        &[0.; 3],
    )
    .unwrap();
    let expected = kernels::rotate_inertial_to_sensor(
        [1.5, 2., 5.],
        qa,
        s.sensor.orientation,
        s.sensor.misalignment,
    );
    let actual = s.noiseless_biasfree_measurement(a, g, qa);
    for i in 0..3 {
        assert!((actual[i] - expected[i]).abs() < 1e-9)
    }
}
#[test]
fn bias_and_noise_variants_are_available() {
    let b = [0.1, -0.2, 0.3];
    let s = sensor(100., b);
    let a = [2., 3., 4.];
    let g = [1.; 3];
    let q = [1., 0., 0., 0.];
    let base = s.noiseless_biasfree_measurement(a, g, q);
    close(
        s.noiseless_biased_measurement(a, g, q),
        [base[0] + b[0], base[1] + b[1], base[2] + b[2]],
    );
    for x in [
        s.noisy_biasfree_measurement(a, g, q),
        s.noisy_biased_measurement(a, g, q),
    ] {
        assert!(x.iter().all(|x| x.is_finite()))
    }
    assert_eq!(
        s.jacobian_bias(),
        [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    );
}
#[test]
fn zero_error_attitude_jacobian_scaling_matches_parameterizations() {
    let s = sensor(100., [0.; 3]);
    let a = [2., 3., 4.];
    let g = [0.5, 1., -1.];
    let q = [1., 0., 0., 0.];
    let jq = s.jacobian_relative_quaternion(a, g, q, q);
    let jm = s.jacobian_relative_mrp(a, g, q, [0.; 3]);
    let ja = s.jacobian_relative_angle_vector(a, g, q, [0.; 3]);
    for i in 0..3 {
        for j in 0..3 {
            assert!((jm[i][j] - 2. * jq[i][j]).abs() < 1e-10);
            assert!((ja[i][j] - 0.5 * jq[i][j]).abs() < 1e-10)
        }
    }
}
#[test]
fn saturation_and_availability() {
    let mut s = sensor(0.5, [0.; 3]);
    close(
        s.noiseless_biasfree_measurement([1.; 3], [0.; 3], [1., 0., 0., 0.]),
        [0.5; 3],
    );
    assert!(s.availability());
    s.sensor.enabled = false;
    assert!(!s.availability());
}
#[test]
fn nonzero_attitude_jacobians_are_finite() {
    let s = sensor(100., [0.; 3]);
    let q = [1., 0., 0., 0.];
    let a = [1.; 3];
    let g = [0.1, 0.2, 0.3];
    let values = [
        s.jacobian_relative_quaternion(a, g, q, [0.9987502604, 0.0499791693, 0., 0.]),
        s.jacobian_relative_mrp(a, g, q, [0.01, -0.02, 0.03]),
        s.jacobian_relative_angle_vector(a, g, q, [0.01, -0.02, 0.03]),
    ];
    assert!(values.iter().flatten().flatten().all(|v| v.is_finite()));
}
